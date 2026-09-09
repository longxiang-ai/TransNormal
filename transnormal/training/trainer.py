"""Formal TransNormal training with resumable distributed state."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .checkpointing import export_model, load_checkpoint, recipe_signature, save_checkpoint
from .data import MixtureBatchSampler, TrainingDataset
from .data.index import KINDS
from .initialization import initialize
from .losses import latent_mask, masked_mse, wavelet_loss


def read_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    expected = {
        "model": {"initial_model", "initial_revision", "dino_model", "dino_revision", "local_files_only", "vae_model"},
        "data": {*KINDS, "probabilities", "resolutions", "random_flip"},
        "training": {"output_dir", "seed", "timestep", "max_steps", "batch_size", "workers", "learning_rate",
                     "projector_learning_rate", "warmup_steps", "adam_betas", "weight_decay", "adam_epsilon",
                     "max_grad_norm", "wavelet_weight", "mixed_precision", "checkpoint_every", "gradient_checkpointing"},
    }
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError("Config must contain exactly model, data and training sections.")
    for section, keys in expected.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"Missing or unknown fields in {section} configuration.")
    data, train = config["data"], config["training"]
    probabilities = data["probabilities"]
    if len(probabilities) != 4 or any(p <= 0 for p in probabilities) or not math.isclose(sum(probabilities), 1):
        raise ValueError("Provide four positive dataset probabilities summing to one.")
    if len(data["resolutions"]) != 4 or any(r < 16 for r in data["resolutions"]):
        raise ValueError("Provide one valid resolution per dataset.")
    if data["resolutions"][3] != 352:
        raise ValueError("Virtual KITTI uses the fixed 352 x 1216 benchmark crop.")
    for key in ("max_steps", "batch_size", "checkpoint_every"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer.")
    if train["mixed_precision"] not in ("bf16", "no") or train["workers"] < 0:
        raise ValueError("Use bf16 or no mixed precision, and a nonnegative worker count.")
    return config


def loss_terms(prediction, changed_prediction, target, rgb_latents, normals, mask, vae, dtype):
    batch_size = target.shape[0]
    prediction_normal = prediction[:batch_size]
    valid = latent_mask(mask)
    if valid.shape != target.shape:
        raise ValueError("Latent mask and VAE shapes differ; check dataset image dimensions.")
    terms = {
        "normal": masked_mse(prediction_normal, target, valid),
        "rgb": F.mse_loss(prediction[batch_size:].float(), rgb_latents.float()),
        "consistency": prediction_normal.float().sum() * 0.0,
    }
    if changed_prediction is not None:
        # Both views receive gradients; material changes preserve the geometry.
        terms["consistency"] = masked_mse(changed_prediction, prediction_normal, valid)
    decoded = vae.decode((prediction_normal / vae.config.scaling_factor).to(dtype), return_dict=False)[0]
    if decoded.shape != normals.shape:
        raise ValueError("Decoded normals and target dimensions differ.")
    terms["wavelet"] = wavelet_loss(decoded, normals, mask)
    return terms


def check_finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; stopping before the optimizer update.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", help="A checkpoint directory created by this trainer")
    parser.add_argument("--stop-after", type=int, help="Stop after this absolute step without changing the LR schedule")
    parser.add_argument("--smoke-test", action="store_true", help="Exercise all four datasets in a six-step training run")
    parser.add_argument("--check-data", action="store_true", help="Validate one real batch per dataset on CPU and exit")
    args = parser.parse_args()
    config = read_config(args.config)
    train, data = config["training"], config["data"]
    if args.smoke_test:
        train["max_steps"] = 6
    manifests = [data[kind] for kind in KINDS]
    dataset = TrainingDataset(manifests, data["resolutions"], data["random_flip"])
    if dataset.kinds != list(KINDS):
        raise ValueError("Dataset indices are assigned to the wrong configuration fields.")
    if args.check_data:
        for i, kind in enumerate(KINDS):
            batch = dataset[(i, 0, train["seed"])]
            print(json.dumps({"dataset": kind, "samples": dataset.lengths[i],
                              "shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}}))
        return
    accelerator = Accelerator(mixed_precision=train["mixed_precision"], step_scheduler_with_optimizer=False)
    if accelerator.device.type != "cuda" or accelerator.num_processes != 8:
        raise RuntimeError("The formal training recipe and GPU smoke test require an eight-process CUDA launch.")
    if train["mixed_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16 training.")
    set_seed(train["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if train["mixed_precision"] == "bf16" else torch.float32
    output = Path(train["output_dir"])
    # A fresh run must not append to or overwrite an existing training run.
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output}; use --resume or choose a new directory.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    accelerator.wait_for_everyone()
    model, vae, dino, scheduler = initialize(config["model"], accelerator.device, dtype)
    if train["gradient_checkpointing"]:
        model.unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        [{"params": model.unet.parameters(), "lr": train["learning_rate"]},
         {"params": model.projector.parameters(), "lr": train["projector_learning_rate"]}],
        betas=tuple(train["adam_betas"]), weight_decay=train["weight_decay"], eps=train["adam_epsilon"],
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / max(1, train["warmup_steps"])) if train["warmup_steps"] else 1.0,
    )
    frozen = list(vae.parameters()) + list(dino.dino_backbone.parameters())
    frozen_versions = [p._version for p in frozen]
    trainable_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in trainable_ids or p.requires_grad for p in frozen)
    assert {id(p) for p in model.parameters() if p.requires_grad} == trainable_ids
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    signature = recipe_signature(config, accelerator.num_processes, args.smoke_test, manifests)
    step = load_checkpoint(accelerator, args.resume, signature) if args.resume else 0
    start = step
    end = min(train["max_steps"], args.stop_after if args.stop_after is not None else train["max_steps"])
    if not start < end:
        raise ValueError("The requested stop step must be greater than the resumed step.")
    sampler = MixtureBatchSampler(dataset.lengths, data["probabilities"], train["batch_size"],
                                  accelerator.process_index, accelerator.num_processes, train["seed"], end, start, args.smoke_test)
    # This sampler already shards global batches. Do not shard this loader again.
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=train["workers"], pin_memory=True,
                        generator=torch.Generator().manual_seed(train["seed"]),
                        **({"prefetch_factor": 2} if train["workers"] else {}))
    unwrapped = accelerator.unwrap_model(model)
    model.train()
    updated_unet = updated_projector = False
    last_checkpoint = None
    for batch in loader:
        kind = batch.pop("dataset_name")[0]
        batch = {key: value.to(accelerator.device, non_blocking=True) for key, value in batch.items()}
        images, normals = batch["pixel_values"].to(dtype), batch["normal_values"].to(dtype)
        with torch.no_grad():
            rgb_latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            target = vae.encode(normals).latent_dist.sample() * vae.config.scaling_factor
            tokens = dino.extract_patch_tokens(images)
            changed_latents = changed_tokens = None
            if "rgb_change_values" in batch:
                change = batch["rgb_change_values"].to(dtype)
                changed_latents = vae.encode(change).latent_dist.sample() * vae.config.scaling_factor
                changed_tokens = dino.extract_patch_tokens(change)
        for name, value in (("DINO features", tokens), ("normal target", target)):
            check_finite(value, name)
        with accelerator.autocast():
            prediction, changed_prediction = model(rgb_latents, tokens, train["timestep"], changed_latents, changed_tokens)
            terms = loss_terms(prediction, changed_prediction, target, rgb_latents, normals,
                               batch["valid_mask_values"], vae, dtype)
            total = terms["normal"] + terms["rgb"] + terms["consistency"] + train["wavelet_weight"] * terms["wavelet"]
        check_finite(total, "loss")
        accelerator.backward(total)
        unet_norm = accelerator.clip_grad_norm_(unwrapped.unet.parameters(), train["max_grad_norm"])
        projector_norm = torch.nn.utils.clip_grad_norm_(unwrapped.projector.parameters(), float("inf"))
        check_finite(unet_norm, "UNet gradients")
        check_finite(projector_norm, "projector gradients")
        if args.smoke_test:
            if unet_norm.item() == 0 or projector_norm.item() == 0:
                raise RuntimeError("Smoke test found a disconnected trainable component.")
            before_unet = unwrapped.unet.conv_out.weight.detach().clone()
            before_projector = unwrapped.projector.weight.detach().clone()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        report = {key: accelerator.reduce(value.detach().float(), reduction="mean").item() for key, value in terms.items()}
        report.update(step=step, dataset=kind, unet_grad_norm=unet_norm.item(), projector_grad_norm=projector_norm.item())
        if args.smoke_test:
            updated_unet |= not torch.equal(before_unet, unwrapped.unet.conv_out.weight)
            updated_projector |= not torch.equal(before_projector, unwrapped.projector.weight)
            assert all(p.grad is None and not p.requires_grad and p._version == version for p, version in zip(frozen, frozen_versions))
            # Verify the complete small projector, not only its scalar norm, across ranks.
            reference = unwrapped.projector.weight.detach().clone()
            dist.broadcast(reference, src=0)
            if not torch.equal(reference, unwrapped.projector.weight):
                raise RuntimeError("Projector parameters diverged across distributed ranks.")
        if accelerator.is_main_process:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
        if step % train["checkpoint_every"] == 0 or step == end:
            last_checkpoint = save_checkpoint(accelerator, output, step, signature)
    if args.smoke_test and not (updated_unet and updated_projector):
        raise RuntimeError("Smoke test did not update both UNet and projector parameters.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        export_model(output / f"export-{step:06d}", unwrapped, vae, scheduler)
        report = {"status": "passed", "start_step": start, "end_step": step, "world_size": accelerator.num_processes,
                  "smoke_test": args.smoke_test, "checkpoint": last_checkpoint.name,
                  "updated_unet": updated_unet if args.smoke_test else None,
                  "updated_projector": updated_projector if args.smoke_test else None}
        (output / f"completion-{step:06d}.json").write_text(json.dumps(report, indent=2) + "\n")
    accelerator.wait_for_everyone()
    accelerator.end_training()
