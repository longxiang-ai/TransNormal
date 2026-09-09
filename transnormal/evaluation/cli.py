"""Run the paper's fixed normal benchmarks with a common inference and scoring path."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import Benchmark, SPLITS, pad_image
from .metrics import normal_errors, summarize
from transnormal.utils import resize_back, get_tv_resample_method

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/evaluation.json"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng(path, cuda):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if cuda:
        state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def restore_rng(path, cuda):
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if cuda:
        torch.cuda.set_rng_state_all(state["cuda"])


def predict(pipe, image):
    image, (top, left, height, width) = pad_image(image)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = pipe(image, timestep=999, processing_res=768, match_input_res=False,
                      output_type="pt", input_is_normalized=True)
        output = resize_back(output, image.shape[-2:], get_tv_resample_method("nearest"))
        output = (output * 2 - 1).float().cpu()
    return output[:, :, top:top + height, left:left + width]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *SPLITS], default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON dataset roots; paths are relative to the working directory")
    parser.add_argument("--data", help="Override the root for one selected dataset")
    parser.add_argument("--split", type=Path, help="Custom split for one dataset; recorded separately from the supplied paper split")
    parser.add_argument("--transnormal-index", type=Path, help="Optional scene/view or tar index for TransNormal-Synthetic evaluation")
    parser.add_argument("--output", type=Path, help="New directory for per-dataset results and summary.csv")
    parser.add_argument("--model-path", help="Released TransNormal weights or a training export")
    parser.add_argument("--dino-path", help="DINOv3 ViT-H+/16 component")
    parser.add_argument("--projector-path", help="Defaults to cross_attention_projector.pt in the model directory")
    parser.add_argument("--prediction-dir", type=Path, help="Score precomputed floating-point .npy normals without model inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng-state", type=Path, help="Trusted initial_rng.pt for one dataset, or a results directory for --dataset all")
    parser.add_argument("--check-data", action="store_true", help="Decode and validate every selected sample without loading a model")
    parser.add_argument("--limit", type=int, help="Smoke test only: first N samples per dataset; results are marked as partial")
    args = parser.parse_args()
    if args.dataset == "all" and (args.data or args.split):
        parser.error("--data and --split require one selected dataset; edit --config for all datasets")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    settings = json.loads(args.config.read_text())
    names = list(SPLITS) if args.dataset == "all" else [args.dataset]
    datasets = []
    for name in names:
        entry = settings[name]
        index = (args.transnormal_index or entry.get("index")) if name == "transnormal" else None
        datasets.append(Benchmark(name, args.data or entry["root"], split=args.split, index=index))
    if args.output and args.output.exists():
        raise FileExistsError("Choose a new output directory; previous results are never overwritten.")
    if not args.check_data and not args.output:
        parser.error("--output is required unless --check-data is selected")
    if not args.check_data and not args.prediction_dir and not (args.model_path and args.dino_path):
        parser.error("Provide --model-path and --dino-path, or --prediction-dir")
    pipe = None
    if not args.check_data and not args.prediction_dir:
        from transnormal import TransNormalPipeline, create_dino_encoder
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("Model inference requires CUDA; precomputed scoring and data checks support CPU.")
        seed_all(args.seed)
        dtype = torch.bfloat16
        projector = args.projector_path or str(Path(args.model_path) / "cross_attention_projector.pt")
        dino = create_dino_encoder(model_name="dinov3_vith16plus", weights_path=args.dino_path,
                                  projector_path=projector, device=args.device, dtype=dtype)
        pipe = TransNormalPipeline.from_pretrained(args.model_path, dino_encoder=dino, torch_dtype=dtype).to(args.device)
    if args.output:
        args.output.mkdir(parents=True)
    results = []
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "Pillow")}
    if pipe is not None:
        versions.update({name: importlib.metadata.version(name) for name in ("diffusers", "transformers")})
    for dataset in datasets:
        keys = dataset.keys[:args.limit] if args.limit else dataset.keys
        destination = args.output / dataset.name if args.output else None
        record = {"dataset": dataset.name, "samples": len(keys), "full_split_samples": len(dataset.keys),
                  "is_full_paper_split": args.split is None and len(keys) == len(dataset.keys),
                  "split_sha256": hashlib.sha256(dataset.split.read_bytes()).hexdigest(),
                  "selected_keys_sha256": hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest(),
                  "seed": args.seed, "versions": versions}
        if dataset.index:
            record["data_index_sha256"] = hashlib.sha256(dataset.index.read_bytes()).hexdigest()
        if args.check_data:
            pixels, sizes = 0, {}
            for key in keys:
                image, _, mask = dataset.load(key)
                pixels += int(mask.sum())
                shape = "x".join(map(str, image.shape[-2:]))
                sizes[shape] = sizes.get(shape, 0) + 1
            record.update(status="data-valid", valid_pixels=pixels, dimensions=sizes)
        else:
            destination.mkdir(parents=True)
            # Reset after model loading: all-dataset and individual commands share the same per-dataset RNG.
            seed_all(args.seed)
            if args.rng_state:
                state_path = args.rng_state / dataset.name / "initial_rng.pt" if args.dataset == "all" else args.rng_state
                restore_rng(state_path, pipe is not None)
            save_rng(destination / "initial_rng.pt", pipe is not None)
            errors, per_image = [], []
            for number, key in enumerate(keys, 1):
                image, target, mask = dataset.load(key)
                if pipe is not None:
                    prediction = predict(pipe, image)
                    path = destination / "predictions" / Path(key).with_suffix(".npy")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, prediction[0].permute(1, 2, 0).numpy())
                else:
                    base = args.prediction_dir / dataset.name / "predictions" if args.dataset == "all" else args.prediction_dir
                    values = np.load(base / Path(key).with_suffix(".npy"), allow_pickle=False)
                    if values.ndim != 3 or values.shape[-1] != 3 or not np.issubdtype(values.dtype, np.floating):
                        raise ValueError(f"Expected a floating-point HxWx3 prediction for {key}")
                    prediction = torch.from_numpy(values).permute(2, 0, 1)[None]
                current = normal_errors(prediction, target, mask)
                errors.append(current)
                per_image.append({"sample": key, "pixels": current.numel(), **summarize(current)})
                print(f"{dataset.name}: {number}/{len(keys)}", flush=True)
            record.update(status="evaluated", valid_pixels=sum(x.numel() for x in errors),
                          weight_dtype="bfloat16" if pipe is not None else None,
                          autocast_dtype="float16" if pipe is not None else None, metric_dtype="float32",
                          rng_restored=args.rng_state is not None, timestep=999 if pipe is not None else None,
                          processing_res=768 if pipe is not None else None,
                          aggregation="all masked pixels pooled across the split", **summarize(torch.cat(errors)))
            (destination / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
            (destination / "per_image.json").write_text(json.dumps(per_image, indent=2) + "\n")
        results.append(record)
        print(json.dumps(record, indent=2), flush=True)
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        if not args.check_data:
            columns = ["dataset", "samples", "valid_pixels", "is_full_paper_split", "mean", "median", "rmse", "a1", "a2", "a3", "a4", "a5"]
            with (args.output / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
