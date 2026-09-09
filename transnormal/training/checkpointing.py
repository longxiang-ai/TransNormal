"""Private resumable state and portable inference exports are kept separate."""

import hashlib
import json
from pathlib import Path

import torch


def recipe_signature(config, world_size, smoke, manifests):
    recipe = json.loads(json.dumps(config))
    recipe["training"].pop("output_dir")
    recipe["training"].pop("workers")
    recipe["index_sha256"] = [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in manifests]
    recipe["world_size"], recipe["smoke_test"] = world_size, smoke
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def save_checkpoint(accelerator, output, step, signature):
    path = Path(output) / f"checkpoint-{step:06d}"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    accelerator.wait_for_everyone()
    accelerator.save_state(str(path))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (path / "progress.json").write_text(json.dumps({"step": step, "signature": signature}, indent=2) + "\n")
    accelerator.wait_for_everyone()
    return path


def load_checkpoint(accelerator, path, signature):
    metadata = json.loads((Path(path) / "progress.json").read_text())
    if metadata["signature"] != signature:
        raise ValueError("Resume recipe, dataset indices, or process count differ from the saved run.")
    # Load only training states that you created or trust.
    accelerator.load_state(str(path))
    return int(metadata["step"])


def export_model(path, model, vae, scheduler):
    from transnormal.pipeline import TransNormalPipeline

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite export: {path}")
    pipeline = TransNormalPipeline(vae=vae, unet=model.unet, scheduler=scheduler,
                                   text_encoder=None, tokenizer=None, dino_encoder=None)
    pipeline.save_pretrained(path, safe_serialization=True)
    projector = {key: value.detach().cpu().clone() for key, value in model.projector.state_dict().items()}
    torch.save(projector, path / "cross_attention_projector.pt")
    # Diffusers may carry the original local load path into saved component configs.
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in ("_name_or_path", "_commit_hash")}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    for config in path.rglob("*.json"):
        config.write_text(json.dumps(clean(json.loads(config.read_text())), indent=2) + "\n")
