"""Export repeatable TransNormal PNG priors for the eight TransLab scenes in TSGS."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch
from transnormal import TransNormalPipeline, create_dino_encoder
from transnormal.evaluation.cli import restore_rng, save_rng, seed_all

SCENES = [f"scene_{i:02d}" for i in range(1, 9)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, outside the input dataset")
    parser.add_argument("--scene", choices=SCENES, help="Default: all eight scenes")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dino-path", required=True)
    parser.add_argument("--projector-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng-state", type=Path, help="Prior export directory containing each scene's initial_rng.pt")
    parser.add_argument("--limit", type=int, help="First N images per scene for a partial smoke test")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        raise FileExistsError("Choose a new export directory.")
    if args.output.resolve().is_relative_to(args.data.resolve()):
        raise ValueError("Export outside the dataset, then copy the completed priors into TSGS.")
    scenes = [args.scene] if args.scene else SCENES
    inputs = {}
    for scene in scenes:
        folder = args.data / scene / "images"
        paths = sorted(folder.glob("*.png")) or sorted(folder.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No RGB images: {folder}")
        inputs[scene] = paths
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Normal export requires CUDA.")
    seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    projector = args.projector_path or str(Path(args.model_path) / "cross_attention_projector.pt")
    dino = create_dino_encoder(model_name="dinov3_vith16plus", weights_path=args.dino_path,
                              projector_path=projector, device=args.device, dtype=dtype)
    pipe = TransNormalPipeline.from_pretrained(args.model_path, dino_encoder=dino,
                                               torch_dtype=dtype).to(args.device)
    args.output.mkdir(parents=True)
    for scene, paths in inputs.items():
        destination = args.output / scene
        (destination / "transnormals").mkdir(parents=True)
        seed_all(args.seed)
        if args.rng_state:
            restore_rng(args.rng_state / scene / "initial_rng.pt", True)
        save_rng(destination / "initial_rng.pt", True)
        selected = paths[:args.limit] if args.limit else paths
        files = []
        for path in selected:
            with Image.open(path) as source:
                normal = pipe(source.convert("RGB"), timestep=999, processing_res=768, output_type="pil")
            output = destination / "transnormals" / f"{path.stem}_normal.png"
            normal.save(output)
            files.append({"image": path.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "normal": output.name, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        receipt = {"scene": scene, "images": len(selected), "total_images": len(paths),
                   "complete_scene": len(selected) == len(paths), "seed": args.seed,
                   "rng_restored": args.rng_state is not None, "dtype": args.dtype,
                   "processing_res": 768, "timestep": 999, "torch": torch.__version__, "files": files}
        (destination / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{scene}: exported {len(selected)}/{len(paths)} images", flush=True)


if __name__ == "__main__":
    main()
