#!/usr/bin/env python
"""Estimate surface normals for one image or a directory of images."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

from transnormal import TransNormalPipeline, create_dino_encoder

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--image", "-i", type=Path, required=True, help="RGB image or image directory")
    parser.add_argument("--output", "-o", type=Path, help="Output file for an image, or output directory for a directory")
    parser.add_argument("--model_path", type=Path, default=Path("weights/transnormal"))
    parser.add_argument("--dino_path", type=Path, default=Path("weights/dinov3_vith16plus"))
    parser.add_argument("--projector_path", type=Path, help="Defaults to cross_attention_projector.pt in --model_path")
    parser.add_argument("--processing_res", type=int, default=768, help="Processing resolution; 0 keeps input resolution")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories and preserve their layout")
    parser.add_argument("--output_suffix", default="_normal", help="Suffix for directory output filenames")
    parser.add_argument("--output_format", choices=["png", "jpg", "npz"], help="Directory output format (default: png); single images infer it from --output")
    parser.add_argument("--save_comparison", "--save_side_by_side", action="store_true", help="Save RGB and normal side by side")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args(argv)
    if args.processing_res < 0:
        parser.error("--processing_res must be nonnegative")
    if any(separator in args.output_suffix for separator in ("/", "\\")):
        parser.error("--output_suffix must be a filename suffix")
    return args


def output_jobs(args):
    source = args.input.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        output = (args.output or Path("normal_output." + (args.output_format or "png"))).resolve()
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported input image: {source}")
        jobs = [(source, output)]
    elif source.is_dir():
        output = (args.output or Path("normal_outputs")).resolve()
        if output == source:
            raise ValueError("Use a separate output directory")
        if output.exists() and not output.is_dir():
            raise ValueError(f"Output must be a directory: {output}")
        images = source.rglob("*") if args.recursive else source.iterdir()
        jobs = []
        for image in sorted(images):
            if not image.is_file() or image.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if output.is_relative_to(source) and image.is_relative_to(output):
                continue
            relative = image.relative_to(source)
            destination = output / relative.parent / (image.stem + args.output_suffix + "." + (args.output_format or "png"))
            jobs.append((image, destination))
        if not jobs:
            raise ValueError(f"No supported images found in {source}")
    else:
        raise ValueError(f"Expected an image file or directory: {source}")
    destinations = set()
    for image, output in jobs:
        if image.resolve() == output.resolve():
            raise ValueError(f"Output would replace the input: {image}")
        if output in destinations:
            raise ValueError(f"Multiple inputs map to {output}; give the input images distinct stems")
        destinations.add(output)
        extension = output.suffix.lower().lstrip(".")
        output_format = "jpg" if extension == "jpeg" else extension
        if output_format not in ("png", "jpg", "npz"):
            raise ValueError("Output files must use .png, .jpg, .jpeg or .npz")
        if args.output_format and output_format != args.output_format:
            raise ValueError("--output_format must match the output filename extension")
        if args.save_comparison and output_format == "npz":
            raise ValueError("--save_comparison requires PNG or JPEG output")
    return jobs


def load_pipeline(args):
    projector = args.projector_path or args.model_path / "cross_attention_projector.pt"
    for path in (args.model_path, args.dino_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")
    if not projector.is_file():
        raise FileNotFoundError(f"Projector weights not found: {projector}")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    dino = create_dino_encoder(model_name="dinov3_vith16plus", cross_attention_dim=1024,
                               weights_path=str(args.dino_path), projector_path=str(projector),
                               device=args.device, dtype=dtype, freeze_encoder=True)
    pipe = TransNormalPipeline.from_pretrained(str(args.model_path), dino_encoder=dino,
                                               torch_dtype=dtype, safety_checker=None)
    return pipe.to(args.device)


@torch.no_grad()
def predict_and_save(pipe, image_path, output_path, args):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    raw = output_path.suffix.lower() == ".npz"
    normal = pipe(image=image, processing_res=args.processing_res, output_type="np" if raw else "pil")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if raw:
        with output_path.open("wb") as handle:
            np.savez_compressed(handle, normal=normal)
    else:
        if args.save_comparison:
            image = image.resize(normal.size)
            combined = Image.new("RGB", (normal.width * 2, normal.height))
            combined.paste(image, (0, 0))
            combined.paste(normal, (normal.width, 0))
            normal = combined
        normal.save(output_path)


def main(argv=None):
    args = parse_args(argv)
    jobs = output_jobs(args)
    pending = [(source, destination) for source, destination in jobs
               if not (args.skip_existing and destination.is_file())]
    skipped = len(jobs) - len(pending)
    if not pending:
        print(f"Skipped {skipped} existing outputs")
        return 0
    print(f"Loading model for {len(pending)} image(s)")
    pipe = load_pipeline(args)
    errors = 0
    for number, (source, destination) in enumerate(pending, 1):
        try:
            predict_and_save(pipe, source, destination, args)
            print(f"[{number}/{len(pending)}] {destination}")
        except Exception as error:
            errors += 1
            print(f"Error processing {source}: {error}", file=sys.stderr)
    print(f"Processed: {len(pending) - errors}; skipped: {skipped}; errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
