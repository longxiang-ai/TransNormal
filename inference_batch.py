#!/usr/bin/env python
"""
TransNormal Batch Inference Script

This script processes multiple images in a directory and saves normal maps.

Usage:
    python inference_batch.py --input_dir path/to/images --output_dir path/to/output

    # With custom settings
    python inference_batch.py \
        --input_dir ./examples/input \
        --output_dir ./examples/output \
        --model_path ./weights/transnormal \
        --dino_path ./weights/dinov3_vith16plus \
        --projector_path ./weights/cross_attention_projector.pt
"""

import argparse
import os
import glob
from tqdm import tqdm
import torch
from PIL import Image

from transnormal import TransNormalPipeline, create_dino_encoder, save_normal_map


SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="TransNormal Batch Inference: Process multiple images"
    )
    
    # Input/Output
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Path to input directory containing images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Path to output directory for normal maps"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_normal",
        help="Suffix to add to output filenames (default: _normal)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "jpg", "npz"],
        help="Output format (default: png)"
    )
    
    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="./weights/transnormal",
        help="Path to TransNormal model weights"
    )
    parser.add_argument(
        "--dino_path",
        type=str,
        default="./weights/dinov3_vith16plus",
        help="Path to DINOv3 pretrained weights"
    )
    parser.add_argument(
        "--projector_path",
        type=str,
        default="./weights/transnormal/cross_attention_projector.pt",
        help="Path to cross-attention projector weights"
    )
    
    # Inference settings
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Processing resolution (default: 768)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Data type for inference (default: bf16, recommended to avoid NaN with DINOv3)"
    )
    
    # Additional options
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images in subdirectories"
    )
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save input and normal side by side"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images that already have output files"
    )
    
    return parser.parse_args()


def find_images(input_dir: str, recursive: bool = False):
    """Find all supported image files in directory."""
    image_paths = []
    
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext}"), recursive=True))
            image_paths.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext.upper()}"), recursive=True))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    return sorted(set(image_paths))


def get_output_path(
    input_path: str,
    input_dir: str,
    output_dir: str,
    suffix: str,
    output_format: str,
) -> str:
    """Generate output path for an input image."""
    # Get relative path from input directory
    rel_path = os.path.relpath(input_path, input_dir)
    
    # Change extension and add suffix
    base, _ = os.path.splitext(rel_path)
    output_filename = f"{base}{suffix}.{output_format}"
    
    return os.path.join(output_dir, output_filename)


def main():
    args = parse_args()
    
    # Check input directory exists
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find images
    image_paths = find_images(args.input_dir, args.recursive)
    if not image_paths:
        print(f"[TransNormal] No images found in {args.input_dir}")
        return
    
    print(f"[TransNormal] Found {len(image_paths)} images")
    
    # Set dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    print(f"[TransNormal] Loading model...")
    print(f"  - Device: {args.device}")
    print(f"  - Dtype: {args.dtype}")
    
    # Create DINO encoder
    dino_encoder = None
    if os.path.exists(args.dino_path):
        print(f"[TransNormal] Loading DINOv3 encoder from {args.dino_path}")
        dino_encoder = create_dino_encoder(
            model_name="dinov3_vith16plus",
            cross_attention_dim=1024,
            weights_path=args.dino_path,
            projector_path=args.projector_path if os.path.exists(args.projector_path) else None,
            device=args.device,
            dtype=dtype,
            freeze_encoder=True,
        )
    else:
        print(f"[TransNormal] Warning: DINOv3 weights not found at {args.dino_path}")
    
    # Load pipeline
    print(f"[TransNormal] Loading pipeline from {args.model_path}")
    pipe = TransNormalPipeline.from_pretrained(
        args.model_path,
        dino_encoder=dino_encoder,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(args.device)
    
    # Process images
    print(f"[TransNormal] Processing {len(image_paths)} images...")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for image_path in tqdm(image_paths, desc="Processing"):
        try:
            # Generate output path
            output_path = get_output_path(
                image_path,
                args.input_dir,
                args.output_dir,
                args.output_suffix,
                args.output_format,
            )
            
            # Create output subdirectory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Skip if output exists
            if args.skip_existing and os.path.exists(output_path):
                skipped += 1
                continue
            
            # Load and process image
            input_image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                if args.output_format == "npz":
                    normal_map = pipe(
                        image=input_image,
                        processing_res=args.processing_res,
                        output_type="np",
                    )
                else:
                    normal_map = pipe(
                        image=input_image,
                        processing_res=args.processing_res,
                        output_type="pil",
                    )
            
            # Save output
            if args.save_comparison and args.output_format != "npz":
                # Create side-by-side comparison
                input_resized = input_image.resize(normal_map.size)
                combined = Image.new('RGB', (input_resized.width * 2, input_resized.height))
                combined.paste(input_resized, (0, 0))
                combined.paste(normal_map, (input_resized.width, 0))
                combined.save(output_path)
            elif args.output_format == "npz":
                save_normal_map(normal_map, output_path, as_rgb=False)
            else:
                normal_map.save(output_path)
            
            processed += 1
            
        except Exception as e:
            print(f"\n[TransNormal] Error processing {image_path}: {e}")
            errors += 1
    
    # Summary
    print(f"\n[TransNormal] Batch processing complete!")
    print(f"  - Processed: {processed}")
    print(f"  - Skipped: {skipped}")
    print(f"  - Errors: {errors}")
    print(f"  - Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
