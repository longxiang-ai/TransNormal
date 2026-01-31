#!/usr/bin/env python
"""
TransNormal Single Image Inference Script

This script demonstrates how to run surface normal estimation on a single image
using the TransNormal pipeline.

Usage:
    python inference.py --image path/to/image.jpg --output output.png

    # With custom model paths
    python inference.py \
        --image input.jpg \
        --output normal.png \
        --model_path path/to/transnormal_model \
        --dino_path path/to/dinov3_weights \
        --projector_path path/to/projector.pt
"""

import argparse
import os
import torch
from PIL import Image

from transnormal import TransNormalPipeline, create_dino_encoder, save_normal_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="TransNormal: Surface Normal Estimation for Transparent Objects"
    )
    
    # Input/Output
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input RGB image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="normal_output.png",
        help="Path to output normal map (default: normal_output.png)"
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
        "--output_type",
        type=str,
        default="pil",
        choices=["pil", "np", "pt"],
        help="Output format (default: pil)"
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
    
    # Visualization
    parser.add_argument(
        "--save_side_by_side",
        action="store_true",
        help="Save input and output side by side"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check input file exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")
    
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
        print(f"[TransNormal] Running without DINO encoder (using CLIP text encoder)")
    
    # Load pipeline
    print(f"[TransNormal] Loading pipeline from {args.model_path}")
    pipe = TransNormalPipeline.from_pretrained(
        args.model_path,
        dino_encoder=dino_encoder,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(args.device)
    
    # Run inference
    print(f"[TransNormal] Processing image: {args.image}")
    input_image = Image.open(args.image).convert("RGB")
    
    with torch.no_grad():
        normal_map = pipe(
            image=input_image,
            processing_res=args.processing_res,
            output_type=args.output_type,
        )
    
    # Save output
    if args.save_side_by_side and args.output_type == "pil":
        # Create side-by-side comparison
        input_resized = input_image.resize(normal_map.size)
        combined = Image.new('RGB', (input_resized.width * 2, input_resized.height))
        combined.paste(input_resized, (0, 0))
        combined.paste(normal_map, (input_resized.width, 0))
        combined.save(args.output)
        print(f"[TransNormal] Saved side-by-side comparison to {args.output}")
    elif args.output_type == "pil":
        normal_map.save(args.output)
        print(f"[TransNormal] Saved normal map to {args.output}")
    else:
        save_normal_map(normal_map, args.output)
        print(f"[TransNormal] Saved normal map to {args.output}")
    
    print("[TransNormal] Done!")


if __name__ == "__main__":
    main()
