"""
TransNormal: Surface Normal Estimation for Transparent Objects

This package provides a diffusion-based pipeline for estimating surface normals
from RGB images, with particular effectiveness on transparent objects.

Example usage:
    from transnormal import TransNormalPipeline, create_dino_encoder
    import torch

    # Create DINO encoder
    dino_encoder = create_dino_encoder(
        model_name="dinov3_vith16plus",
        weights_path="path/to/dinov3_weights",
        projector_path="path/to/projector.pt",
        device="cuda",
    )

    # Load pipeline
    pipe = TransNormalPipeline.from_pretrained(
        "path/to/transnormal_model",
        dino_encoder=dino_encoder,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    # Run inference
    normal_map = pipe("path/to/image.jpg", output_type="np")
"""

__version__ = "1.0.0"
__author__ = "TransNormal Team"

from .pipeline import TransNormalPipeline
from .dino_encoder import DINOv3Encoder, create_dino_encoder
from .utils import (
    resize_max_res,
    resize_back,
    get_tv_resample_method,
    get_pil_resample_method,
    normal_to_rgb,
    save_normal_map,
    load_image,
    concatenate_images,
)

__all__ = [
    "TransNormalPipeline",
    "DINOv3Encoder",
    "create_dino_encoder",
    "resize_max_res",
    "resize_back",
    "get_tv_resample_method",
    "get_pil_resample_method",
    "normal_to_rgb",
    "save_normal_map",
    "load_image",
    "concatenate_images",
]
