"""
Utility functions for TransNormal pipeline.

Includes image processing utilities for preprocessing and postprocessing.
"""

from typing import List, Union
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img: Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution: Maximum edge length (pixels)
        resample_method: Resampling method used to resize images

    Returns:
        Resized image tensor
    """
    assert img.dim() == 4, f"Invalid input shape {img.shape}, expected [B, C, H, W]"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width,
        max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def resize_back(
    img: Union[torch.Tensor, np.ndarray, Image.Image, List[Image.Image]],
    target_size: Union[int, tuple],
    resample_method: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
) -> Union[torch.Tensor, np.ndarray, Image.Image, List[Image.Image]]:
    """
    Resize image back to target size.

    Args:
        img: Image to be resized (tensor, numpy, PIL, or list of PIL)
        target_size: Target size (H, W) or single int for square
        resample_method: Resampling method for resizing

    Returns:
        Resized image in the same format as input
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    if isinstance(img, torch.Tensor):
        resized_img = resize(img, target_size, resample_method, antialias=True)
    elif isinstance(img, np.ndarray):
        # Convert to tensor
        if img.ndim == 3:  # HWC
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        else:  # BHWC
            img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2)
        
        resized_tensor = resize(img_tensor, target_size, resample_method, antialias=True)
        
        # Convert back
        if img.ndim == 3:
            resized_img = resized_tensor.squeeze(0).permute(1, 2, 0).numpy()
        else:
            resized_img = resized_tensor.permute(0, 2, 3, 1).numpy()
    elif isinstance(img, Image.Image):
        # PIL uses (width, height)
        pil_size = (target_size[1], target_size[0])
        resized_img = img.resize(pil_size, resample_method)
    elif isinstance(img, list) and all(isinstance(i, Image.Image) for i in img):
        pil_size = (target_size[1], target_size[0])
        resized_img = [i.resize(pil_size, resample_method) for i in img]
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    return resized_img


def get_tv_resample_method(method_str: str) -> InterpolationMode:
    """
    Get torchvision interpolation mode from string.

    Args:
        method_str: Resampling method name ("bilinear", "bicubic", "nearest")

    Returns:
        Corresponding InterpolationMode
    """
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str.lower())
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {method_str}")
    return resample_method


def get_pil_resample_method(method_str: str) -> int:
    """
    Get PIL resampling method from string.

    Args:
        method_str: Resampling method name ("bilinear", "bicubic", "nearest")

    Returns:
        Corresponding PIL resampling constant
    """
    resample_method_dict = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST,
    }
    resample_method = resample_method_dict.get(method_str.lower())
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {method_str}")
    return resample_method


def normal_to_rgb(normal: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert normal map to RGB visualization.
    
    Normal vectors are assumed to be in range [-1, 1] or [0, 1].
    Output is RGB image in range [0, 255].
    
    Args:
        normal: Normal map tensor/array, shape (H, W, 3) or (B, H, W, 3) or (B, 3, H, W)
    
    Returns:
        RGB visualization as uint8 numpy array
    """
    if isinstance(normal, torch.Tensor):
        normal = normal.cpu().numpy()
    
    # Handle different formats
    if normal.ndim == 4:
        if normal.shape[1] == 3:  # BCHW
            normal = np.transpose(normal, (0, 2, 3, 1))  # BHWC
        normal = normal[0]  # Take first batch
    
    # Convert from [-1, 1] to [0, 1] if needed
    if normal.min() < 0:
        normal = (normal + 1.0) / 2.0
    
    # Clamp and convert to uint8
    normal = np.clip(normal, 0, 1)
    rgb = (normal * 255).astype(np.uint8)
    
    return rgb


def save_normal_map(
    normal: Union[torch.Tensor, np.ndarray],
    output_path: str,
    as_rgb: bool = True,
):
    """
    Save normal map to file.
    
    Args:
        normal: Normal map tensor/array
        output_path: Output file path
        as_rgb: If True, save as RGB visualization; if False, save raw values as NPZ
    """
    if as_rgb:
        rgb = normal_to_rgb(normal)
        Image.fromarray(rgb).save(output_path)
    else:
        if isinstance(normal, torch.Tensor):
            normal = normal.cpu().numpy()
        np.savez_compressed(output_path, normal=normal)


def load_image(image_path: str) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image in RGB mode
    """
    return Image.open(image_path).convert("RGB")


def concatenate_images(*image_lists) -> Image.Image:
    """
    Concatenate multiple rows of images into a single image.
    
    Args:
        *image_lists: Variable number of image lists, each list is a row
    
    Returns:
        Concatenated PIL Image
    """
    if not image_lists or not image_lists[0]:
        raise ValueError("At least one non-empty image list must be provided")
    
    max_width = 0
    total_height = 0
    row_heights = []
    
    for image_list in image_lists:
        if image_list:
            width = sum(img.width for img in image_list)
            height = image_list[0].height
            max_width = max(max_width, width)
            total_height += height
            row_heights.append(height)
    
    new_image = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for i, image_list in enumerate(image_lists):
        x_offset = 0
        for img in image_list:
            new_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_heights[i]
    
    return new_image
