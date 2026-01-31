"""
TransNormal Pipeline for Surface Normal Estimation

This pipeline is designed for transparent object surface normal estimation,
using DINOv3 encoder for semantic-guided geometry estimation.

Based on the Lotus-D deterministic pipeline architecture.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from diffusers import DiffusionPipeline, StableDiffusionMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer

from .utils import resize_max_res, resize_back, get_tv_resample_method
from torchvision.transforms import InterpolationMode

logger = logging.get_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Get timesteps from scheduler.
    
    Args:
        scheduler: The scheduler to get timesteps from
        num_inference_steps: Number of diffusion steps
        device: Device to move timesteps to
        timesteps: Custom timesteps (optional)
    
    Returns:
        Tuple of (timesteps, num_inference_steps)
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom "
                f"timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TransNormalPipeline(DiffusionPipeline, StableDiffusionMixin):
    """
    TransNormal Pipeline for Surface Normal Estimation
    
    This pipeline uses DINOv3 encoder for semantic-guided geometry estimation,
    particularly effective for transparent objects where traditional methods fail.
    
    Args:
        vae: Variational Autoencoder for encoding/decoding images
        text_encoder: CLIP text encoder (kept for compatibility)
        tokenizer: CLIP tokenizer (kept for compatibility)
        unet: UNet2DConditionModel for denoising
        scheduler: Noise scheduler
        dino_encoder: Optional DINOv3 encoder for semantic features
    """
    
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["text_encoder", "tokenizer", "dino_encoder"]
    
    # Default processing resolution
    default_processing_resolution = 768
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        dino_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            dino_encoder=dino_encoder,
        )
        
        # VAE scale factor (typically 8 for SD)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        # DINOv3 encoder usage flag
        self._use_dino_for_cross_attention = dino_encoder is not None
    
    def set_dino_encoder(self, dino_encoder: Optional[nn.Module], device: torch.device = None):
        """
        Set or remove the DINOv3 encoder.
        
        Args:
            dino_encoder: DINOv3 encoder module, or None to disable
            device: Target device for the encoder
        """
        if dino_encoder is not None and device is not None:
            dino_encoder = dino_encoder.to(device)
            if hasattr(dino_encoder, 'dino_backbone') and dino_encoder.dino_backbone is not None:
                dino_encoder.dino_backbone = dino_encoder.dino_backbone.to(device)
        
        # Update registered module
        self.register_modules(dino_encoder=dino_encoder)
        self._use_dino_for_cross_attention = dino_encoder is not None
    
    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        """
        Encode text prompt using CLIP text encoder.
        
        Args:
            prompt: Text prompt
            device: Target device
            num_images_per_prompt: Number of images per prompt
        
        Returns:
            Text embeddings tensor
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds
    
    def _get_encoder_hidden_states(
        self,
        rgb_in: torch.Tensor,
        prompt: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get encoder hidden states for cross-attention.
        
        Uses DINOv3 features if encoder is available, otherwise uses CLIP text embeddings.
        
        Args:
            rgb_in: Input RGB image tensor, shape (B, 3, H, W), range [-1, 1]
            prompt: Text prompt (used only if DINO encoder is not available)
            device: Target device
        
        Returns:
            Encoder hidden states for cross-attention
        """
        if self._use_dino_for_cross_attention and self.dino_encoder is not None:
            # Use DINOv3 to extract semantic features
            encoder_hidden_states = self.dino_encoder.get_cross_attention_features(rgb_in)
            
            # Ensure dtype matches UNet
            if self.unet is not None:
                encoder_hidden_states = encoder_hidden_states.to(dtype=self.unet.dtype)
            return encoder_hidden_states
        else:
            # Fallback to CLIP text encoder
            return self.encode_prompt(prompt, device)
    
    def preprocess_image(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray, str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Preprocess input image to tensor format.
        
        Args:
            image: Input image (PIL, numpy, tensor, or path)
            device: Target device
            dtype: Target dtype
        
        Returns:
            Preprocessed image tensor, shape (1, 3, H, W), range [-1, 1]
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            # Ensure HWC format
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            
            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Convert to tensor (B, C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Normalize to [-1, 1]
        if image.min() >= 0 and image.max() <= 1:
            image = image * 2.0 - 1.0
        
        return image.to(device=device, dtype=dtype)
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray, str],
        prompt: str = "",
        timestep: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        output_type: str = "np",
        return_dict: bool = False,
        **kwargs,
    ):
        """
        Run surface normal estimation on input image.
        
        Args:
            image: Input RGB image (PIL, numpy, tensor, or file path)
            prompt: Text prompt (optional, used only if DINO encoder is not available)
            timestep: Diffusion timestep for deterministic prediction (default: 1)
            processing_res: Processing resolution (default: 768)
            match_input_res: Whether to resize output to match input resolution
            resample_method: Resampling method for resizing
            output_type: Output format - "np" (numpy), "pt" (tensor), or "pil" (PIL Image)
            return_dict: Whether to return a dict with additional info
        
        Returns:
            Normal map in specified format. Normal vectors are in camera coordinates:
            - X: right (positive = right)
            - Y: down (positive = down)  
            - Z: forward (positive = into screen)
            
            Output range is [0, 1] where 0.5 represents zero in each axis.
        """
        # Set default processing resolution
        if processing_res is None:
            processing_res = self.default_processing_resolution
        
        device = self._execution_device
        dtype = self.unet.dtype if self.unet is not None else torch.float32
        
        # Preprocess input image
        rgb_in = self.preprocess_image(image, device, dtype)
        input_size = rgb_in.shape[-2:]
        
        # Resize to processing resolution
        resample_method_tv = get_tv_resample_method(resample_method)
        if processing_res > 0:
            rgb_in = resize_max_res(
                rgb_in,
                max_edge_resolution=processing_res,
                resample_method=resample_method_tv,
            )
        
        # Get encoder hidden states (DINO or CLIP)
        encoder_hidden_states = self._get_encoder_hidden_states(
            rgb_in=rgb_in,
            prompt=prompt,
            device=device,
        )
        
        # Prepare timestep
        timesteps = torch.tensor([timestep], device=device).long()
        
        # Encode RGB to latent space
        rgb_latents = self.vae.encode(rgb_in).latent_dist.sample()
        rgb_latents = rgb_latents * self.vae.config.scaling_factor
        
        # Task embedding for normal estimation
        task_emb = torch.tensor([1, 0], dtype=dtype, device=device).unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        
        # Single-step deterministic prediction
        t = timesteps[0]
        pred = self.unet(
            rgb_latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            class_labels=task_emb,
        )[0]
        
        # Decode prediction
        normal_latent = pred / self.vae.config.scaling_factor
        normal_image = self.vae.decode(normal_latent, return_dict=False)[0]
        
        # Post-process to [0, 1] range
        normal_image = (normal_image / 2 + 0.5).clamp(0, 1)
        
        # Resize back to input resolution if requested
        if match_input_res and processing_res > 0:
            normal_image = F.interpolate(
                normal_image,
                size=input_size,
                mode='bilinear',
                align_corners=False,
            )
        
        # Convert to output format
        if output_type == "pt":
            output = normal_image  # (B, 3, H, W), range [0, 1]
        elif output_type == "np":
            # Convert to float32 first (bfloat16 not supported by numpy)
            output = normal_image.float().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
            if output.shape[0] == 1:
                output = output[0]  # (H, W, 3)
        elif output_type == "pil":
            # Convert to float32 first (bfloat16 not supported by numpy)
            output = normal_image.float().cpu().permute(0, 2, 3, 1).numpy()
            output = (output * 255).astype(np.uint8)
            if output.shape[0] == 1:
                output = Image.fromarray(output[0])
            else:
                output = [Image.fromarray(img) for img in output]
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        if return_dict:
            return {"normal": output, "resolution": normal_image.shape[-2:]}
        return output
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dino_encoder: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Load TransNormalPipeline from pretrained weights.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or HuggingFace model ID
            dino_encoder: Optional pre-loaded DINO encoder
            **kwargs: Additional arguments passed to DiffusionPipeline.from_pretrained
        
        Returns:
            TransNormalPipeline instance
        """
        # Load base pipeline components
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Set DINO encoder if provided
        if dino_encoder is not None:
            pipeline.set_dino_encoder(dino_encoder)
        
        return pipeline
