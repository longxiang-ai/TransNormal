"""
DINOv3 Encoder for Semantic-Guided Surface Normal Estimation

This module provides a simplified DINOv3 encoder that extracts semantic features
from RGB images for cross-attention in the TransNormal pipeline.

The encoder is particularly effective for transparent objects, as DINOv3's
strong semantic features can "see through" refraction artifacts.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# DINOv3 model configurations
DINOV3_CONFIGS = {
    "dinov3_vits16": {
        "embed_dim": 384,
        "patch_size": 16,
        "n_storage_tokens": 4,
    },
    "dinov3_vitb16": {
        "embed_dim": 768,
        "patch_size": 16,
        "n_storage_tokens": 4,
    },
    "dinov3_vitl16": {
        "embed_dim": 1024,
        "patch_size": 16,
        "n_storage_tokens": 4,
    },
    "dinov3_vith16plus": {
        "embed_dim": 1280,
        "patch_size": 16,
        "n_storage_tokens": 4,
    },
}


class DINOv3Encoder(nn.Module):
    """
    DINOv3 Encoder for extracting semantic features from RGB images.
    
    This encoder provides projected patch tokens for cross-attention in the UNet,
    replacing CLIP text embeddings with visual semantic features.
    
    Args:
        model_name: DINOv3 model name (e.g., "dinov3_vith16plus")
        cross_attention_dim: Target dimension for cross-attention (1024 for SD 2.x)
        weights_path: Path to DINOv3 pretrained weights (HuggingFace format)
        freeze_encoder: Whether to freeze the DINOv3 backbone
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_vith16plus",
        cross_attention_dim: int = 1024,
        weights_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.cross_attention_dim = cross_attention_dim
        self.weights_path = weights_path
        self.freeze_encoder = freeze_encoder
        
        # Get model configuration
        if model_name not in DINOV3_CONFIGS:
            raise ValueError(f"Unknown DINOv3 model: {model_name}. Available: {list(DINOV3_CONFIGS.keys())}")
        
        self.config = DINOV3_CONFIGS[model_name]
        self.dino_hidden_dim = self.config["embed_dim"]
        self.patch_size = self.config["patch_size"]
        self.n_storage_tokens = self.config["n_storage_tokens"]
        
        # DINOv3 backbone (loaded later)
        self.dino_backbone = None
        self._use_hf_interface = False
        self._is_loaded = False
        
        # Cross-attention projector: DINO hidden_dim -> SD cross_attention_dim
        self.cross_attention_projector = nn.Linear(self.dino_hidden_dim, cross_attention_dim)
        self._init_projector()
        
        # ImageNet normalization for DINOv3
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False
        )
    
    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder (for diffusers compatibility)."""
        return self.cross_attention_projector.weight.dtype
    
    @property
    def device(self) -> torch.device:
        """Return the device of the encoder."""
        return self.cross_attention_projector.weight.device
    
    def _init_projector(self):
        """Initialize the cross-attention projector with Xavier initialization."""
        nn.init.xavier_uniform_(self.cross_attention_projector.weight)
        nn.init.zeros_(self.cross_attention_projector.bias)
    
    def _preprocess_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image from [-1, 1] to ImageNet normalized format.
        
        Args:
            pixel_values: Input images, shape (B, 3, H, W), normalized to [-1, 1]
        
        Returns:
            Preprocessed images with ImageNet normalization
        """
        # Convert from [-1, 1] to [0, 1]
        pixel_values = (pixel_values + 1.0) / 2.0
        
        # Ensure mean/std are on the same device and dtype
        mean = self.imagenet_mean.to(device=pixel_values.device, dtype=pixel_values.dtype)
        std = self.imagenet_std.to(device=pixel_values.device, dtype=pixel_values.dtype)
        
        # Apply ImageNet normalization
        pixel_values = (pixel_values - mean) / std
        
        return pixel_values
    
    def load_dino_model(self, device: torch.device = None, dtype: torch.dtype = None):
        """
        Load the DINOv3 model from HuggingFace format.
        
        Args:
            device: Device to load the model on
            dtype: Data type for the model weights
        """
        if self._is_loaded:
            return
        
        if self.weights_path is None:
            raise ValueError("weights_path must be provided to load DINOv3 model")
        
        try:
            from transformers import AutoModel
            
            print(f"[DINOv3] Loading from: {self.weights_path}")
            self.dino_backbone = AutoModel.from_pretrained(
                self.weights_path,
                trust_remote_code=True,
            )
            
            # Update config from loaded model
            hf_config = getattr(self.dino_backbone, "config", None)
            if hf_config is not None:
                self.dino_hidden_dim = getattr(hf_config, "hidden_size", self.dino_hidden_dim)
                self.patch_size = getattr(hf_config, "patch_size", self.patch_size)
                self.n_storage_tokens = getattr(hf_config, "num_register_tokens", self.n_storage_tokens)
                
                # Reinitialize projector if hidden dim changed
                if self.cross_attention_projector.in_features != self.dino_hidden_dim:
                    self.cross_attention_projector = nn.Linear(
                        self.dino_hidden_dim, self.cross_attention_dim
                    )
                    self._init_projector()
            
            self._use_hf_interface = True
            
            # Move to device/dtype
            if device is not None:
                self.dino_backbone = self.dino_backbone.to(device)
                self.cross_attention_projector = self.cross_attention_projector.to(device)
            
            if dtype is not None:
                self.dino_backbone = self.dino_backbone.to(dtype)
                self.cross_attention_projector = self.cross_attention_projector.to(dtype)
            
            # Freeze backbone
            if self.freeze_encoder:
                self.dino_backbone.requires_grad_(False)
                self.dino_backbone.eval()
            
            self._is_loaded = True
            print(f"[DINOv3] Successfully loaded {self.model_name}")
            print(f"  - Hidden dim: {self.dino_hidden_dim}")
            print(f"  - Patch size: {self.patch_size}")
            print(f"  - Cross-attention dim: {self.cross_attention_dim}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 model from {self.weights_path}.\n"
                f"Error: {e}"
            )
    
    def _ensure_loaded(self):
        """Ensure the model is loaded before forward pass."""
        if not self._is_loaded:
            raise RuntimeError(
                "DINOv3 model not loaded. Call load_dino_model() first."
            )
    
    def extract_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch tokens from DINOv3.
        
        Args:
            pixel_values: Input images, shape (B, 3, H, W), normalized to [-1, 1]
        
        Returns:
            patch_tokens: Shape (B, N, D) where N is number of patches, D is hidden_dim
        """
        self._ensure_loaded()
        
        # Preprocess image
        preprocessed = self._preprocess_image(pixel_values)
        
        # Ensure dimensions are multiples of patch_size
        _, _, H, W = preprocessed.shape
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        if new_H != H or new_W != W:
            preprocessed = F.interpolate(
                preprocessed,
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            )
        
        # Forward through DINOv3
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            if self._use_hf_interface:
                outputs = self.dino_backbone(
                    pixel_values=preprocessed,
                    output_hidden_states=True
                )
                last_hidden = outputs.last_hidden_state
                # Remove CLS and register tokens
                n_special = 1 + self.n_storage_tokens
                patch_tokens = last_hidden[:, n_special:, :]
            else:
                outputs = self.dino_backbone.forward_features(preprocessed, masks=None)
                patch_tokens = outputs['x_norm_patchtokens']
        
        return patch_tokens
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract features for cross-attention.
        
        Args:
            pixel_values: Input images, shape (B, 3, H, W), normalized to [-1, 1]
        
        Returns:
            dict with 'cross_attention_features': Projected features, shape (B, N, cross_attention_dim)
        """
        self._ensure_loaded()
        
        # Extract patch tokens
        patch_tokens = self.extract_patch_tokens(pixel_values)
        
        # Project to cross-attention dimension
        projector_dtype = next(self.cross_attention_projector.parameters()).dtype
        if patch_tokens.dtype != projector_dtype:
            patch_tokens = patch_tokens.to(dtype=projector_dtype)
        
        cross_attention_features = self.cross_attention_projector(patch_tokens)
        
        return {'cross_attention_features': cross_attention_features}
    
    def get_cross_attention_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get only cross-attention features.
        
        Args:
            pixel_values: Input images, shape (B, 3, H, W), normalized to [-1, 1]
        
        Returns:
            cross_attention_features: Shape (B, N, cross_attention_dim)
        """
        return self.forward(pixel_values)['cross_attention_features']
    
    def load_projector(self, projector_path: str, device: torch.device = None):
        """
        Load pretrained projector weights.
        
        Args:
            projector_path: Path to projector weights file (.pt)
            device: Device to load weights on
        """
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector weights not found: {projector_path}")
        
        state_dict = torch.load(projector_path, map_location=device or "cpu")
        self.cross_attention_projector.load_state_dict(state_dict)
        print(f"[DINOv3] Loaded projector weights from {projector_path}")


def create_dino_encoder(
    model_name: str = "dinov3_vith16plus",
    cross_attention_dim: int = 1024,
    weights_path: Optional[str] = None,
    projector_path: Optional[str] = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
    freeze_encoder: bool = True,
) -> DINOv3Encoder:
    """
    Factory function to create and initialize a DINOv3 encoder.
    
    Args:
        model_name: DINOv3 model name
        cross_attention_dim: Target dimension for cross-attention
        weights_path: Path to DINOv3 pretrained weights
        projector_path: Path to projector weights (optional)
        device: Device to load the model on
        dtype: Data type for the model
        freeze_encoder: Whether to freeze the backbone
    
    Returns:
        Initialized DINOv3Encoder
    """
    encoder = DINOv3Encoder(
        model_name=model_name,
        cross_attention_dim=cross_attention_dim,
        weights_path=weights_path,
        freeze_encoder=freeze_encoder,
    )
    
    # Load DINO backbone
    if weights_path is not None:
        encoder.load_dino_model(device=device, dtype=dtype)
    
    # Load projector weights if provided
    if projector_path is not None:
        encoder.load_projector(projector_path, device=device)
    
    # Move to device
    if device is not None:
        encoder = encoder.to(device)
    
    if dtype is not None:
        encoder = encoder.to(dtype)
    
    return encoder
