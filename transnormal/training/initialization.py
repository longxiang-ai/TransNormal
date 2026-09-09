"""Load the pretrained components and group all trainable parameters for DDP."""

from pathlib import Path

import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel

from transnormal.dino_encoder import DINOv3Encoder
from .losses import task_embeddings


def load_component(cls, source, subfolder, revision=None, local_files_only=False, **kwargs):
    options = {"subfolder": subfolder, "local_files_only": local_files_only, **kwargs}
    if not Path(source).is_dir() and revision:
        options["revision"] = revision
    model, info = cls.from_pretrained(source, output_loading_info=True, **options)
    failures = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if failures:
        raise RuntimeError(f"Incomplete {cls.__name__} initialization: {failures}")
    return model


class TrainableModel(torch.nn.Module):
    """UNet and projector share one distributed wrapper and one checkpoint."""

    def __init__(self, unet, projector):
        super().__init__()
        self.unet, self.projector = unet, projector

    def forward(self, rgb_latents, tokens, timestep, changed_latents=None, changed_tokens=None):
        batch_size = rgb_latents.shape[0]
        labels = task_embeddings(batch_size, rgb_latents.device)
        features = self.projector(tokens.to(self.projector.weight.dtype))
        prediction = self.unet(
            torch.cat((rgb_latents, rgb_latents)),
            torch.full((batch_size * 2,), timestep, device=rgb_latents.device, dtype=torch.long),
            encoder_hidden_states=torch.cat((features, features)),
            class_labels=labels, return_dict=False,
        )[0]
        changed_prediction = None
        if changed_latents is not None:
            changed_features = self.projector(changed_tokens.to(self.projector.weight.dtype))
            changed_prediction = self.unet(
                changed_latents,
                torch.full((batch_size,), timestep, device=rgb_latents.device, dtype=torch.long),
                encoder_hidden_states=changed_features,
                class_labels=labels[:batch_size], return_dict=False,
            )[0]
        return prediction, changed_prediction


def initialize(config, device, dtype):
    source, revision = config["initial_model"], config["initial_revision"]
    local = config["local_files_only"]
    # Start from Lotus's task-conditioned normal model, not an uninitialized SD UNet.
    unet = load_component(UNet2DConditionModel, source, "unet", revision, local, torch_dtype=torch.float32)
    expected = {"in_channels": 4, "out_channels": 4, "cross_attention_dim": 1024,
                "class_embed_type": "projection", "projection_class_embeddings_input_dim": 4}
    for key, value in expected.items():
        if getattr(unet.config, key, None) != value:
            raise ValueError(f"Incompatible UNet {key}; expected {value}.")
    vae_source = config["vae_model"] or source
    vae = load_component(AutoencoderKL, vae_source, "vae", revision if vae_source == source else None,
                         local, torch_dtype=dtype)
    if vae.config.latent_channels != 4 or abs(vae.config.scaling_factor - 0.18215) > 1e-8:
        raise ValueError("Incompatible VAE latent configuration.")
    vae.requires_grad_(False).eval().to(device)
    dino = DINOv3Encoder(model_name="dinov3_vith16plus", weights_path=config["dino_model"], freeze_encoder=True)
    kwargs = {"local_files_only": local}
    if not Path(config["dino_model"]).is_dir():
        kwargs["revision"] = config["dino_revision"]
    dino.load_dino_model(device=device, dtype=dtype, **kwargs)
    dino.train()
    dino.cross_attention_projector.requires_grad_(True)
    unet.requires_grad_(True)
    scheduler_kwargs = {"subfolder": "scheduler", "local_files_only": local}
    if not Path(source).is_dir():
        scheduler_kwargs["revision"] = revision
    scheduler = PNDMScheduler.from_pretrained(source, **scheduler_kwargs)
    return TrainableModel(unet, dino.cross_attention_projector), vae, dino, scheduler
