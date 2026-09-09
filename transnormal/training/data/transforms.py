"""Dataset-specific geometry conventions used by the formal training recipe."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


def resize_geometry(array, size):
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
    return F.interpolate(tensor[None], size=size, mode="nearest")[0]


def transform(kind, image, normal, mask, resolution, seed, random_flip=True, depth=None, changed=None):
    rng = random.Random(seed)
    normal = np.array(normal, dtype=np.float32, copy=True)
    if kind == "cleargrasp":
        normal[..., 0] *= -1
        side = min(image.size)
        left = rng.randint(0, image.width - side) if image.width >= image.height else 0
        top = rng.randint(0, image.height - side) if image.height > image.width else 0
        image = image.crop((left, top, left + side, top + side))
        normal = normal[top:top + side, left:left + side]
        mask = mask[top:top + side, left:left + side]
        size = (resolution, resolution)
    elif kind == "hypersim":
        # Orient the camera-space normals using the camera viewing rays.
        normal[..., 0] *= -1
        h, w = normal.shape[:2]
        y, x = np.mgrid[:h, :w]
        rays = np.stack(((x - w / 2) / 886.81, (y - h / 2) / 886.81, np.ones_like(x)), axis=-1)
        normal[(normal * rays * depth[..., None]).sum(-1) < 0] *= -1
        size = (int(resolution * h / min(h, w)), int(resolution * w / min(h, w)))
    elif kind == "vkitti":
        h, w = image.height, image.width
        scale = max(352 / h, 1216 / w, 1.0)
        h, w = int(h * scale), int(w * scale)
        image = TF.resize(image, (h, w), interpolation=Image.Resampling.BILINEAR)
        normal = np.asarray(Image.fromarray(normal.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)).copy()
        depth = resize_geometry(depth, (h, w))[0].numpy()
        top, left = h - 352, (w - 1216) // 2
        image = image.crop((left, top, left + 1216, top + 352))
        normal = normal[top:top + 352, left:left + 1216]
        depth = depth[top:top + 352, left:left + 1216]
        valid_normal = np.any(normal != 0, axis=-1)
        mask = ((depth > 1e-5) & (depth < 80) & valid_normal)[..., None]
        # Preserve black invalid normals when reflecting the encoded x component.
        flip = random_flip and torch.rand((), generator=torch.Generator().manual_seed(seed)).item() > 0.5
        if flip:
            image = TF.hflip(image)
            normal = normal[:, ::-1].copy()
            valid_normal = np.any(normal != 0, axis=-1)
            normal[..., 0][valid_normal] = 255 - normal[..., 0][valid_normal]
            mask = mask[:, ::-1].copy()
        return {
            "pixel_values": torch.from_numpy(np.asarray(image).astype(np.float32) / 127.5 - 1).permute(2, 0, 1),
            "normal_values": torch.from_numpy(normal.astype(np.float32) / 127.5 - 1).permute(2, 0, 1),
            "valid_mask_values": torch.from_numpy(mask.copy()).permute(2, 0, 1).bool(),
        }
    else:
        size = (resolution, resolution)

    if kind == "transnormal":
        # Tensor bilinear resize matches the synthetic training loader (no antialias).
        rgb = F.interpolate(TF.to_tensor(image)[None], size=size, mode="bilinear", align_corners=False)[0]
        change = F.interpolate(TF.to_tensor(changed)[None], size=size, mode="bilinear", align_corners=False)[0]
    else:
        rgb = TF.to_tensor(TF.resize(image, size, interpolation=Image.Resampling.BILINEAR))
        change = None
    normals = resize_geometry(normal, size)
    valid = resize_geometry(mask, size) > 0.5
    if kind in ("cleargrasp", "hypersim"):
        normals = normals.clamp(-1, 1)
        # The formal recipe supervises the full image for these two datasets.
        valid = torch.ones_like(valid, dtype=torch.bool)
    if random_flip and rng.random() > 0.5:
        rgb, normals, valid = (x.flip(-1) for x in (rgb, normals, valid))
        normals[0] *= -1
        if change is not None:
            change = change.flip(-1)
    batch = {"pixel_values": rgb.clamp(0, 1) * 2 - 1, "normal_values": normals, "valid_mask_values": valid}
    if change is not None:
        batch["rgb_change_values"] = change.clamp(0, 1) * 2 - 1
    return batch
