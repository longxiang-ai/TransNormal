"""Latent regression and spatially weighted Haar supervision."""

import torch
import torch.nn.functional as F


def task_embeddings(batch_size, device):
    codes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    return torch.cat((codes.sin(), codes.cos()), dim=-1).repeat_interleave(batch_size, dim=0)


def latent_mask(mask):
    """A latent cell is valid only if every pixel in its 8x8 block is valid."""
    return (~F.max_pool2d((~mask.bool()).float(), 8, 8).bool()).repeat(1, 4, 1, 1)


def masked_mse(prediction, target, mask):
    if mask.any():
        return F.mse_loss(prediction[mask].float(), target[mask].float())
    return prediction.float().sum() * 0.0


def haar_dwt2d(x):
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="reflect")
    low = (x[..., 0::2] + x[..., 1::2]) / 2.0
    high = (x[..., 0::2] - x[..., 1::2]) / 2.0
    ll = (low[..., 0::2, :] + low[..., 1::2, :]) / 2.0
    lh = (low[..., 0::2, :] - low[..., 1::2, :]) / 2.0
    hl = (high[..., 0::2, :] + high[..., 1::2, :]) / 2.0
    hh = (high[..., 0::2, :] - high[..., 1::2, :]) / 2.0
    return ll, torch.cat((lh, hl, hh), dim=1)


def normal_edges(normal, mask=None):
    dx = ((normal[..., 1:] - normal[..., :-1]).square().sum(1, keepdim=True) + 1e-8).sqrt()
    dy = ((normal[..., 1:, :] - normal[..., :-1, :]).square().sum(1, keepdim=True) + 1e-8).sqrt()
    if mask is not None:
        dx = dx * (mask[..., 1:] & mask[..., :-1])
        dy = dy * (mask[..., 1:, :] & mask[..., :-1, :])
    edges = (F.pad(dx, (0, 1, 0, 0), mode="replicate") + F.pad(dy, (0, 0, 0, 1), mode="replicate")) / 2.0
    maximum = edges.flatten(1).amax(1).view(-1, 1, 1, 1)
    return edges / torch.where(maximum > 1e-8, maximum, torch.ones_like(maximum))


def wavelet_loss(prediction, target, mask):
    """Match low frequencies and edge-weighted high frequencies of the normals."""
    mask = mask.bool()
    padded_mask = mask.float()
    if mask.shape[-2] % 2 or mask.shape[-1] % 2:
        padded_mask = F.pad(padded_mask, (0, mask.shape[-1] % 2, 0, mask.shape[-2] % 2), mode="reflect")
    valid = F.max_pool2d(1 - padded_mask, 2, 2) == 0
    if not valid.any():
        return prediction.float().sum() * 0.0
    pred_ll, pred_hf = haar_dwt2d(prediction)
    gt_ll, gt_hf = haar_dwt2d(target)
    size = pred_ll.shape[-2:]
    edge = F.interpolate(normal_edges(target, mask), size=size, mode="bilinear", align_corners=False).clamp(0, 1)
    low_mask = valid.repeat(1, 3, 1, 1)
    high_mask = valid.repeat(1, 9, 1, 1).float()
    low = F.l1_loss(pred_ll[low_mask], gt_ll[low_mask])
    high = ((pred_hf - gt_hf).abs() * edge.repeat(1, 9, 1, 1) * high_mask).sum() / (high_mask.sum() + 1e-8)
    return low + high
