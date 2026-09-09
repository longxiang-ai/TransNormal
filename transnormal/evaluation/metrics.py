"""Pixel-pooled angular metrics in the original evaluation convention."""

import numpy as np
import torch


def normal_errors(prediction, target, mask):
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError("Prediction must be finite and match the ground-truth shape.")
    cosine = torch.nn.functional.cosine_similarity(prediction.float(), target.float(), dim=1)
    angles = torch.acos(cosine.clamp(-1., 1.)) * 180. / np.pi
    return angles[:, None][mask]


def summarize(errors):
    values = errors.detach().cpu().numpy()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("No finite evaluation pixels.")
    result = {"mean": float(np.average(values)), "median": float(np.median(values)),
              "rmse": float(np.sqrt(np.sum(values * values) / values.size))}
    result.update({f"a{i}": float(100. * (np.sum(values < threshold) / values.size))
                   for i, threshold in enumerate([5., 7.5, 11.25, 22.5, 30.], 1)})
    return result
