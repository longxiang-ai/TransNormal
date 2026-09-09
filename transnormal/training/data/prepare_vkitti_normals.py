"""Generate Virtual KITTI normal PNGs using local plane fitting.

Adapted from the Lotus data preprocessing implementation (Apache-2.0).
See docs/third_party.md and LICENSE-APACHE-2.0.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def fit_normals(depth, intrinsics):
    """Fit a plane in each 5x5 neighborhood, rejecting depth discontinuities."""
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, dtype=depth.dtype, device=depth.device),
                          torch.arange(w, dtype=depth.dtype, device=depth.device), indexing="ij")
    fx, fy, cx, cy = intrinsics
    # Keep the inverse-intrinsics multiplication order used to generate the training normals.
    inverse = depth.new_tensor([[1 / fx, 0., -cx / fx], [0., 1 / fy, -cy / fy], [0., 0., 1.]])
    pixels = torch.stack((x, y, torch.ones_like(x))).reshape(1, 3, -1)
    points = inverse[None].bmm(pixels).reshape(3, h, w) * depth
    patches = F.unfold(points[None], 5, padding=2).view(1, 3, 25, h, w).permute(0, 3, 4, 2, 1)
    valid_depth = ((depth > 1e-3) & (depth < 80)).to(depth.dtype)
    valid = F.unfold(valid_depth[None, None], 5, padding=2).view(1, 1, 25, h, w).permute(0, 3, 4, 2, 1)
    center_depth = patches[..., 12:13, 2:]
    valid *= ((patches[..., 2:] - center_depth).abs() / center_depth < .05)
    matrix = torch.cat((patches, torch.ones_like(patches[..., :1])), -1)
    matrix = torch.where(valid.bool(), matrix, torch.zeros_like(matrix)).reshape(-1, 25, 4)
    eigenvalues, eigenvectors = torch.linalg.eig(matrix.transpose(1, 2).bmm(matrix))
    real = (eigenvalues.imag.sum(1) == 0) & (eigenvectors.imag.sum((1, 2)) == 0)
    minimum = eigenvalues.real.argmin(1)
    planes = eigenvectors.real[torch.arange(h * w, device=depth.device), :, minimum]
    normal = F.normalize(planes[:, :3], dim=1).reshape(h, w, 3).permute(2, 0, 1)
    normal *= torch.sign((normal * points).sum(0, keepdim=True))
    mask = valid[0, ..., 12, 0].bool() & (valid[0, ..., 0].sum(-1) >= 4) & real.reshape(h, w)
    mask &= normal.norm(dim=0) > .5
    return normal, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device; preprocessing only")
    args = parser.parse_args()
    root = Path(args.root)
    for intrinsic_file in sorted(root.glob("Scene*/*/intrinsic.txt")):
        rows = np.loadtxt(intrinsic_file, skiprows=1)
        calibration = {(int(row[0]), int(row[1])): row[2:6] for row in rows}
        for path in sorted((intrinsic_file.parent / "frames/depth").glob("Camera_*/*.png")):
            relative = path.relative_to(root)
            output = root / str(relative).replace("depth", "normal")
            if output.exists():
                continue
            with Image.open(path) as image:
                depth = torch.from_numpy(np.asarray(image).astype(np.float64) / 100).to(args.device)
            frame = int(path.stem.split("_")[-1])
            camera = int(path.parent.name.split("_")[-1])
            normal, mask = fit_normals(depth, calibration[(frame, camera)])
            normals = normal.permute(1, 2, 0).cpu().numpy()
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
            pixels = (((normals + 1) * .5) * 255).astype(np.uint8)
            pixels *= mask.cpu().numpy()[..., None]
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(output)
            print(output, flush=True)


if __name__ == "__main__":
    main()
