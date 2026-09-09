"""Decode the six paper benchmarks without training-time augmentation."""

import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

SPLIT_DIR = Path(__file__).resolve().parents[2] / "configs/splits"
SPLITS = {"cleargrasp": "cleargrasp_test_synthetic.txt", "transnormal": "transnormal_test.txt",
          "clearpose": "clearpose_test.txt", "nyuv2": "nyuv2_test.txt",
          "scannet": "scannet_test.txt", "ibims": "ibims_test.txt"}


def read_split(path):
    keys = Path(path).read_text().splitlines()
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("The split must be non-empty and contain unique sample keys.")
    if any(not key or Path(key).is_absolute() or ".." in Path(key).parts for key in keys):
        raise ValueError("Split keys must be relative paths without parent traversal.")
    return keys


def read_rgb(path):
    if isinstance(path, bytes):
        pixels = cv2.imdecode(np.frombuffer(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit RGB image: {path if not isinstance(path, bytes) else 'tar member'}")
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def read_png(path):
    with Image.open(io.BytesIO(path) if isinstance(path, bytes) else path) as image:
        return np.array(image)


def read_exr(path):
    import Imath
    import OpenEXR
    handle = OpenEXR.InputFile(str(path))
    try:
        window = handle.header()["dataWindow"]
        shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
        return np.stack([np.frombuffer(handle.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)),
                                       dtype=np.float32).reshape(shape) for c in "RGB"], axis=-1)
    finally:
        handle.close()


def first_file(candidates):
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Required annotation missing; checked: " + ", ".join(map(str, candidates)))


class Benchmark:
    def __init__(self, name, root, split=None, index=None):
        if name not in SPLITS:
            raise ValueError(f"Unsupported benchmark: {name}")
        self.name, self.root = name, Path(root)
        self.split = Path(split) if split else SPLIT_DIR / SPLITS[name]
        self.keys = read_split(self.split)
        self.index, self.records = Path(index) if index else None, None
        if index:
            if name != "transnormal":
                raise ValueError("Tar indices are only used for TransNormal-Synthetic.")
            with self.index.open() as handle:
                header = json.loads(next(handle))
                records = [json.loads(line) for line in handle if line.strip()]
            if header.get("format") != "transnormal-training-index-v1" or header.get("kind") != name:
                raise ValueError("Invalid TransNormal data index.")
            self.records = {row["key"]: row for row in records}
            if len(records) != header["samples"] or len(self.records) != len(records):
                raise ValueError("Index sample count is incorrect or keys are duplicated.")
            if set(self.keys) - self.records.keys():
                raise ValueError("The index does not contain every selected evaluation view.")
            self.root = self.index.parent / header["root"]

    def paths(self, key):
        rel = Path(key)
        if self.name == "clearpose":
            base = self.root / rel.parent
            return {name: base / f"{rel.stem}-{suffix}.png" for name, suffix in
                    [("image", "color"), ("normal", "rendered_normal"), ("mask", "rendered_mask"), ("depth", "rendered_depth")]}
        if self.name == "transnormal":
            scene, view = rel.parts
            suffix = f"{int(view.removeprefix('view_')):04d}"
            base = self.root / scene / "views" / view / "with_transparent"
            return {"image": base / "rgb_.png", "normal": base / f"normal_{suffix}.png",
                    "mask": base / "masks" / f"all_{suffix}.png"}
        if self.name == "cleargrasp":
            if len(rel.parts) != 4 or rel.parts[0] != "synthetic-test":
                raise ValueError("ClearGrasp paper protocol requires synthetic-test frame paths.")
            base = self.root / rel.parts[0] / rel.parts[1]
            stem = rel.name.removesuffix("-cameraNormals.exr")
            image = first_file([base / folder / (stem + suffix) for folder in ("rgb-imgs", "transparent-rgb-imgs")
                                for suffix in ("-rgb.jpg", "-transparent-rgb-img.jpg", "-input-img.jpg")])
            normal = first_file([base / folder / (stem + suffix) for folder in ("camera-normals", "normals")
                                 for suffix in ("-cameraNormals.exr", "-normals.exr")])
            mask = first_file([base / folder / (stem + suffix) for folder in ("segmentation-masks", "masks")
                               for suffix in ("-segmentation-mask.png", "-mask.png")])
            return {"image": image, "normal": normal, "mask": mask}
        image = self.root / rel
        stem, _ = image.name.split("_img")
        return {"image": image, "normal": image.with_name(stem + ("_normal.exr" if self.name == "ibims" else "_normal.png")),
                "intrinsics": image.with_name(stem + "_intrins.npy")}

    def load(self, key):
        if self.records is not None:
            row = self.records[key]
            if "shard" not in row:
                fields = {name: self.root / row[field] for name, field in
                          [("with_rgb.png", "image"), ("with_normal.png", "normal"), ("with_mask_all.png", "mask")]}
            else:
                fields = {}
                with (self.root / row["shard"]).open("rb") as handle:
                    for name in ("with_rgb.png", "with_normal.png", "with_mask_all.png"):
                        offset, size = row["files"][name]
                        handle.seek(offset)
                        fields[name] = handle.read(size)
                        if len(fields[name]) != size:
                            raise ValueError("Truncated tar member; rebuild the index from intact shards.")
            image = read_rgb(fields["with_rgb.png"])
            normal = read_png(fields["with_normal.png"])[..., :3].astype(np.float32) / 127.5 - 1.
            mask = read_png(fields["with_mask_all.png"])
        else:
            paths = self.paths(key)
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            image = read_rgb(paths["image"])
            if self.name in ("cleargrasp", "ibims"):
                normal = read_exr(paths["normal"])
                if self.name == "cleargrasp":
                    normal[..., 0] *= -1
                    mask = read_png(paths["mask"])
                else:
                    mask = np.linalg.norm(normal, axis=-1) > .5
            else:
                encoded = read_png(paths["normal"])[..., :3]
                if self.name in ("nyuv2", "scannet"):
                    mask = np.sum(encoded, axis=-1) > 0
                    normal = (encoded.astype(np.float32) / 255.) * 2. - 1.
                else:
                    normal = encoded.astype(np.float32) / 127.5 - 1.
                    mask = read_png(paths["mask"])
            if "intrinsics" in paths:
                intrinsic = np.load(paths["intrinsics"], allow_pickle=False)
                if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                    raise ValueError(f"Invalid intrinsics for {key}")
            if self.name == "clearpose" and read_png(paths["depth"]).shape != image.shape[:2]:
                raise ValueError(f"Rendered depth dimensions disagree for {key}")
        if self.name == "transnormal":
            if mask.ndim == 3:
                mask = mask.sum(-1)
            mask = mask.astype(np.float32) / 255. > .5
        elif mask.ndim == 3:
            mask = mask[..., 0] > 0
        else:
            mask = mask > 0
        h, w = image.shape[:2]
        if normal.shape != (h, w, 3) or mask.shape != (h, w) or not mask.any():
            raise ValueError(f"Invalid dimensions or empty evaluation mask: {key}")
        if not np.isfinite(normal[mask]).all():
            raise ValueError(f"Non-finite ground truth in the evaluation mask: {key}")
        rgb = torch.from_numpy(image.astype(np.float32) / 255.).permute(2, 0, 1)[None]
        rgb = (rgb - .5) / .5
        return rgb, torch.from_numpy(normal.copy()).permute(2, 0, 1)[None], torch.from_numpy(mask)[None, None]


def pad_image(image):
    """Apply the original evaluator's symmetric padding to a multiple of 32."""
    h, w = image.shape[-2:]
    dh, dw = (-h) % 32, (-w) % 32
    top, left = dh // 2, dw // 2
    padding = (left, dw - left, top, dh - top)
    if dh or dw:
        values = [(0. - mean) / std for mean, std in zip((.485, .456, .406), (.229, .224, .225))]
        image = torch.cat([F.pad(image[:, c:c+1], padding, value=values[c]) for c in range(3)], dim=1)
    return image, (top, left, h, w)
