"""Read only the RGB/geometry fields used by the formal training objectives."""

import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import transform


def read_exr(path, channels):
    import Imath
    import OpenEXR

    handle = OpenEXR.InputFile(str(path))
    try:
        window = handle.header()["dataWindow"]
        shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
        return np.stack([np.frombuffer(handle.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(shape)
                         for channel in channels], axis=-1)
    finally:
        handle.close()


def rgb(source):
    with Image.open(source) as image:
        return image.convert("RGB")


def png_array(source):
    with Image.open(source) as image:
        return np.array(image)


class TrainingDataset(Dataset):
    def __init__(self, manifests, resolutions, random_flip=True):
        self.records, self.roots, self.kinds = [], [], []
        self.resolutions, self.random_flip = resolutions, random_flip
        for manifest in manifests:
            path = Path(manifest)
            with path.open() as handle:
                header = json.loads(next(handle))
                records = [json.loads(line) for line in handle if line.strip()]
            if header.get("format") != "transnormal-training-index-v1" or len(records) != header["samples"]:
                raise ValueError(f"Invalid training index: {path}")
            self.roots.append(path.parent / header["root"])
            self.kinds.append(header["kind"])
            self.records.append(records)

    def __len__(self):
        return sum(self.lengths)

    @property
    def lengths(self):
        return [len(records) for records in self.records]

    def __getitem__(self, index):
        dataset, sample, seed = index
        record, root, kind = self.records[dataset][sample], self.roots[dataset], self.kinds[dataset]
        changed, depth = None, None
        if "shard" in record:
            fields = {}
            with (root / record["shard"]).open("rb") as handle:
                for field, (offset, size) in record["files"].items():
                    handle.seek(offset)
                    fields[field] = io.BytesIO(handle.read(size))
            image, changed = rgb(fields["with_rgb.png"]), rgb(fields["change_rgb.png"])
            normal = png_array(fields["with_normal.png"])[..., :3].astype(np.float32) / 127.5 - 1
            mask = png_array(fields["with_mask_all.png"])
        else:
            image = rgb(root / record["image"])
            if kind == "cleargrasp":
                normal = read_exr(root / record["normal"], "RGB")
                mask = read_exr(root / record["mask"], "R") > 0
            elif kind == "hypersim":
                with h5py.File(root / record["normal"], "r") as handle:
                    normal = np.array(handle["dataset"])
                with h5py.File(root / record["depth"], "r") as handle:
                    distance = np.array(handle["dataset"])
                h, w = distance.shape
                x = np.linspace(-w / 2 + 0.5, w / 2 - 0.5, w, dtype=np.float32)
                y = np.linspace(-h / 2 + 0.5, h / 2 - 0.5, h, dtype=np.float32)
                depth = distance / np.sqrt(x[None] ** 2 + y[:, None] ** 2 + np.float32(886.81) ** 2) * 886.81
                mask = np.ones((*normal.shape[:2], 1), dtype=bool)
            elif kind == "vkitti":
                normal = png_array(root / record["normal"])[..., :3]
                depth = png_array(root / record["depth"]).astype(np.float64) / 100
                mask = np.ones((*normal.shape[:2], 1), dtype=bool)
            else:
                changed = rgb(root / record["changed"])
                normal = png_array(root / record["normal"])[..., :3].astype(np.float32) / 127.5 - 1
                mask = png_array(root / record["mask"])
        if kind == "transnormal":
            if mask.ndim == 3:
                mask = mask.sum(-1)
            mask = (mask[..., None].astype(np.float32) / 255) > 0.5
        result = transform(kind, image, normal, mask, self.resolutions[dataset], seed,
                           self.random_flip, depth, changed)
        if any(not value.isfinite().all() for value in result.values()):
            raise ValueError(f"Non-finite training data: {kind} sample {sample}")
        result["dataset_name"] = kind
        return result
