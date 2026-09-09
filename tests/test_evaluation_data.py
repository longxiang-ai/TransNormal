"""Regression checks for fixed training subsets and the normal evaluation protocol."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from transnormal.evaluation.data import Benchmark
from transnormal.evaluation.metrics import normal_errors, summarize
from transnormal.training.data.index import create_index


def test_hypersim_index_excludes_unlisted_frames_and_rejects_missing(tmp_path):
    root = tmp_path / "hypersim"
    preview = root / "train/scene/images/scene_cam_00_final_preview"
    geometry = preview.with_name("scene_cam_00_geometry_hdf5")
    preview.mkdir(parents=True)
    geometry.mkdir()
    for frame in (0, 1):
        (preview / f"frame.{frame:04d}.tonemap.jpg").touch()
        for field in ("normal_cam", "depth_meters"):
            (geometry / f"frame.{frame:04d}.{field}.hdf5").touch()
    split = tmp_path / "split.txt"
    split.write_text("scene/images/scene_cam_00_final_preview/frame.0001.tonemap.jpg\n")
    output = tmp_path / "index.jsonl"
    create_index("hypersim", root, output, split)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["samples"] == 1
    assert rows[1]["image"].endswith("0001.tonemap.jpg")
    (geometry / "frame.0001.normal_cam.hdf5").unlink()
    with pytest.raises(FileNotFoundError):
        create_index("hypersim", root, tmp_path / "missing.jsonl", split)


def test_signed_angle_and_mask():
    target = torch.tensor([1., 0., 0.]).view(1, 3, 1, 1).expand(1, 3, 1, 3)
    prediction = torch.tensor([[1., 0., -1.], [0., 1., 0.], [0., 0., 0.]]).view(1, 3, 1, 3)
    mask = torch.tensor([True, False, True]).view(1, 1, 1, 3)
    assert torch.equal(normal_errors(prediction, target, mask), torch.tensor([0., 180.]))


def test_pixel_pooling_and_strict_thresholds():
    errors = torch.tensor([0., 0., 0., 30.])
    metrics = summarize(errors)
    assert metrics["mean"] == 7.5
    assert metrics["median"] == 0.
    assert metrics["rmse"] == 15.
    assert metrics["a5"] == 75.


def test_rendered_mask_is_not_filtered_by_depth(tmp_path):
    arrays = {"image": np.zeros((32, 32, 3), dtype=np.uint8),
              "normal": np.full((32, 32, 3), 127, dtype=np.uint8),
              "depth": np.full((32, 32), 2000, dtype=np.uint16),
              "mask": np.full((32, 32), 255, dtype=np.uint8)}
    folder = tmp_path / "set1/scene1"
    folder.mkdir(parents=True)
    suffixes = {"image": "color", "normal": "rendered_normal", "depth": "rendered_depth", "mask": "rendered_mask"}
    for name, array in arrays.items():
        Image.fromarray(array).save(folder / f"000100-{suffixes[name]}.png")
    split = tmp_path / "split.txt"
    split.write_text("set1/scene1/000100\n")
    _, _, mask = Benchmark("clearpose", tmp_path, split=split).load("set1/scene1/000100")
    assert mask.sum().item() == 1024
