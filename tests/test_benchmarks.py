"""Checks for common benchmark decoding and normalized inference inputs."""

import numpy as np
from PIL import Image
import torch

from transnormal import TransNormalPipeline
from transnormal.evaluation.data import pad_image, read_rgb


def test_rgba_rgb_input_ignores_alpha(tmp_path):
    pixels = np.array([[[12, 34, 56, 0], [78, 90, 12, 255]]], dtype=np.uint8)
    path = tmp_path / "rgb.png"
    Image.fromarray(pixels).save(path)
    assert np.array_equal(read_rgb(path), pixels[..., :3])


def test_padding_keeps_pixels_and_uses_original_channel_values():
    image = torch.rand(1, 3, 33, 35)
    padded, (top, left, height, width) = pad_image(image)
    assert padded.shape == (1, 3, 64, 64)
    assert torch.equal(padded[:, :, top:top + height, left:left + width], image)
    expected = torch.tensor([-0.485 / .229, -0.456 / .224, -0.406 / .225])
    assert torch.equal(padded[0, :, 0, 0], expected)


def test_normalized_bright_image_is_not_normalized_twice():
    image = torch.full((1, 3, 32, 32), .75)
    output = TransNormalPipeline.preprocess_image(None, image, torch.device("cpu"), torch.float32,
                                                  input_is_normalized=True)
    assert torch.equal(image, output)
    ordinary = TransNormalPipeline.preprocess_image(None, image, torch.device("cpu"), torch.float32)
    assert torch.equal(ordinary, image * 2 - 1)


def test_divisible_dimensions_do_not_add_padding():
    image = torch.zeros(1, 3, 480, 640)
    padded, crop = pad_image(image)
    assert padded is image
    assert crop == (0, 0, 480, 640)
