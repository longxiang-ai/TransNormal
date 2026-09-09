import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from transnormal.dino_encoder import DINOv3Encoder
from transnormal.training.data.mixture import MixtureBatchSampler
from transnormal.training.data.transforms import transform
from transnormal.training.losses import haar_dwt2d, latent_mask, masked_mse, task_embeddings, wavelet_loss
from transnormal.training.initialization import TrainableModel, load_component


def test_distributed_sampling_and_resume():
    options = dict(lengths=[256, 256, 512, 256], probabilities=[.35, .15, .45, .05], batch_size=4,
                   world_size=8, seed=42, steps=80)
    ranks = [list(MixtureBatchSampler(rank=rank, **options)) for rank in range(8)]
    for step in range(80):
        batch = [sample for rank in ranks for sample in rank[step]]
        assert len({sample[0] for sample in batch}) == 1
        assert len({sample[1] for sample in batch}) == 32
        assert len({sample[2] for sample in batch}) == 32
    assert list(MixtureBatchSampler(rank=3, start=19, **options)) == ranks[3][19:]


def test_mask_erosion_and_empty_supervision():
    mask = torch.ones(1, 1, 16, 24, dtype=torch.bool)
    mask[..., 0, 0] = False
    valid = latent_mask(mask)
    assert valid.shape == (1, 4, 2, 3)
    assert not valid[..., 0, 0].any() and valid.sum() == 20
    prediction = torch.randn(1, 4, 2, 3, requires_grad=True)
    loss = masked_mse(prediction, torch.zeros_like(prediction), torch.zeros_like(valid))
    loss.backward()
    assert loss.item() == 0 and prediction.grad is not None


def test_haar_and_wavelet_gradient():
    constant = torch.ones(2, 3, 17, 19)
    low, high = haar_dwt2d(constant)
    assert torch.equal(low, torch.ones_like(low)) and torch.count_nonzero(high) == 0
    target = torch.randn(2, 3, 18, 20)
    assert wavelet_loss(target, target, torch.ones(2, 1, 18, 20, dtype=torch.bool)).item() == 0
    prediction = (target + .1).requires_grad_()
    loss = wavelet_loss(prediction, target, torch.ones(2, 1, 18, 20, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


def test_normal_flip_and_geometry_mask():
    image = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
    normal = np.zeros((16, 16, 3), dtype=np.float32)
    normal[..., 0], normal[..., 2] = .6, .8
    mask = np.ones((16, 16, 1), dtype=bool)
    seed = next(seed for seed in range(30) if random.Random(seed).random() > .5)
    result = transform("transnormal", image, normal, mask, 16, seed, changed=image)
    assert torch.all(result["normal_values"][0] == -.6)
    assert torch.all(result["normal_values"][2] == .8)
    assert torch.equal(result["pixel_values"], result["rgb_change_values"])


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1280))

    def forward(self, pixel_values, **kwargs):
        features = pixel_values.mean((1, 2, 3))[:, None, None] * self.weight[None, None, :]
        return SimpleNamespace(last_hidden_state=features.expand(-1, 7, -1))


def test_frozen_dino_keeps_projector_gradients():
    encoder = DINOv3Encoder()
    encoder.dino_backbone = TinyBackbone().requires_grad_(False)
    encoder._use_hf_interface, encoder._is_loaded = True, True
    encoder.train()
    assert not encoder.dino_backbone.training and encoder.cross_attention_projector.training
    encoder(torch.rand(2, 3, 32, 32))["cross_attention_features"].square().mean().backward()
    assert encoder.dino_backbone.weight.grad is None
    assert encoder.cross_attention_projector.weight.grad.abs().sum() > 0


def test_missing_initializer_weights_are_not_accepted():
    class Incomplete:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs["subfolder"] == "unet"
            return object(), {"missing_keys": ["conv_in.weight"]}
    with pytest.raises(RuntimeError, match="Incomplete"):
        load_component(Incomplete, "owner/model", "unet")


def test_tiny_model_training_export_and_reload(tmp_path):
    from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
    from transnormal.pipeline import TransNormalPipeline
    from transnormal.training.checkpointing import export_model
    from transnormal.training.trainer import loss_terms

    unet = UNet2DConditionModel(sample_size=4, in_channels=4, out_channels=4, layers_per_block=1,
                               block_out_channels=(8, 16), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                               up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=8,
                               attention_head_dim=2, norm_num_groups=4, class_embed_type="projection",
                               projection_class_embeddings_input_dim=4)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4,
                        block_out_channels=(8, 8, 8, 8), layers_per_block=1,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4).eval().requires_grad_(False)
    model = TrainableModel(unet, torch.nn.Linear(6, 8))
    rgb_latents = torch.randn(1, 4, 4, 4)
    tokens = torch.randn(1, 4, 6)
    prediction, changed = model(rgb_latents, tokens, 999, rgb_latents * .7, tokens * .8)
    terms = loss_terms(prediction, changed, torch.randn_like(rgb_latents), rgb_latents,
                       torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32, dtype=torch.bool), vae, torch.float32)
    sum(terms.values()).backward()
    assert all(torch.isfinite(value) for value in terms.values())
    assert model.projector.weight.grad.abs().sum() > 0
    assert model.unet.conv_out.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in vae.parameters())
    export_model(tmp_path / "export", model, vae, PNDMScheduler())
    pipe = TransNormalPipeline.from_pretrained(tmp_path / "export", local_files_only=True)
    assert pipe.unet.config.class_embed_type == "projection"
    assert torch.equal(pipe.unet.conv_out.weight, model.unet.conv_out.weight)
    assert (tmp_path / "export/cross_attention_projector.pt").is_file()
    for path in (tmp_path / "export").rglob("*.json"):
        assert "_name_or_path" not in path.read_text()


def test_checkpoint_restores_optimizer_and_rng(tmp_path):
    from accelerate import Accelerator
    from transnormal.training.checkpointing import load_checkpoint, save_checkpoint

    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.backward(model(torch.ones(2, 2)).square().mean())
    optimizer.step()
    scheduler.step()
    checkpoint = save_checkpoint(accelerator, tmp_path, 1, "recipe")
    expected = torch.rand(5)
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer_step = next(iter(optimizer.state.values()))["step"].item()
    with torch.no_grad():
        next(model.parameters()).add_(5)
    assert load_checkpoint(accelerator, checkpoint, "recipe") == 1
    assert torch.equal(torch.rand(5), expected)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in weights.items())
    assert next(iter(optimizer.state.values()))["step"].item() == optimizer_step
    with pytest.raises(ValueError):
        load_checkpoint(accelerator, checkpoint, "different-recipe")


def test_shard_index_preserves_scene_keys(tmp_path):
    import io
    import tarfile
    from transnormal.training.data.index import TN_FIELDS, shard_records

    keys = ["scene_a/view_0001", "scene_b/view_0001"]
    with tarfile.open(tmp_path / "samples.tar", "w") as archive:
        for key in keys:
            fields = {field: b"image-bytes" for field in TN_FIELDS}
            fields["meta.json"] = json.dumps(dict(zip(("scene_name", "view_id"), key.split("/")))).encode()
            for field, content in fields.items():
                info = tarfile.TarInfo(key + "." + field)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    indexed = shard_records(tmp_path, keys)
    assert [record["key"] for record in indexed] == keys
    with (tmp_path / "samples.tar").open("rb") as handle:
        for record in indexed:
            for offset, size in record["files"].values():
                handle.seek(offset)
                assert handle.read(size) == b"image-bytes"


def test_depth_plane_recovers_camera_normal():
    from transnormal.training.data.prepare_vkitti_normals import fit_normals

    height, width = 24, 32
    fx, fy, cx, cy = 43.7, 45.2, 15.1, 11.4
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                          torch.arange(width, dtype=torch.float64), indexing="ij")
    rays = torch.stack(((x - cx) / fx, (y - cy) / fy, torch.ones_like(x)))
    expected = torch.tensor([.2, -.3, 1.], dtype=torch.float64)
    expected /= expected.norm()
    depth = 2. / (rays * expected[:, None, None]).sum(0)
    normal, mask = fit_normals(depth, [fx, fy, cx, cy])
    assert mask[3:-3, 3:-3].all()
    assert torch.allclose(normal[:, 3:-3, 3:-3], expected[:, None, None].expand(3, height - 6, width - 6),
                          atol=1e-8, rtol=0)


def test_vkitti_flip_preserves_invalid_black_normals():
    image = Image.fromarray(np.zeros((352, 1216, 3), dtype=np.uint8))
    normal = np.zeros((352, 1216, 3), dtype=np.uint8)
    normal[:, :608] = [0, 127, 255]
    depth = np.ones((352, 1216), dtype=np.float64) * 2
    seed = next(s for s in range(30) if torch.rand((), generator=torch.Generator().manual_seed(s)) > .5)
    batch = transform("vkitti", image, normal, np.ones((352, 1216, 1)), 352, seed, depth=depth)
    result = batch["normal_values"]
    assert torch.all(result[:, :, :608] == -1)
    assert not batch["valid_mask_values"][..., :608].any()
    assert batch["valid_mask_values"][..., 608:].all()
    assert torch.all(result[0, :, 608:] == 1)
    assert torch.all(result[2, :, 608:] == 1)
    assert torch.allclose(result[1, :, 608:], torch.full_like(result[1, :, 608:], 127 / 127.5 - 1))


@pytest.mark.parametrize("shape", [(352, 1216), (176, 608)])
@pytest.mark.parametrize("flip", [False, True])
def test_vkitti_mask_combines_depth_and_normals_after_resize(shape, flip):
    height, width = shape
    image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    normal[:, width // 4:] = [0, 127, 255]
    depth = np.full((height, width), 2., dtype=np.float64)
    depth[:, width // 4:width // 2] = 80.
    depth[:, width // 2:3 * width // 4] = 0.
    seed = next(s for s in range(30) if bool(torch.rand((), generator=torch.Generator().manual_seed(s)) > .5) == flip)
    result = transform("vkitti", image, normal, np.ones((height, width, 1)), 352, seed, depth=depth)
    expected = torch.zeros((1, 352, 1216), dtype=torch.bool)
    expected[..., :304] = flip
    expected[..., 912:] = not flip
    assert torch.equal(result["valid_mask_values"], expected)


@pytest.mark.parametrize("shape", [(16, 20), (17, 19)])
def test_wavelet_ignores_invalid_targets_and_predictions(shape):
    torch.manual_seed(11)
    target = torch.rand(1, 3, *shape) * 2 - 1
    prediction = torch.rand_like(target).requires_grad_()
    mask = torch.ones(1, 1, *shape, dtype=torch.bool)
    for y, x in [(2, 2), (2, 5), (5, 2), (5, 5), (shape[0] - 1, shape[1] - 1)]:
        mask[..., y, x] = False
    invalid = ~mask.expand_as(target)
    reference = wavelet_loss(prediction, target, mask)
    gradient = torch.autograd.grad(reference, prediction)[0]
    changed_target = target.clone()
    changed_target[invalid] = 100.
    changed_prediction = prediction.detach().clone()
    changed_prediction[invalid] = -100.
    changed_prediction.requires_grad_()
    actual = wavelet_loss(changed_prediction, changed_target, mask)
    actual_gradient = torch.autograd.grad(actual, changed_prediction)[0]
    assert torch.equal(reference, actual)
    assert torch.equal(gradient, actual_gradient)
    assert torch.count_nonzero(gradient[invalid]) == 0
    assert gradient.abs().sum() > 0


def test_wavelet_without_valid_blocks_has_zero_loss_and_gradient():
    prediction = torch.randn(1, 3, 16, 16, requires_grad=True)
    target = torch.randn_like(prediction)
    for mask in [torch.zeros(1, 1, 16, 16, dtype=torch.bool),
                 torch.ones(1, 1, 16, 16, dtype=torch.bool)]:
        mask[..., 1::2, 1::2] = False
        loss = wavelet_loss(prediction, target, mask)
        gradient = torch.autograd.grad(loss, prediction)[0]
        assert loss.item() == 0
        assert torch.count_nonzero(gradient) == 0
