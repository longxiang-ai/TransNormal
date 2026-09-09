# Training TransNormal

This guide covers the final model configuration. The existing inference commands and model format remain supported.

## Environment and pretrained components

Use Python 3.10 and eight CUDA GPUs for the formal configuration (batch size 4 per GPU, effective batch size 32). Install PyTorch 2.4.0 and torchvision 0.19.0 with the CUDA build suitable for your system, then install `requirements-training.txt`, which also includes evaluation and data-preparation dependencies.

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-training.txt
```

The UNet is initialized from `jingheya/lotus-normal-d-v1-1`. Its VAE and scheduler are loaded from the same repository by default; a separate Stable Diffusion 2 download is not required. `model.vae_model` optionally accepts a compatible repository or local directory containing a `vae/` component. The distributed Lotus VAE stores FP16 weights; converting these to BF16 can differ slightly from directly converting the original FP32 VAE.

Request access to [DINOv3 ViT-H+/16](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) and authenticate with Hugging Face before downloading it. Model revisions are pinned in `configs/train.yaml`. For offline training, download all required components first, set `initial_model` and `dino_model` to their local directories, and set `local_files_only: true`. Tokens belong in your local Hugging Face authentication store, never in the configuration or source files.

Initialization checks all loaded parameters. Missing pretrained components cause an error rather than a fallback to a random model. The DINO backbone and VAE remain frozen; the linear projector and UNet are trained.

## Prepare the four datasets

Follow the [dataset preparation guide](data.md) for download sources, file layouts, fixed subsets and dataset-specific transforms. Dataset licenses and access requirements remain those of their providers.

| Dataset | Expected files | Training preprocessing |
|---|---|---|
| ClearGrasp synthetic training | Scene directories containing RGB images, `camera-normals` and `variant-masks` EXRs | Random square crop, resize to 576, camera-normal x conversion |
| TransNormal-Synthetic | Public WebDataset `.tar` shards, or the scene/view directory layout | Paired original/material-changed RGB, 400 × 400, supplied training split |
| Hypersim | `train/` with tonemapped RGB, distance and camera-normal HDF5 files | Short edge 576, camera-facing normal alignment |
| Virtual KITTI | `Scene*/condition/frames/{rgb,depth,normal}/Camera_*` | Scenes 02/06/18/20; 352 × 1216 bottom-center crop; valid depth and non-black normals |

Use the supplied 39,648-image Hypersim subset of the official training partition. The supplied TransNormal split selects 3,555 training views; do not include held-out views. Virtual KITTI scene 01 is excluded from training. Its normal maps can be generated from depth and intrinsics if they are not already prepared:

```bash
python -m transnormal.training.data.prepare_vkitti_normals --root data/virtualkitti
```

The conversion skips existing normal files. The CPU implementation can be slow on a complete dataset; `--device cuda:0` is an optional preprocessing device selection, not a training command.

Build indices once. Images remain in their original locations; index paths are relative to each index file so the data tree can be relocated together.

```bash
python -m transnormal.training.data.index cleargrasp --root data/cleargrasp --output data/index/cleargrasp.jsonl
python -m transnormal.training.data.index transnormal --root data/transnormal_shards --shards --output data/index/transnormal.jsonl
python -m transnormal.training.data.index hypersim --root data/hypersim --output data/index/hypersim.jsonl
python -m transnormal.training.data.index vkitti --root data/virtualkitti --output data/index/vkitti.jsonl
python train.py --config configs/train.yaml --check-data
```

For the original TransNormal scene/view layout, omit `--shards`. Both formats use the same split and transformations. Indexing fails on missing required fields rather than silently changing the dataset. Choose a new index filename when rebuilding an index.

Normals use camera-space x/y/z channels. Horizontal augmentation also negates the x component. The formal recipe uses full-image supervision for ClearGrasp/Hypersim, the all-object mask for TransNormal and valid-depth regions with non-black normals for Virtual KITTI. A latent cell is supervised only when its entire 8 × 8 pixel region is valid.

## Formal recipe

The training objective combines normal latent regression, RGB latent reconstruction, paired material-invariant latent consistency (on TransNormal pairs), and edge-weighted Haar supervision. The wavelet coefficient is 0.1. The VAE decoder is frozen but remains differentiable with respect to predicted latents.

`configs/train.yaml` uses 15,000 steps, timestep 999, BF16, UNet LR 3e-5, projector LR 3e-4, 300 warmup steps and dataset probabilities `[0.35, 0.15, 0.45, 0.05]` in the table order. Virtual KITTI uses a fixed 352 × 1216 crop. Loader worker count affects throughput rather than the training recipe.

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train.py --config configs/train.yaml
```

The UNet and projector share a distributed wrapper so both receive synchronized gradients. Dataset choice is shared across ranks, and each global batch is split across GPUs. The sampler records its position implicitly through the training step and deterministic seeds; worker prefetch does not alter the resumed sample/augmentation sequence.

## Save, resume and use the trained model

Each `checkpoint-NNNNNN/` contains model, optimizer, LR scheduler and per-process random state, together with a recipe signature. `export-NNNNNN/` contains only the inference components and projector. Resuming requires the same configuration, dataset indices and GPU count. `output_dir` and loader worker count may differ. Only load checkpoint state from trusted training runs.

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train.py \
  --config configs/train.yaml --resume outputs/transnormal/checkpoint-000500
```

To pause at a chosen absolute step while preserving the LR schedule, use `--stop-after`. Existing checkpoints and exports are never overwritten. Keep `checkpoint-*` for continued training; distribute `export-*` when sharing inference weights. Model exports exclude local source paths from their component JSON files.

Use the exported directory as the model path in the existing inference commands. Load its `cross_attention_projector.pt` using `create_dino_encoder` exactly as for the original released weights. Inference defaults to timestep 999, matching the final protocol.

```bash
python inference.py --input examples/input/test.png --output normal.png \
  --model_path outputs/transnormal/export-015000 \
  --projector_path outputs/transnormal/export-015000/cross_attention_projector.pt \
  --dino_path weights/dinov3_vith16plus
```

## Smoke test

The optional CPU regression tests in `tests/test_training.py` check sampling and resume, geometry transforms, loss gradients, frozen components, and checkpoint/export handling. They use small test inputs, require no downloaded datasets or model weights, and are not called by normal training.

```bash
pip install "pytest>=8,<10"
python -m pytest tests/test_training.py -q
```

For a real GPU test, choose a fresh `output_dir` in a copy of the config and run:

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train.py \
  --config configs/smoke-local.yaml --smoke-test --stop-after 4
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 train.py \
  --config configs/smoke-local.yaml --smoke-test --resume outputs/smoke/checkpoint-000004
```

The smoke flag limits the run to six steps and cycles through all four datasets. It checks finite losses/gradients, updates to both trainable components, frozen parameters and equality of projector weights across ranks. The second command restores training state and continues through step six. Finally load the export with the existing inference script and inspect its output.
