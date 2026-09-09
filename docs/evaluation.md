# Paper benchmark evaluation

Evaluate all six paper benchmarks with `evaluate.py`. Data preparation and fixed test splits are listed below.

## Benchmarks and fixed splits

| CLI name | Dataset | Images | Native H × W | Ground truth and scoring region |
|---|---|---:|---|---|
| `cleargrasp` | ClearGrasp Synthetic test | 408 | 576 × 1024 | Float EXR camera normals, x flipped; segmentation mask > 0 |
| `transnormal` | TransNormal-Synthetic test | 395 | 800 × 800 | RGB normal PNG; **all-object** mask, not transparent-only mask |
| `clearpose` | ClearPose custom normal subset | 120 | 480 × 640 | Rendered normal PNG; rendered object mask > 0 |
| `nyuv2` | NYUv2 | 654 | 480 × 640 | GeoNet normal PNG from the DSINE evaluation package; nonzero encoded RGB |
| `scannet` | ScanNet | 300 | 480 × 640 | FrameNet normal PNG and split from the DSINE package; nonzero encoded RGB |
| `ibims` | iBims-1 | 100 | 480 × 640 | Plane-fit float EXR normals from the DSINE package; vector norm > 0.5 |

The first three appear in the main normal-comparison table; the other three are also evaluated in the appendix. The lists are in [`configs/splits`](../configs/splits): `cleargrasp_test_synthetic.txt`, `transnormal_test.txt`, `clearpose_test.txt`, `nyuv2_test.txt`, `scannet_test.txt`, and `ibims_test.txt`. They total **1,977 images**. Do not replace them with a training list, a random subset, or a dataset provider's different test partition.

## Prepare data

Install the shared training and evaluation dependencies with `pip install -r requirements-training.txt`. OpenEXR is required for ClearGrasp and iBims-1; the OpenGL rendering dependencies are only needed when preparing ClearPose. Keep each provider's data and license terms with the downloaded archives.

### ClearGrasp Synthetic

Download the **Testing and Validation Datasets** archive linked by the [official ClearGrasp repository](https://github.com/Shreeyak/cleargrasp#datasets). Extract it and point the configuration at the directory containing `synthetic-test/`. Only the synthetic test subset is selected; real and validation subsets are not included in these 408 images.

```text
data/cleargrasp_eval/synthetic-test/<object>/
  rgb-imgs/<id>-rgb.jpg
  camera-normals/<id>-cameraNormals.exr
  segmentation-masks/<id>-segmentation-mask.png
```

Use the supplied camera-normal EXRs and segmentation masks directly. The loader handles the RGB channel order and x-axis conversion. Do not convert EXRs into visualization PNGs or flip the x axis again.

### TransNormal-Synthetic

Download the public tar shards from [TransNormal-Synthetic](https://huggingface.co/datasets/Longxiang-ai/TransNormal-Synthetic):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Longxiang-ai/TransNormal-Synthetic', repo_type='dataset', allow_patterns=['*.tar'], local_dir='data/transnormal_shards')"
python scripts/index_transnormal_eval.py \
  --root data/transnormal_shards --output data/indices/transnormal_test.jsonl
```

Set `transnormal.index` in the evaluation JSON to `data/indices/transnormal_test.jsonl`, or pass `--transnormal-index` on the command line. The index selects the fixed 395 scene/view keys using shard metadata; it does not randomly split the public samples. Do not pass the training index. Keep the index with its shards; its relative root is resolved against the index location.

The evaluator reads `with_rgb.png`, `with_normal.png` and `with_mask_all.png`. It discards RGB alpha, decodes normals as `RGB / 127.5 - 1`, and thresholds the all-object mask at 0.5 after division by 255. Multi-channel masks are summed before thresholding, matching the original loader. GT is scored at 800 × 800. Material-changed RGB, depth and camera metadata are not model inputs for this benchmark.

The original scene/view layout is also supported without an index:

```text
data/transnormal/<scene>/views/view_<id>/with_transparent/
  rgb_.png
  normal_<id>.png
  masks/all_<id>.png
```

### ClearPose

Follow [ClearPose preparation](data.md#clearpose) using downsample-100 and the fixed 120-frame list. Set the prepared output directory as `clearpose.root`.

### NYUv2, ScanNet and iBims-1

Download **`dsine_eval.zip`** through the Evaluation section of the [official DSINE repository](https://github.com/baegwangbin/DSINE#evaluation). Use this prepared normal-evaluation package, rather than independently converting arbitrary raw depth downloads. Its three normal annotation sources and masks are part of the evaluation protocol. Extraction should provide:

```text
data/dsine_eval/
  nyuv2/test/<id>_img.png
  nyuv2/test/<id>_normal.png
  nyuv2/test/<id>_intrins.npy
  scannet/<scene>/<id>_img.png
  scannet/<scene>/<id>_normal.png
  scannet/<scene>/<id>_intrins.npy
  ibims/<name>_img.png
  ibims/<name>_normal.exr
  ibims/<name>_intrins.npy
```

Set each root to the corresponding dataset subdirectory, removing any extra wrapper directory introduced during extraction. NYUv2 and ScanNet decode PNGs as `(RGB / 255) * 2 - 1` and ignore pixels whose encoded RGB sum is zero. iBims-1 reads floating-point EXRs without a channel sign flip. Intrinsics are checked for consistency but are not passed to the model. No additional NYUv2 crop, depth cutoff, GT regeneration or mask erosion is applied. Data and annotation rights remain with their respective providers; this repository does not redistribute the DSINE implementation or dataset archive.

## Validate, infer and score

Edit the six roots in [`configs/evaluation.json`](../configs/evaluation.json), or copy that file and pass `--config your-evaluation.json`. Relative paths are resolved from the current working directory. Run commands from the repository root.

```bash
# Decode every selected RGB, GT and mask before loading the model.
python evaluate.py --dataset all --check-data

# One CUDA process; each fixed sample is evaluated exactly once.
python evaluate.py --dataset all \
  --model-path weights/transnormal --dino-path weights/dinov3_vith16plus \
  --seed 42 --output outputs/paper_eval

# Evaluate only one dataset, optionally overriding its data root.
python evaluate.py --dataset nyuv2 --data data/dsine_eval/nyuv2 \
  --model-path weights/transnormal --dino-path weights/dinov3_vith16plus \
  --seed 42 --output outputs/nyuv2_eval
```

Use the released model and DINOv3 component from the [README](../README.md#download-model-weights), or a training export. The projector defaults to `cross_attention_projector.pt` inside the model directory; `--projector-path` overrides it. Retain the exact model files or their download revisions with your results. `--limit 1` runs one smoke-test image per dataset and explicitly marks the result as partial; omit it for a benchmark.

Inference uses processing resolution 768 and timestep 999, then restores predictions to the input dimensions.

Every run requires a new output directory:

```text
outputs/paper_eval/
  summary.csv
  summary.json
  <dataset>/metrics.json
  <dataset>/per_image.json
  <dataset>/initial_rng.pt
  <dataset>/predictions/<sample-key>.npy
```

Prediction filenames replace any sample-key extension with `.npy`; arrays are H × W × 3 float normals in [-1, 1]. `summary.csv` has one row per dataset, not an average across different datasets.

Saved predictions can be rescored on CPU:

```bash
python evaluate.py --dataset all --prediction-dir outputs/paper_eval \
  --output outputs/paper_rescored
python evaluate.py --dataset nyuv2 \
  --prediction-dir outputs/paper_eval/nyuv2/predictions \
  --output outputs/nyuv2_rescored
```

For `all`, pass the parent results directory; for one dataset, pass its `predictions` directory. Score raw float arrays, not PNG visualizations. All datasets use the same `evaluate.py --dataset <name>` entry point and output layout.

## Metrics

Angular errors are pooled over all valid pixels in the selected dataset. Results include mean, median, RMSE and percentages below 5°, 7.5°, 11.25°, 22.5° and 30° (`a1`–`a5`). Per-image metrics are saved separately. Use floating-point `.npy` predictions for scoring.

## Replay a run

Use the same model, inputs and environment, and restore the saved RNG:

```bash
python evaluate.py --dataset all \
  --model-path weights/transnormal --dino-path weights/dinov3_vith16plus \
  --rng-state outputs/paper_eval --output outputs/paper_replay
```

For an individual dataset, pass its `initial_rng.pt` file. Use RNG files produced by your own or trusted runs.
