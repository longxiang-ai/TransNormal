# Data preparation

Use these inputs with the [training guide](training.md). The following sections describe training data preparation and the ClearPose evaluation data pipeline.

## Downloads and required preparation

Download data directly from its providers and follow their access and license terms:

- [ClearGrasp](https://github.com/Shreeyak/cleargrasp): synthetic training images and geometry annotations.
- [TransNormal-Synthetic](https://huggingface.co/datasets/Longxiang-ai/TransNormal-Synthetic): uncompressed WebDataset `.tar` shards.
- [Hypersim](https://github.com/apple/ml-hypersim): tonemapped RGB previews and geometry HDF5 files.
- [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/): RGB, depth and camera metadata.

ClearPose is used only for evaluation in this recipe; see [ClearPose](#clearpose) below.

### ClearGrasp

Extract synthetic training data so that each scene is directly below `data/cleargrasp/`. Required files for an image ID are:

```text
data/cleargrasp/<scene>/
  rgb-imgs/<id>-rgb.jpg
  camera-normals/<id>-cameraNormals.exr
  variant-masks/<id>-variantMasks.exr
```

Some archives use `transparent-rgb-imgs/` instead of `rgb-imgs/`; both are supported, as are RGB PNGs. Preserve the filenames from the download. Do not place duplicate RGB copies in both directories. Install the OpenEXR dependency from `requirements-training.txt`; normal EXRs are read as floating-point RGB channels, not as display images. The reviewed training tree contains 45,454 samples.

The transform randomly crops a square, resizes to 576 × 576, flips the normal x component to the training convention, and clamps normal values to [-1, 1]. RGB uses PIL bilinear interpolation; normals use nearest-neighbor interpolation. The formal objective supervises the full image. The variant mask is required by the source layout, but it does not restrict the final loss to transparent pixels.

### TransNormal-Synthetic

Place the downloaded `.tar` shards below `data/transnormal_shards/` and use `--shards` when indexing. No extraction is needed. The index records byte offsets, so shards must remain uncompressed and unchanged after indexing. Each selected sample requires `with_rgb.png`, `change_rgb.png`, `with_normal.png`, `with_mask_all.png`, and scene/view identification in `meta.json`.

[`configs/splits/transnormal_train.txt`](../configs/splits/transnormal_train.txt) fixes the 3,555 training views. The original directory format is also supported:

```text
<scene>/views/view_<n>/
  with_transparent/rgb_.png
  with_transparent/normal_<n padded to four digits>.png
  with_transparent/masks/all_<n padded to four digits>.png
  change_material/rgb_.png
```

RGB and changed-material RGB must depict the same scene and view. Both receive identical spatial augmentation and are resized to 400 × 400 using tensor bilinear interpolation without antialiasing. Normals decode as `RGB / 127.5 - 1`; normals and masks use nearest-neighbor resizing. Supervision uses the all-object mask, not only the transparent-object mask. Horizontal flips negate the normal x component. Missing changed-material images must not be replaced with the original image: that would remove the intended consistency supervision.

### Hypersim

The formal recipe uses **39,648 images**, a subset of the official training partition, fixed by [`configs/splits/hypersim_train.txt`](../configs/splits/hypersim_train.txt). All entries were checked against the provider's [scene split metadata](https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv). Using every official training frame would enlarge this recipe's training set.

Download the scenes referenced in this list with the provider's download tools. Put their scene directories under `data/hypersim/train/`; moving, copying or linking each downloaded scene directory is sufficient. Each selected RGB requires the following matching files:

```text
data/hypersim/train/ai_001_001/images/
  scene_cam_00_final_preview/frame.0000.tonemap.jpg
  scene_cam_00_geometry_hdf5/frame.0000.normal_cam.hdf5
  scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
```

Both HDF5 files must contain the `dataset` array. Use the supplied tonemapped preview, not linear HDR RGB, and camera-space normals, not world-space normals. The depth file stores distance along the viewing ray. The loader converts it to camera z depth using focal length 886.81, as in the original recipe. It then applies the original camera-facing normal alignment, resizes the short edge to 576 (normally 576 × 768), and uses full-image supervision. Do not pre-flip or reorient the stored normals yourself. Indexing only includes the fixed list, even if other images are present; it errors when any listed triplet is missing.

### Virtual KITTI 2

Merge the extracted RGB and depth trees by their existing `Scene*/condition/frames/` paths, and place the provided `intrinsic.txt` at each scene/condition root. Do not replace one extracted tree with another. The training index uses scenes 02, 06, 18 and 20, both cameras, and the ten conditions listed in the indexer; scene 01 is excluded. The reviewed tree has 33,580 images.

Generate the normal PNGs before indexing:

```bash
python -m transnormal.training.data.prepare_vkitti_normals --root data/virtualkitti
```

This uses the provided depth and intrinsics, local 5 × 5 neighborhoods, plane fitting through an eigendecomposition, depth range (0.001, 80) meters, relative depth threshold 0.05 and at least four neighbors. Existing normal files are skipped; use a fresh output tree when changing the conversion. `--device cuda:0` optionally accelerates preparation.

```text
data/virtualkitti/Scene02/clone/
  intrinsic.txt
  frames/rgb/Camera_0/rgb_00000.jpg
  frames/depth/Camera_0/depth_00000.png
  frames/normal/Camera_0/normal_00000.png
```

Depth PNG values are centimeters and are divided by 100. Training uses a bottom-center 352 × 1216 crop. Valid supervision requires both `1e-5 < depth < 80` meters and a non-black normal PNG pixel. A horizontal flip also flips the encoded normal x channel while preserving invalid black pixels. These depth-derived normals are an additional preprocessing product; the RGB/depth download alone is not training-ready.

## Normal encoding

Read normal PNGs as RGB and decode with `RGB / 127.5 - 1`. EXR and HDF5 normals are already signed floating-point vectors; do not apply the PNG conversion to them. An 8-bit PNG cannot represent a zero component exactly: values 127 and 128 lie on either side of zero.

Dataset-specific axis conversion and horizontal-flip correction are performed by the loaders. Preserve source annotations without pre-flipping channels or converting them to display images.

## Shared checks

Build the four indices with the commands in the [training guide](training.md), then run its `--check-data` command. RGB is normalized to [-1, 1]. Spatial augmentation is shared among paired fields. A latent cell is valid only if every pixel in its 8 × 8 region is valid. A Haar coefficient is supervised only when all four pixels in its 2 × 2 region are valid; empty regions contribute zero normal loss. Indexing checks required file presence; loading checks finite tensors. Neither check estimates model accuracy.

## ClearPose

### Download data

Use **ClearPose downsample-100**. It contains the RGB frames needed by the fixed [120-frame test list](../configs/splits/clearpose_test.txt); the full-frame dataset is not required. This subset spans 24 scenes in sets 1–5.

Obtain the data from the [ClearPose project](https://github.com/opipari/ClearPose), including the selected scenes' `metadata.mat` and object models. Download [`objects.csv`](https://raw.githubusercontent.com/opipari/ClearPose/main/data/objects.csv) for the object-ID mapping. Keep the original frame numbers and mesh companion files.

```text
data/clearpose_downsample_100/
  set1/scene1/metadata.mat
  set1/scene1/000900-color.png
  ...
  model/beaker_1/beaker_1.obj
  ...
data/clearpose_objects.csv
```

### Generate ground truth

Render normal maps and object masks from the annotated poses and CAD models. Vertices use the object-to-camera rotation and translation; normal directions use only the rotation. The script handles the OpenCV/OpenGL axis conversion and occlusion. Output normals are already in the evaluation convention, and rendered depth PNGs store millimeters.

```bash
pip install -r requirements-training.txt
python scripts/prepare_clearpose.py \
  --root data/clearpose_downsample_100 \
  --models data/clearpose_downsample_100/model \
  --objects-csv data/clearpose_objects.csv \
  --output data/clearpose_eval --check-only
python scripts/prepare_clearpose.py \
  --root data/clearpose_downsample_100 \
  --models data/clearpose_downsample_100/model \
  --objects-csv data/clearpose_objects.csv \
  --output data/clearpose_eval --backend egl
```

Use Linux with EGL support; `EGL_DEVICE_ID` selects the rendering GPU. The script reads the fixed test list and writes a new output directory containing RGB, `rendered_normal`, `rendered_depth` and `rendered_mask` PNGs. Existing source data is unchanged. `preparation.json` lists the generated files and their hashes.

The evaluation uses these object-rendered annotations, rather than `normal_true.png` from the depth benchmark.

### Evaluate

Download the [model weights](../README.md#download-model-weights), then run:

```bash
python evaluate.py --dataset clearpose --data data/clearpose_eval --check-data
python evaluate.py --dataset clearpose --data data/clearpose_eval \
  --model-path weights/transnormal --dino-path weights/dinov3_vith16plus \
  --output outputs/clearpose_eval --seed 42
```

Results are saved under `outputs/clearpose_eval/clearpose/`: raw normal predictions, `metrics.json`, `per_image.json` and `initial_rng.pt`. The summary CSV is in the parent output directory.

To score saved predictions:

```bash
python evaluate.py --dataset clearpose --data data/clearpose_eval \
  --prediction-dir outputs/clearpose_eval/clearpose/predictions \
  --output outputs/clearpose_rescored
```

Metrics use pixels where `rendered_mask > 0`, without erosion or a depth cutoff. Normal PNGs decode as `RGB / 127.5 - 1`; no additional axis flip is needed. See [evaluation](evaluation.md) for the shared metric definitions and RNG replay command.
