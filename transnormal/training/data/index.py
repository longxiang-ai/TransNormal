"""Create portable JSONL indices without copying training images."""

import argparse
import json
import os
import tarfile
from pathlib import Path

KINDS = ("cleargrasp", "transnormal", "hypersim", "vkitti")
TN_FIELDS = ("with_rgb.png", "change_rgb.png", "with_normal.png", "with_mask_all.png")


def raw_records(kind, root, split):
    if kind == "cleargrasp":
        for image in sorted(root.glob("*/rgb-imgs/*")) + sorted(root.glob("*/transparent-rgb-imgs/*")):
            if image.suffix.lower() not in (".jpg", ".png"):
                continue
            base = image.name.split("-")[0]
            scene = image.parent.parent
            yield {"image": str(image.relative_to(root)),
                   "normal": str((scene / "camera-normals" / (base + "-cameraNormals.exr")).relative_to(root)),
                   "mask": str((scene / "variant-masks" / (base + "-variantMasks.exr")).relative_to(root))}
    elif kind == "hypersim":
        for key in split:
            rel = str(Path("train") / key)
            yield {"image": rel,
                   "normal": rel.replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "normal_cam.hdf5"),
                   "depth": rel.replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "depth_meters.hdf5")}
    elif kind == "vkitti":
        conditions = ("15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone", "fog", "morning", "overcast", "rain", "sunset")
        for scene in ("02", "06", "18", "20"):
            for condition in conditions:
                for image in sorted((root / ("Scene" + scene) / condition / "frames" / "rgb").glob("Camera_*/*.jpg")):
                    rel = str(image.relative_to(root))
                    yield {"image": rel, "normal": rel.replace("rgb", "normal").replace(".jpg", ".png"),
                           "depth": rel.replace("rgb", "depth").replace(".jpg", ".png")}
    else:
        for key in split:
            scene, view = key.split("/")
            suffix = f"{int(view.removeprefix('view_')):04d}"
            base = Path(scene) / "views" / view
            yield {"key": key, "image": str(base / "with_transparent/rgb_.png"),
                   "changed": str(base / "change_material/rgb_.png"),
                   "normal": str(base / "with_transparent" / f"normal_{suffix}.png"),
                   "mask": str(base / "with_transparent/masks" / f"all_{suffix}.png")}


def shard_records(root, split):
    """Index uncompressed public WebDataset tar files by byte offset."""
    wanted = set(split)
    found = {}
    for shard in sorted(root.rglob("*.tar")):
        samples = {}
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name
                if "." not in name:
                    continue
                key, field = name.split(".", 1)
                entry = samples.setdefault(key, {"files": {}})
                if field == "meta.json":
                    metadata = json.load(archive.extractfile(member))
                    entry["key"] = f"{metadata['scene_name']}/{metadata['view_id']}"
                elif field in TN_FIELDS:
                    entry["files"][field] = [member.offset_data, member.size]
        for entry in samples.values():
            key = entry.get("key")
            if key not in wanted:
                continue
            if set(entry["files"]) != set(TN_FIELDS):
                raise ValueError(f"Incomplete training pair in {shard.name}: {key}")
            if key in found:
                raise ValueError(f"Duplicate training pair: {key}")
            found[key] = {**entry, "shard": str(shard.relative_to(root))}
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} training pairs absent from shards; first: {sorted(missing)[:3]}")
    return [found[key] for key in split]


def create_index(kind, root, output, split_file=None, shards=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(f"Index already exists: {output}; choose a new output path.")
    split = []
    if kind in ("transnormal", "hypersim"):
        if split_file is None:
            split_file = Path(__file__).resolve().parents[3] / "configs" / "splits" / f"{kind}_train.txt"
        split = Path(split_file).read_text().splitlines()
        if not split or len(split) != len(set(split)):
            raise ValueError("The training split must contain unique non-empty sample keys.")
        if any(Path(key).is_absolute() or ".." in Path(key).parts for key in split):
            raise ValueError("Training split entries must be relative paths without parent traversal.")
    if shards and kind != "transnormal":
        raise ValueError("Shard indexing is supported for the public TransNormal dataset.")
    records = shard_records(root, split) if shards else list(raw_records(kind, root, split))
    if not records:
        raise ValueError(f"No {kind} samples found under the supplied root.")
    for record in records:
        for field in ("image", "normal", "mask", "depth", "changed"):
            if field in record and not (root / record[field]).is_file():
                raise FileNotFoundError(root / record[field])
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {"format": "transnormal-training-index-v1", "kind": kind,
              "root": os.path.relpath(root, output.parent), "samples": len(records)}
    with output.open("x") as handle:
        handle.write(json.dumps(header) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", help="Optional replacement for the supplied TransNormal or Hypersim training split")
    parser.add_argument("--shards", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_index(args.kind, args.root, args.output, args.split, args.shards), indent=2))


if __name__ == "__main__":
    main()
