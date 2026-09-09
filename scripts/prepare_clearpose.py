"""Render the fixed ClearPose normal-evaluation subset from official meshes and poses."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.io import loadmat

SPLIT = Path(__file__).resolve().parents[1] / "configs/splits/clearpose_test.txt"


def read_split(path):
    keys = Path(path).read_text().splitlines()
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("The split must contain unique sample paths.")
    for key in keys:
        parts = Path(key).parts
        if len(parts) != 3 or not parts[0].startswith("set") or not parts[1].startswith("scene") or not parts[2].endswith(".png") or not parts[2][:-4].isdigit():
            raise ValueError(f"Invalid sample path: {key}")
    return keys


class Renderer:
    """Preserve the mesh/vertex-color rendering convention used for evaluation GT."""

    def __init__(self, models, mapping, width, height, intrinsic):
        import pyrender
        self.pyrender = pyrender
        self.models, self.mapping, self.cache = Path(models), mapping, {}
        self.axis = np.diag([1., -1., -1., 1.])
        self.scene = pyrender.Scene()
        camera = pyrender.IntrinsicsCamera(fx=intrinsic[0, 0], fy=intrinsic[1, 1],
                                           cx=intrinsic[0, 2], cy=intrinsic[1, 2], znear=.01, zfar=100.)
        self.scene.add_node(pyrender.Node(camera=camera, matrix=np.eye(4)))
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def mesh(self, object_id):
        import trimesh
        if object_id not in self.cache:
            name = self.mapping[object_id]
            path = self.models / name / (name + ".obj")
            if not path.is_file():
                raise FileNotFoundError(path)
            self.cache[object_id] = trimesh.load(str(path), force="mesh")
        return self.cache[object_id].copy()

    def render(self, ids, poses):
        from trimesh.visual import ColorVisuals
        for normal_pass in (False, True):
            for node in list(self.scene.mesh_nodes):
                self.scene.remove_node(node)
            for i, object_id in enumerate(np.asarray(ids).reshape(-1)):
                mesh = self.mesh(int(object_id))
                pose = poses[:, :, i]
                transform = np.eye(4)
                transform[:3] = pose
                if normal_pass:
                    normals = mesh.vertex_normals @ pose[:, :3].T @ self.axis[:3, :3].T
                    colors = (normals + 1.) / 2.
                    colors[:, 0] = 1. - colors[:, 0]
                    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=np.column_stack((colors, np.ones(len(colors)))))
                    py_mesh = self.pyrender.Mesh.from_trimesh(mesh, smooth=False)
                else:
                    py_mesh = self.pyrender.Mesh.from_trimesh(mesh)
                self.scene.add_node(self.pyrender.Node(mesh=py_mesh, matrix=self.axis @ transform))
            if normal_pass:
                normal, normal_depth = self.renderer.render(self.scene, flags=self.pyrender.RenderFlags.FLAT)
                normal = normal.copy()
                normal[normal_depth == 0] = 0
            else:
                _, depth = self.renderer.render(self.scene)
        return (depth * 1000).astype(np.uint16), normal, (depth > 0).astype(np.uint8) * 255

    def close(self):
        self.renderer.delete()


def prepare(args):
    keys = read_split(args.split)
    with Path(args.objects_csv).open(newline="") as handle:
        mapping = {int(row[0]): row[1] for row in csv.reader(handle) if len(row) >= 2}
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing annotations will not be overwritten.")
    scenes = {}
    required = set()
    # Validate every selected frame before rendering or creating output files.
    for key in keys:
        rel = Path(key)
        scene, frame = str(rel.parent), rel.stem
        if scene not in scenes:
            scenes[scene] = loadmat(root / scene / "metadata.mat")
        row = scenes[scene][frame][0, 0]
        if not (root / scene / f"{frame}-color.png").is_file():
            raise FileNotFoundError(root / scene / f"{frame}-color.png")
        intrinsic = row["intrinsic_matrix"]
        ids, poses = row["cls_indexes"], row["poses"]
        if intrinsic.shape != (3, 3) or poses.shape != (3, 4, ids.size) or ids.size == 0:
            raise ValueError(f"Invalid metadata: {key}")
        required.update(int(x) for x in ids.reshape(-1))
    for object_id in required:
        name = mapping[object_id]
        if not (Path(args.models) / name / f"{name}.obj").is_file():
            raise FileNotFoundError(f"Missing mesh for object {object_id}: {name}")
    if args.check_only:
        print(json.dumps({"samples": len(keys), "scenes": len(scenes), "objects": len(required), "status": "inputs-valid"}))
        return
    os.environ["PYOPENGL_PLATFORM"] = args.backend
    output.mkdir(parents=True)
    hashes = {}
    for scene in scenes:
        selected = [Path(k).stem for k in keys if str(Path(k).parent) == scene]
        first = scenes[scene][selected[0]][0, 0]["intrinsic_matrix"]
        # The original renderer initialized one camera per scene; reject changes instead of silently ignoring them.
        for frame in selected:
            if not np.array_equal(scenes[scene][frame][0, 0]["intrinsic_matrix"], first):
                raise ValueError(f"Camera intrinsics vary within scene {scene}")
        with Image.open(root / scene / f"{selected[0]}-color.png") as image:
            width, height = image.size
        renderer = Renderer(args.models, mapping, width, height, first)
        destination = output / scene
        destination.mkdir(parents=True)
        shutil.copyfile(root / scene / "metadata.mat", destination / "metadata.mat")
        try:
            for frame in selected:
                row = scenes[scene][frame][0, 0]
                source = root / scene / f"{frame}-color.png"
                with Image.open(source) as image:
                    if image.size != (width, height):
                        raise ValueError(f"Image dimensions vary within scene {scene}")
                depth, normal, mask = renderer.render(row["cls_indexes"], row["poses"])
                shutil.copyfile(source, destination / source.name)
                for suffix, array in [("rendered_depth", depth), ("rendered_normal", normal), ("rendered_mask", mask)]:
                    Image.fromarray(array).save(destination / f"{frame}-{suffix}.png")
                print(f"Prepared {scene}/{frame}", flush=True)
        finally:
            renderer.close()
    for path in output.rglob("*"):
        if path.is_file():
            hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"samples": len(keys), "split_sha256": hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
               "objects_csv_sha256": hashlib.sha256(Path(args.objects_csv).read_bytes()).hexdigest(),
               "versions": {name: importlib.metadata.version(name) for name in ("pyrender", "trimesh", "PyOpenGL", "numpy", "Pillow")},
               "backend": args.backend, "files_sha256": hashes}
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Official extracted data containing set*/scene*/")
    parser.add_argument("--models", required=True, help="Official model/ directory")
    parser.add_argument("--objects-csv", required=True, help="data/objects.csv from the official ClearPose repository")
    parser.add_argument("--output", required=True, help="New directory for the selected RGB and rendered annotations")
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--backend", choices=["osmesa", "egl"], default="egl", help="EGL matches the original backend; OSMesa uses CPU rendering with additional dependencies")
    parser.add_argument("--check-only", action="store_true")
    prepare(parser.parse_args())
