"""Single and directory inference CLI behavior without model downloads."""

import numpy as np
from PIL import Image
import pytest

import inference


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), (12, 34, 56)).save(path)


def test_recursive_input_preserves_layout_and_excludes_output(tmp_path):
    root = tmp_path / "images"
    write_image(root / "one.JpG")
    write_image(root / "nested/two.png")
    write_image(root / "results/old.png")
    args = inference.parse_args(["--input", str(root), "--output", str(root / "results"), "--recursive"])
    jobs = inference.output_jobs(args)
    assert {(p.relative_to(root).as_posix(), q.relative_to(root / "results").as_posix()) for p, q in jobs} == {
        ("one.JpG", "one_normal.png"), ("nested/two.png", "nested/two_normal.png")}
    args.output = tmp_path
    assert len(inference.output_jobs(args)) == 3


def test_output_collisions_and_input_overwrite_are_rejected(tmp_path):
    root = tmp_path / "images"
    write_image(root / "same.png")
    write_image(root / "same.jpg")
    with pytest.raises(ValueError, match="Multiple inputs"):
        inference.output_jobs(inference.parse_args(["--input", str(root), "--output", str(tmp_path / "out")]))
    with pytest.raises(ValueError, match="replace the input"):
        inference.output_jobs(inference.parse_args(["--input", str(root / "same.png"), "--output", str(root / "same.png")]))


def test_skip_existing_avoids_loading_model(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    write_image(source)
    output.write_bytes(b"keep existing")
    monkeypatch.setattr(inference, "load_pipeline", lambda args: pytest.fail("Model should not load"))
    assert inference.main(["--input", str(source), "--output", str(output), "--skip_existing"]) == 0
    assert output.read_bytes() == b"keep existing"


def test_single_and_directory_share_pipeline_and_outputs(tmp_path, monkeypatch):
    root = tmp_path / "images"
    write_image(root / "one.png")
    write_image(root / "two.jpg")
    loads = []

    def load(args):
        loads.append(args)
        def pipe(image, processing_res, output_type):
            if output_type == "np":
                return np.full((image.height, image.width, 3), .25, dtype=np.float32)
            return Image.new("RGB", image.size, (10, 20, 30))
        return pipe

    monkeypatch.setattr(inference, "load_pipeline", load)
    assert inference.main(["--input", str(root), "--output", str(tmp_path / "out"), "--save_comparison"]) == 0
    assert len(loads) == 1
    for name in ("one_normal.png", "two_normal.png"):
        with Image.open(tmp_path / "out" / name) as image:
            assert image.size == (24, 8)
    destination = tmp_path / "nested/raw.NPZ"
    assert inference.main(["--image", str(root / "one.png"), "--output", str(destination)]) == 0
    with np.load(destination) as archive:
        assert archive["normal"].shape == (8, 12, 3)
        assert np.all(archive["normal"] == .25)
    with pytest.raises(ValueError, match="requires PNG"):
        inference.output_jobs(inference.parse_args(["--input", str(root), "--output_format", "npz", "--save_comparison"]))


def test_missing_projector_is_not_silently_ignored(tmp_path):
    model = tmp_path / "model"
    dino = tmp_path / "dino"
    model.mkdir()
    dino.mkdir()
    args = inference.parse_args(["--input", "unused.png", "--model_path", str(model), "--dino_path", str(dino)])
    with pytest.raises(FileNotFoundError, match="Projector"):
        inference.load_pipeline(args)


def test_failed_image_makes_batch_exit_nonzero(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    (root / "broken.png").write_bytes(b"not an image")
    monkeypatch.setattr(inference, "load_pipeline", lambda args: object())
    assert inference.main(["--input", str(root), "--output", str(tmp_path / "out")]) == 1
