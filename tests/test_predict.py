"""Smoke test for the inference pipeline: does it run end-to-end without crashing
on a handful of awkward inputs, using the random fallback (no checkpoint needed)?
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn

import predict

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_test_images(root: Path) -> int:
    """Create a small set of deliberately awkward test images under root.

    Returns the total number of image files created (what predict.py should
    report as "found" and include in its output).
    """
    (root / "sub").mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (64, 64), "red").save(root / "plain.jpg")
    Image.new("L", (64, 64), 128).save(root / "grayscale.png")
    Image.new("RGBA", (64, 64), (0, 255, 0, 128)).save(root / "sub" / "nested_rgba.png")
    (root / "corrupted.png").write_bytes(b"this is not a real image file")

    return 4


def run_predict(input_dir: Path, output_path: Path, *extra_args: str):
    """Run predict.py as a subprocess, exactly as a judge would invoke it."""
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "predict.py"),
            "--input_dir", str(input_dir),
            "--output", str(output_path),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_predict_runs_end_to_end_with_random_fallback(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    num_images = make_test_images(input_dir)

    output_path = tmp_path / "preds.json"

    result = run_predict(input_dir, output_path)

    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"
    assert output_path.exists(), "predict.py did not write an output file"

    with open(output_path) as f:
        predictions = json.load(f)

    assert isinstance(predictions, list)
    assert len(predictions) == num_images

    for row in predictions:
        # The submission format is exactly these two keys -- no more, no less.
        # An extra key could fail a strict schema check during judging.
        assert set(row.keys()) == {"image_path", "pred"}
        assert isinstance(row["pred"], float)
        assert 0.0 <= row["pred"] <= 1.0


def test_predict_include_label_flag_adds_predicted_label(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    num_images = make_test_images(input_dir)

    output_path = tmp_path / "preds.json"

    result = run_predict(input_dir, output_path, "--include-label")

    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"

    with open(output_path) as f:
        predictions = json.load(f)

    assert len(predictions) == num_images

    for row in predictions:
        assert set(row.keys()) == {"image_path", "pred", "predicted_label"}
        assert row["predicted_label"] in (0, 1)


def test_predict_include_label_respects_threshold(tmp_path):
    """A threshold of 0 labels everything 1; a threshold above 1 labels everything 0."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    make_test_images(input_dir)

    all_positive = tmp_path / "all_positive.json"
    result = run_predict(input_dir, all_positive, "--include-label", "--threshold", "0.0")
    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"
    with open(all_positive) as f:
        assert all(row["predicted_label"] == 1 for row in json.load(f))

    all_negative = tmp_path / "all_negative.json"
    result = run_predict(input_dir, all_negative, "--include-label", "--threshold", "1.1")
    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"
    with open(all_negative) as f:
        assert all(row["predicted_label"] == 0 for row in json.load(f))


def test_predict_handles_empty_input_dir(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_path = tmp_path / "preds.json"

    result = run_predict(input_dir, output_path)

    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"

    with open(output_path) as f:
        predictions = json.load(f)

    assert predictions == []


def test_load_model_uses_checkpoint_architecture_metadata(
    tmp_path,
    monkeypatch,
):
    def make_model():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 1),
        )

    original_model = make_model()

    checkpoint_path = tmp_path / "model.pt"

    torch.save(
        {
            "architecture": "convnext_tiny",
            "model_state": original_model.state_dict(),
        },
        checkpoint_path,
    )

    requested = {}

    def fake_build_model(
        pretrained,
        architecture="efficientnet_b0",
    ):
        requested["architecture"] = architecture
        return make_model()

    monkeypatch.setattr(
        predict,
        "build_model",
        fake_build_model,
    )

    loaded_model = predict.load_model(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert requested["architecture"] == "convnext_tiny"
    assert loaded_model is not None
