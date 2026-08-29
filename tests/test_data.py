"""Tests for deterministic manifest-backed evaluation loading."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data import get_evaluation_dataloader


def _write_image(data_root: Path, relative_path: str, colour: tuple[int, int, int]):
    image_path = data_root / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(image_path)


def _make_manifest(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "images"
    rows = [
        ("train/REAL/val-real.jpg", 0.0, "val", (20, 30, 40)),
        ("train/FAKE/val-ai.jpg", 1.0, "val", (220, 210, 200)),
        ("train/REAL/test-real.jpg", 0.0, "test", (60, 70, 80)),
        ("train/FAKE/train-ai.jpg", 1.0, "train", (180, 170, 160)),
    ]
    for relative_path, _, _, colour in rows:
        _write_image(data_root, relative_path, colour)

    manifest = pd.DataFrame([
        {
            "image_path": relative_path,
            "label": label,
            "dataset": "dummy",
            "generator": "dummy-ai" if label == 1 else "none",
            "width": 32,
            "height": 32,
            "format": "JPEG",
            "split": split,
        }
        for relative_path, label, split, _ in rows
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return data_root, manifest_path


def test_evaluation_dataloader_selects_split_in_manifest_order(tmp_path):
    data_root, manifest_path = _make_manifest(tmp_path)

    data_loader = get_evaluation_dataloader(
        data_root=str(data_root),
        manifest_path=str(manifest_path),
        split="val",
        batch_size=2,
        num_workers=0,
    )
    images, labels, metadata = next(iter(data_loader))

    assert images.shape == (2, 3, 224, 224)
    assert labels.reshape(-1).tolist() == [0.0, 1.0]
    assert metadata["image_path"] == [
        "train/REAL/val-real.jpg",
        "train/FAKE/val-ai.jpg",
    ]


def test_evaluation_dataloader_rejects_training_split(tmp_path):
    data_root, manifest_path = _make_manifest(tmp_path)

    with pytest.raises(ValueError, match="'val' or 'test'"):
        get_evaluation_dataloader(
            data_root=str(data_root),
            manifest_path=str(manifest_path),
            split="train",
            num_workers=0,
        )
