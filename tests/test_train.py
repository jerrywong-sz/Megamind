import random
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    fit,
    main,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)


def make_tiny_loader():
    images = torch.randn(8, 3, 8, 8)

    labels = torch.tensor([
        0, 1, 0, 1,
        0, 1, 0, 1,
    ])

    return DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )


def make_tiny_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 1),
    )


def test_set_seed_is_reproducible():
    set_seed(42)

    python_1 = random.random()
    numpy_1 = np.random.rand()
    torch_1 = torch.rand(1)

    set_seed(42)

    python_2 = random.random()
    numpy_2 = np.random.rand()
    torch_2 = torch.rand(1)

    assert python_1 == python_2
    assert numpy_1 == numpy_2
    assert torch.equal(torch_1, torch_2)


def test_training_and_validation_loops_run():
    device = torch.device("cpu")

    model = make_tiny_model().to(device)
    loader = make_tiny_loader()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    train_loss, train_accuracy = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        config={"train_mode": "clean"},
    )

    val_loss, val_accuracy = validate_one_epoch(
        model,
        loader,
        criterion,
        device,
    )

    assert train_loss >= 0
    assert val_loss >= 0

    assert 0 <= train_accuracy <= 1
    assert 0 <= val_accuracy <= 1


def test_fit_saves_best_checkpoint(tmp_path):
    device = torch.device("cpu")

    model = make_tiny_model().to(device)
    loader = make_tiny_loader()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_path = tmp_path / "baseline.pt"

    history = fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=2,
        checkpoint_path=checkpoint_path,
        config={"train_mode": "clean"},
    )

    assert len(history) == 2
    assert checkpoint_path.exists()

    # Make sure the saved checkpoint can actually be loaded.
    loaded_model = make_tiny_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint
    assert "epoch" in checkpoint
    assert "architecture" in checkpoint
    assert checkpoint["architecture"] == "efficientnet_b0"

    loaded_model.load_state_dict(
        checkpoint["model_state"]
    )


def test_training_loop_accepts_metadata():
    device = torch.device("cpu")

    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])

    metadata = [
        {"image_path": f"image_{i}.jpg"}
        for i in range(4)
    ]

    class MetadataDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            return (
                images[index],
                labels[index],
                metadata[index],
            )

    loader = DataLoader(
        MetadataDataset(),
        batch_size=2,
        shuffle=False,
    )

    model = make_tiny_model().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    loss, accuracy = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        config={"train_mode": "clean"},
    )

    assert loss >= 0
    assert 0 <= accuracy <= 1


def test_main_connects_manifest_dataloaders_to_fit(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join([
            "experiment: clean",
            "seed: 42",
            "pretrained: false",
            "batch_size: 8",
            "epochs: 3",
            "learning_rate: 0.0001",
            "weight_decay: 0.0001",
            "checkpoint_dir: checkpoints",
            "checkpoint_name: baseline.pt",
        ])
    )

    manifest_path = tmp_path / "manifest.csv"
    train_loader = object()
    val_loader = object()
    model = make_tiny_model()

    build_model_mock = Mock(return_value=model)
    get_dataloaders_mock = Mock(
        return_value=(train_loader, val_loader)
    )
    fit_mock = Mock()

    monkeypatch.setattr("train.build_model", build_model_mock)
    monkeypatch.setattr(
        "train.get_dataloaders",
        get_dataloaders_mock,
    )
    monkeypatch.setattr("train.fit", fit_mock)
    data_root = tmp_path / "images"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
        ],
    )

    main()

    get_dataloaders_mock.assert_called_once_with(
        data_root=str(data_root),
        manifest_path=str(manifest_path),
        batch_size=8,
        train_mode="clean",
    )

    fit_call = fit_mock.call_args.kwargs

    assert fit_call["model"] is model
    assert fit_call["train_loader"] is train_loader
    assert fit_call["val_loader"] is val_loader
    assert fit_call["epochs"] == 3
    assert fit_call["checkpoint_path"] == Path(
        "checkpoints/baseline.pt"
    )
