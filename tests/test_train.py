import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    fit,
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
    )

    assert len(history) == 2
    assert checkpoint_path.exists()

    # Make sure the saved checkpoint can actually be loaded.
    loaded_model = make_tiny_model()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    loaded_model.load_state_dict(state_dict)


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
    )

    assert loss >= 0
    assert 0 <= accuracy <= 1
