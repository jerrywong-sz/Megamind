import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import train_one_epoch, validate_one_epoch


def test_training_loop_runs():
    device = torch.device("cpu")

    # Tiny toy model — we're testing the training loop,
    # not EfficientNet itself.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 1),
    ).to(device)

    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

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
