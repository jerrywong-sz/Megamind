import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models import build_model



def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, *_ in train_loader:
        images = images.to(device)

        labels = (
            labels.float()
            .view(-1, 1)
            .to(device)
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += images.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels, *_ in val_loader:
            images = images.to(device)

            labels = (
                labels.float()
                .view(-1, 1)
                .to(device)
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += images.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy



def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
):
    """Train and validate the model across multiple epochs."""

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                checkpoint_path,
            )

            print(
                f"✅ Best checkpoint saved: "
                f"{checkpoint_path}"
            )

    return history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    set_seed(config["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        pretrained=config["pretrained"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print("Experiment:", config["experiment"])
    print("Device:", device)
    print("Model:", model.__class__.__name__)
    print("Loss:", criterion.__class__.__name__)
    print("Optimizer:", optimizer.__class__.__name__)

    print(
        "✅ Training components ready!"
    )

    # Dataset loaders will be connected next.


if __name__ == "__main__":
    main()
