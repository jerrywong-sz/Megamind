import argparse

import torch
import torch.nn as nn
import yaml

from src.models import build_model


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

    for images, labels in train_loader:
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
        for images, labels in val_loader:
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
