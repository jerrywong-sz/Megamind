"""Shared data-loading and preprocessing utilities."""

import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.augmentations import random_transform


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform() -> transforms.Compose:
    """Build the deterministic evaluation/inference transform."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_train_transform() -> transforms.Compose:
    """Build the conventional Experiment A training transform."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_robust_train_transform() -> transforms.Compose:
    """Build the robustness-augmented Experiment B training transform."""
    return transforms.Compose([
        transforms.Lambda(random_transform),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


class ManifestDataset(Dataset):
    """Dataset loader backed by the shared CSV manifest."""

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str,
        transform=None,
    ):
        self.data_root = data_root
        self.df = pd.read_csv(manifest_path)
        self.df = (
            self.df[self.df["split"] == split]
            .reset_index(drop=True)
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        full_path = os.path.join(
            self.data_root,
            row["image_path"],
        )

        img = Image.open(full_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(
            [row["label"]],
            dtype=torch.float32,
        )

        metadata = {
            "image_path": row["image_path"],
            "dataset": row["dataset"],
            "generator": row["generator"],
            "format": row["format"],
        }

        return img, label, metadata


class ConsistencyManifestDataset(Dataset):
    """
    Experiment C dataset.

    Each source image produces two views:
    1. a clean view
    2. a randomly damaged view

    Both views keep the same label.
    """

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str,
    ):
        self.data_root = data_root

        self.df = pd.read_csv(manifest_path)
        self.df = (
            self.df[self.df["split"] == split]
            .reset_index(drop=True)
        )

        # After the clean/damaged pair has been created,
        # both views receive exactly the same deterministic preprocessing.
        self.base_transform = get_eval_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        full_path = os.path.join(
            self.data_root,
            row["image_path"],
        )

        image = Image.open(full_path).convert("RGB")

        # Apply the same random horizontal flip to BOTH views.
        # This prevents the consistency loss from comparing two
        # unnecessarily different orientations.
        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        clean_image = image.copy()

        damaged_image = random_transform(
            image.copy()
        )

        clean_tensor = self.base_transform(
            clean_image
        )

        damaged_tensor = self.base_transform(
            damaged_image
        )

        label = torch.tensor(
            [row["label"]],
            dtype=torch.float32,
        )

        metadata = {
            "image_path": row["image_path"],
            "dataset": row["dataset"],
            "generator": row["generator"],
            "format": row["format"],
        }

        return (
            clean_tensor,
            damaged_tensor,
            label,
            metadata,
        )


def get_dataloaders(
    data_root: str,
    manifest_path: str,
    batch_size: int = 32,
    train_mode: str = "clean",
) -> tuple[DataLoader, DataLoader]:
    """Build reproducible train/validation DataLoaders."""

    if train_mode == "clean":
        train_dataset = ManifestDataset(
            data_root=data_root,
            manifest_path=manifest_path,
            split="train",
            transform=get_train_transform(),
        )

    elif train_mode == "robust":
        train_dataset = ManifestDataset(
            data_root=data_root,
            manifest_path=manifest_path,
            split="train",
            transform=get_robust_train_transform(),
        )

    elif train_mode == "consistency":
        train_dataset = ConsistencyManifestDataset(
            data_root=data_root,
            manifest_path=manifest_path,
            split="train",
        )

    else:
        raise ValueError(
            f"Unknown train_mode '{train_mode}'. "
            "Expected 'clean', 'robust', or 'consistency'."
        )

    # Validation always stays clean and identical across A/B/C.
    val_dataset = ManifestDataset(
        data_root=data_root,
        manifest_path=manifest_path,
        split="val",
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader
