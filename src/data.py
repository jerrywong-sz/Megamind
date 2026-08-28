"""Shared data-loading and preprocessing utilities."""


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform() -> transforms.Compose:
    """Build the image transform used at evaluation/inference time.

    This is the single source of truth for eval-time preprocessing: training
    and inference (predict.py) must both call this function so the two
    pipelines can never drift apart. If you need to change resize size, crop
    size, or normalization stats, change it here only -- do not duplicate
    these values elsewhere.

    Returns:
        A torchvision.transforms.Compose that resizes to 256x256, center
        crops to 224x224, converts to a tensor, and normalizes with
        ImageNet mean/std.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_train_transform() -> transforms.Compose:
    """Build the image transform used for Experiment A training."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
class ManifestDataset(Dataset):
    """The official dataset loader that relies on the CSV manifest."""
    def __init__(self, data_root: str, manifest_path: str, split: str, transform=None):
        self.data_root = data_root  # <--- UPDATE
        self.df = pd.read_csv(manifest_path)
        # Filter down to only the requested split (train, val, or test)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        import os
        # Glue the root directory to the relative path from the CSV
        full_path = os.path.join(self.data_root, row['image_path']) # <--- UPDATE

        # Always convert to RGB to prevent format bias crashes later
        img = Image.open(full_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor([row['label']], dtype=torch.float32)
        
        # Pass metadata back so Person 4 (Evaluation Lead) can analyze errors later
        metadata = {
            "image_path": row['image_path'],
            "dataset": row['dataset'],
            "generator": row['generator'],
            "format": row['format']
        }
        return img, label, metadata
def get_dataloaders(
    data_root: str,       # <--- UPDATE
    manifest_path: str, 
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    """
    Builds reproducible train/val dataloaders for the rest of the team.
    """
    train_dataset = ManifestDataset(data_root=data_root, manifest_path=manifest_path, split='train', transform=get_train_transform())
    val_dataset = ManifestDataset(data_root=data_root, manifest_path=manifest_path, split='val', transform=get_eval_transform())
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    return train_loader, val_loader