import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train import train_one_epoch_consistency


class TinyConsistencyDataset(Dataset):

    def __init__(self):
        torch.manual_seed(42)

        self.clean = torch.randn(
            8,
            3,
            8,
            8,
        )

        self.damaged = (
            self.clean
            + 0.05 * torch.randn_like(self.clean)
        )

        self.labels = torch.tensor(
            [0, 1, 0, 1, 0, 1, 0, 1],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.clean[index],
            self.damaged[index],
            self.labels[index],
            {"image_path": f"image_{index}.jpg"},
        )


def test_consistency_training_loop_runs():

    loader = DataLoader(
        TinyConsistencyDataset(),
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    (
        loss,
        accuracy,
        classification_loss,
        consistency_loss,
    ) = train_one_epoch_consistency(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        consistency_lambda=0.5,
    )

    assert loss >= 0
    assert classification_loss >= 0
    assert consistency_loss >= 0
    assert 0 <= accuracy <= 1
