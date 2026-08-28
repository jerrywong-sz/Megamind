# Experiment A — Clean Baseline

Model: EfficientNet-B0
Training set: CIFAKE
Training images: 68,712
Validation images: 14,724
Epochs: 5
Batch size: 32
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
AMP: enabled
Checkpoint selection: lowest validation loss

## Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
| 1 | 0.1534 | 94.05% | 0.0721 | 97.41% |
| 2 | 0.0773 | 97.14% | 0.0624 | 97.64% |
| 3 | 0.0519 | 98.14% | 0.0552 | 98.06% |
| 4 | 0.0402 | 98.53% | 0.0557 | 98.14% |
| 5 | 0.0307 | 98.91% | 0.0528 | 98.31% |

Best checkpoint: Epoch 5
Best validation loss: 0.0528
Validation accuracy at best checkpoint: 98.31%
