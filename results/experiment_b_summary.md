# Experiment B — Robustness Augmentation

## Goal

Measure the effect of training EfficientNet-B0 with random real-world image
degradations while keeping the model architecture and core training settings
consistent with Experiment A.

Experiment B uses robustness augmentation during training but does not yet use
the consistency loss introduced in Experiment C.

## Training setup

- Model: EfficientNet-B0
- Seed: 42
- Image size: 224
- Batch size: 32
- Epochs: 5
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Mixed precision: enabled on CUDA
- Training mode: robust
- Training data: CIFAKE train split
- Validation data: clean CIFAKE validation split

Robustness augmentation may apply JPEG compression, Gaussian blur,
resize-and-upscale, Gaussian noise, colour jitter, and crop transformations.

## Results

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|-----------:|---------------:|---------:|-------------:|
| 1 | 0.2603 | 88.90% | 0.1029 | 96.12% |
| 2 | 0.1735 | 92.97% | 0.0803 | 96.98% |
| 3 | 0.1442 | 94.24% | 0.0684 | 97.47% |
| 4 | 0.1262 | 95.01% | 0.0627 | 97.81% |
| 5 | 0.1129 | 95.58% | 0.0563 | 98.02% |

Best checkpoint: Epoch 5, selected by lowest validation loss.

## Comparison with Experiment A

| Experiment | Training | Clean Val Accuracy | Val Loss |
|---|---|---:|---:|
| A | Clean baseline | 98.31% | 0.0528 |
| B | Robustness augmentation | 98.02% | 0.0563 |

Experiment B retains nearly all clean-image performance, with a decrease of
only 0.29 percentage points in clean validation accuracy relative to
Experiment A.

The key purpose of Experiment B is robustness rather than clean accuracy.
Its benefit will therefore be assessed by comparing Experiments A and B under
the same JPEG, blur, resize, noise, colour, and crop evaluation conditions.

## Checkpoint

The best checkpoint is stored separately as:

`experiment_b_robustness_best.pt`

Model checkpoints are intentionally not committed to Git.
