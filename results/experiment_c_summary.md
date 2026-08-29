# Experiment C — Consistency Training

## Goal

Measure whether explicitly encouraging prediction consistency between clean and
damaged versions of the same image improves robustness beyond the robustness
augmentation used in Experiment B.

Experiment C keeps the same EfficientNet-B0 architecture and core training
settings as Experiments A and B, while adding clean/damaged image pairs and a
consistency loss.

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
- Training mode: consistency
- Consistency lambda: 0.5
- Training data: CIFAKE train split
- Validation data: clean CIFAKE validation split

For each training image, Experiment C creates:

1. a clean view
2. a randomly damaged view

Both views are classified using the same model. The training objective combines
classification loss on both views with a consistency penalty that encourages
the model to produce similar probabilities for the clean and damaged versions.

## Results

| Epoch | Train Loss | Cls Loss | Con Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|-----------:|---------:|---------:|---------------:|---------:|-------------:|
| 1 | 0.4255 | 0.4101 | 0.0309 | 91.56% | 0.0854 | 96.79% |
| 2 | 0.2572 | 0.2450 | 0.0245 | 95.17% | 0.0806 | 96.89% |
| 3 | 0.2008 | 0.1895 | 0.0227 | 96.37% | 0.0538 | 98.06% |
| 4 | 0.1711 | 0.1604 | 0.0212 | 96.97% | 0.0548 | 98.04% |
| 5 | 0.1463 | 0.1361 | 0.0204 | 97.44% | 0.0503 | 98.21% |

Best checkpoint: Epoch 5, selected by lowest clean validation loss.

Across training, both the classification loss and consistency loss decreased,
while training accuracy increased. This indicates that the model improved its
Real/AI classification while also reducing disagreement between clean and
damaged views.

## Clean validation comparison

| Experiment | Training | Clean Val Accuracy | Val Loss |
|---|---|---:|---:|
| A | Clean baseline | 98.31% | 0.0528 |
| B | Robustness augmentation | 98.02% | 0.0563 |
| C | Robustness augmentation + consistency | 98.21% | 0.0503 |

Experiment C recovers most of Experiment A's clean validation accuracy while
improving clean validation loss relative to both Experiments A and B.

However, clean validation performance alone is not sufficient to conclude that
Experiment C is better than Experiment B.

## Evaluation status

Robustness evaluation of Experiment C is pending.

Experiment C should be compared with Experiment B using the same individual
robustness conditions, followed by compound-corruption stress tests.

No claim is made yet that Experiment C is the final or best-performing model.

## Checkpoint

The best checkpoint is stored separately as:

`experiment_c_consistency_best.pt`

Model checkpoints are intentionally not committed to Git.
