# techjam-aigc-detector

## Project overview

A binary image classifier that detects whether an image is AI-generated,
built for TikTok TechJam Track 5. The core challenge this project targets is
**robustness**: real-world images get compressed, blurred, resized, noised,
colour-jittered, and cropped before anyone ever sees them (e.g. after passing
through a social platform's upload pipeline), and a detector that only works
on pristine images isn't useful. The model is trained and evaluated to stay
accurate under these transformations, not just on clean inputs.

Architecture: EfficientNet-B0 backbone with a single-logit head (sigmoid ->
probability the image is AI-generated). See [src/models.py](src/models.py)
and [src/data.py](src/data.py) for the model definition and the shared
preprocessing pipeline used by both training and inference.

## Setup and installation

Requires Python 3.10+.

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run inference over a directory of images:

```bash
python predict.py --input_dir test_images --checkpoint <path_to_checkpoint.pt> --output results/preds.json
```

Options:

- `--input_dir` (required) — directory to scan recursively for images
  (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, case-insensitive)
- `--checkpoint` (optional) — path to a trained model checkpoint. If
  omitted, predictions fall back to random values with a warning printed to
  stderr — useful for testing the pipeline before a checkpoint exists, but
  **not a real detector output**
- `--output` (required) — path to write the output JSON to
- `--batch_size` (optional, default `32`) — images per inference batch

Example output (`results/preds.json`):

```json
[
  {"image_path": "photo1.jpg", "pred": 0.873},
  {"image_path": "subfolder/photo2.png", "pred": 0.041}
]
```

`pred` is the model's estimated probability that the image is AI-generated,
in `[0, 1]`. `image_path` is relative to `--input_dir`, using forward
slashes on every platform. Unreadable/corrupted images are skipped with a
warning but still appear in the output with `pred: 0.5`, so the row count
always matches the number of images found.

## Reproducing results

Follow these steps in order to go from raw images to a submission-ready
`results/preds.json`.

### 1. Environment

Set up and activate the venv as described in **Setup and installation**
above, then keep it active for every command below.

### 2. Obtain the datasets

- **CIFAKE** (Kaggle) — <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images>
- **SID_Set** (Hugging Face, **gated**) — <https://huggingface.co/datasets/saberzl/SID_Set>.
  You must request access on that page before you can download it; approval
  isn't instant, so request it early.
- **WildFake subset** (COCO val2017 real images + DALL·E Advanced generated
  images) — <https://modelscope.cn/datasets/hy2628982280/WildFake/summary>.
  This is a **held-out demonstration benchmark only**. Do **not** train on
  it — it exists to show generalization to a source/generator the model
  never saw during training.

### 3. Expected raw directory layout

`src/build_manifest.py` expects each dataset's raw images split into
`REAL/` and `FAKE/` subfolders (searched recursively), e.g.:

```
data/
  CIFAKE/
    train/
      REAL/
        ...
      FAKE/
        ...
  SID_Set/
    REAL/
      ...
    FAKE/
      ...
```

### 4. Build the manifest CSV

**`src/build_manifest.py` currently hardcodes Colab paths in its
`if __name__ == "__main__"` block instead of accepting CLI arguments** —
running `python src/build_manifest.py` as-is will try to read
`/content/images/...` and fail on a local machine. Call the function
directly with your own paths instead:

```bash
python -c "from src.build_manifest import build_and_split_manifest; build_and_split_manifest('data/CIFAKE/train', 'CIFAKE', 'SD1.4', 'data/cifake_manifest.csv')"
```

(identical command on Windows PowerShell and Mac/Linux)

This hashes and de-duplicates images, balances classes 50/50, and writes a
70/15/15 train/val/test split into the CSV. Run it once per dataset. TODO:
there's no built-in way yet to merge multiple datasets' manifests into one
combined CSV for `get_dataloaders()` — do that manually (e.g.
`pandas.concat`) until that's added.

### 5. Train the three experiments

TODO — `train.py` is still empty (open as PR #2, not yet merged) and
`configs/` has no committed config yet, so there is no real training
command to give here. Once it exists, it should be wired to:

- **Experiment A (clean baseline):** base preprocessing only —
  `get_train_transform()` in [src/data.py](src/data.py) (`Resize(256) →
  RandomCrop(224) → RandomHorizontalFlip → Normalize`; its docstring
  already labels it "Experiment A"), no robustness damage applied.
- **Experiment B (augmented):** same base preprocessing, plus
  `random_transform()` from
  [src/augmentations.py](src/augmentations.py), which applies 0–3 of
  {JPEG, blur, resize, noise, colour, crop} per image with randomized
  parameters.
- **Experiment C (consistency):** same as B, plus a consistency penalty
  between clean and damaged views of the same image. TODO: the loss isn't
  implemented yet — `src/losses.py` is currently empty.

Backbone (`build_model()`), seed, epochs, batch size, learning rate, and
weight decay must be held identical across A/B/C so the ablation isolates
the training method rather than confounding hyperparameters. TODO: fill in
the actual values and command once `train.py` exists, e.g.:

```
python train.py --config configs/experiment_a.yaml   # TODO: not real yet
```

### 6. Evaluate against the challenge transform grid

`evaluate.py` loads two named checkpoints, evaluates both on the same manifest
rows in the same order, and records checkpoint hashes so a run can be traced
back to the exact model files. The defaults call the two result sets
`experiment_a` and `experiment_b`; pass explicit model IDs when comparing a
different pair.

First reproduce the clean validation results:

```bash
python evaluate.py --mode clean --data-root <dataset_root> --manifest <manifest.csv> --checkpoint-a <experiment_a.pt> --checkpoint-b <experiment_b.pt> --output-dir results/clean_validation --split val --device auto
```

After the clean integration check passes, run the fixed robustness grid:

```bash
python evaluate.py --mode robustness --data-root <dataset_root> --manifest <manifest.csv> --checkpoint-a <experiment_a.pt> --checkpoint-b <experiment_b.pt> --output-dir results/robustness_validation --split val --device auto --seed 42
```

To measure the additional contribution of consistency training, compare B
against C with the same validation split, threshold, transformations, and
seed. The `checkpoint-a`/`checkpoint-b` names mean first/second comparison
slots here; the explicit IDs ensure every output row is labelled correctly:

```bash
python evaluate.py --mode robustness --data-root <dataset_root> --manifest <manifest.csv> --checkpoint-a <experiment_b_robustness_best.pt> --model-a-id experiment_b --checkpoint-b <experiment_c_consistency_best.pt> --model-b-id experiment_c --output-dir results/robustness_b_vs_c_validation --split val --device auto --seed 42
```

Keep the earlier A-versus-B result directory. Do not overwrite it. The saved
A/B metrics and this B/C run can later be joined by transform and severity to
produce the final A/B/C ablation table without rerunning A.

The robustness runner uses:

- `exact_transform(img, name, param)` in
  [src/augmentations.py](src/augmentations.py) to apply each named transform.
  Names match the six challenge dimensions: `jpeg`, `blur`, `resize`, `noise`,
  `colour`, `crop`. Gaussian noise is seeded per image so A and B receive the
  same noisy pixels and later runs are reproducible.
- `compute_binary_metrics(labels, probabilities, threshold=0.5)` in
  [src/metrics.py](src/metrics.py) — returns accuracy, balanced accuracy,
  precision/recall/F1, AUROC, AUPRC, FPR/FNR, and Brier score.

It evaluates these 15 transformed conditions, plus clean:

| Transform | Values |
|---|---|
| JPEG quality | 90, 70, 50, 30 |
| Blur (sigma) | 0.5, 1.0, 2.0 |
| Resize (scale) | 0.5x, 0.25x |
| Noise (sigma) | 0.02, 0.05, 0.10 |
| Colour jitter | ±20% |
| Crop (fraction) | 80% |

The robustness run writes:

- `robustness_predictions.csv` — one row per image, model, and condition.
- `robustness_metrics.csv` — full metrics for each model and condition.
- `robustness_comparison.csv` — A/B accuracy, drop from clean, and the better
  model for every condition.
- `robustness_config.json` — split, seed, preprocessing, condition grid, and
  checkpoint hashes used for the run.

### 7. Run predict.py for the final submission

This step is real today — verified against the actual CLI:

```bash
python predict.py --input_dir <path_to_test_images> --checkpoint <path_to_checkpoint.pt> --output results/preds.json --batch_size 32
```

(identical command on Windows PowerShell and Mac/Linux; just make sure the
venv from step 1 is active)

TODO: fill in the real `--checkpoint` path once an experiment (A/B/C — TBD)
produces the checkpoint we're submitting.

## Results tables

### Clean validation — Experiments A and B

Both trained on CIFAKE (68,712 train / 14,724 val images), 5 epochs, batch
size 32, AdamW (lr 1e-4, weight decay 1e-4), AMP enabled; best checkpoint
selected by lowest validation loss (epoch 5 for both).

| Experiment | Training mode | Val Accuracy | Val Loss |
|---|---|---:|---:|
| A — Clean baseline | clean | 98.31% | 0.0528 |
| B — Robustness augmentation | robust | 98.02% | 0.0563 |

Source: [results/experiment_a_summary.md](results/experiment_a_summary.md),
[results/experiment_b_summary.md](results/experiment_b_summary.md).
Experiment B gives up only 0.29 points of clean accuracy relative to A —
expected, since B optimizes for robustness rather than clean-set accuracy.

### Robustness evaluation — Experiment B vs. Experiment C

Source: `results/robustness_comparison.csv`. In that file's `model_a`/
`model_b` columns, "a" and "b" mean first/second comparison slot, **not**
Experiment A/B — for this run, slot **a is Experiment B** (robustness
augmentation) and slot **b is Experiment C** (consistency training). The
table below is relabeled with the actual experiment names to avoid that
confusion.

| Condition | Exp B Acc | Exp C Acc | Exp B AUROC | Exp C AUROC | Exp B drop from clean | Exp C drop from clean |
|---|---:|---:|---:|---:|---:|---:|
| clean | 98.03% | 98.21% | 99.80% | 99.84% | — | — |
| jpeg q90 | 98.00% | 98.08% | 99.79% | 99.81% | 0.03pp | 0.14pp |
| jpeg q70 | 98.06% | 98.10% | 99.81% | 99.83% | -0.03pp | 0.12pp |
| jpeg q50 | 97.34% | 97.17% | 99.64% | 99.58% | 0.69pp | 1.05pp |
| jpeg q30 | 96.46% | 96.04% | 99.44% | 99.38% | 1.57pp | 2.17pp |
| blur σ0.5 | 97.53% | 97.33% | 99.72% | 99.73% | 0.50pp | 0.88pp |
| blur σ1.0 | 95.89% | 95.33% | 99.33% | 99.16% | 2.14pp | 2.88pp |
| blur σ2.0 | 91.77% | 91.13% | 97.68% | 97.26% | 6.26pp | 7.08pp |
| resize 0.5x | 95.35% | 95.41% | 99.32% | 99.20% | 2.68pp | 2.80pp |
| resize 0.25x | 91.37% | 90.96% | 97.54% | 97.04% | 6.66pp | 7.25pp |
| noise σ0.02 | 97.63% | 97.77% | 99.74% | 99.80% | 0.40pp | 0.44pp |
| noise σ0.05 | 96.84% | 97.20% | 99.60% | 99.67% | 1.20pp | 1.02pp |
| noise σ0.10 | 94.91% | 94.57% | 99.01% | 98.99% | 3.12pp | 3.64pp |
| colour -20% | 96.95% | 97.05% | 99.63% | 99.66% | 1.08pp | 1.17pp |
| colour +20% | 96.34% | 96.54% | 99.59% | 99.58% | 1.69pp | 1.67pp |
| crop 80% | 95.96% | 96.15% | 99.33% | 99.40% | 2.07pp | 2.06pp |

**Finding, stated plainly: consistency training (C) performed comparably
to augmentation alone (B) — this is not a clear win.** C was marginally
better on clean and on mild single-transform conditions (low JPEG
compression, light noise, colour jitter, crop), and marginally worse under
heavy single-transform degradation (blur, resize, aggressive JPEG/noise)
and under every chained transform but one. Every single-condition
difference between the two is under 1.5 percentage points either way — a
genuine tie within noise, not evidence the consistency loss achieved its
intended goal.

**Chained conditions.** Evaluation also covered 5 chained transforms
beyond the challenge's 15 single conditions, testing whether
transformations compose — real-world images are rarely degraded by just
one operation:

| Chain | Exp B Acc | Exp C Acc | Exp B drop from clean | Exp C drop from clean | Seen during training? |
|---|---:|---:|---:|---:|---|
| resize 0.5x + jpeg q70 | 95.48% | 94.70% | 2.55pp | 3.52pp | Yes |
| resize 0.25x + jpeg q50 | 88.90% | 87.86% | 9.13pp | 10.35pp | No |
| crop 80% + colour +20% + jpeg q50 | 93.32% | 91.93% | 4.71pp | 6.28pp | No |
| screenshot resample + jpeg q70 | 95.02% | 94.04% | 3.01pp | 4.17pp | No |
| repeated jpeg (q90→q70→q50) | 97.17% | 97.22% | 0.86pp | 0.99pp | No |

**resize 0.25x + jpeg q50 is the hardest condition in the entire
evaluation** — its ~9-10pp accuracy drop exceeds even the worst single
transform (blur σ2.0, ~6-7pp), for both models. Chaining transforms
compounds damage rather than just adding it. This is also the only chain
that resembles something the robustness augmentation's random parameter
ranges could have produced during training; the other three novel chains
(screenshot resample, triple-repeated JPEG, crop+colour+JPEG) were not,
and still degraded gracefully rather than catastrophically — except for
the resize+JPEG combination above.

**TODO — the missing piece: Experiment A (the clean baseline) has not
been evaluated under this robustness grid.** Every comparison above is
augmentation (B) vs. consistency (C); neither has been benchmarked
against the un-augmented baseline. Without A's robustness numbers, we
cannot currently quantify what robustness augmentation itself
contributed over doing nothing — only that B and C perform similarly to
*each other* once augmented. Running `evaluate.py --mode robustness`
with A's checkpoint against either B or C is required before the A/B/C
ablation in `results/decisions.md` can actually be completed.

TODO: error analysis (which images/generators/conditions each model gets
wrong) hasn't been done either.

## Limitations and what we'd improve with more time

TODO: known failure modes, transformation strengths not covered by
training/eval, and any planned improvements that didn't make the hackathon
deadline.

## Team member contributions

TODO: list each team member and what they owned (model/training, data
pipeline, inference/integration, evaluation, etc.).
