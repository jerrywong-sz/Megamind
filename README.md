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
- `--architecture` (optional) — which backbone to build (`efficientnet_b0`
  or `convnext_tiny`). **Normally unnecessary:** checkpoints written by
  `train.py` record their own architecture, and that recorded value is used.
  This flag supplies the architecture for older checkpoints saved before
  that metadata existed; if the checkpoint does record one and you pass a
  different value, `predict.py` raises rather than guessing. With neither a
  recorded value nor this flag, it falls back to `efficientnet_b0`
- `--include-label` (optional, off by default) — adds a `predicted_label`
  key (`0`/`1`) to each row. **Leave this off for submission** — the
  required output format is exactly `image_path` + `pred`
- `--threshold` (optional, default `0.5`) — probability at or above which
  an image is labelled AI-generated. Only affects `predicted_label`, so it
  does nothing unless `--include-label` is also passed

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

`src/build_manifest.py` looks for up to three subfolders under the
directory you point it at: `REAL/`, `FAKE/`, and `TAMPERED/` (each
searched recursively). A subfolder that doesn't exist is skipped, so a
dataset without tampered images works with just `REAL/`/`FAKE/`:

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
    TAMPERED/          # SID_Set only -- held out, see step 4
      ...
```

Labels are assigned from the folder name: `REAL` → `0.0`, `FAKE` → `1.0`,
`TAMPERED` → `2.0`.

### 4. Build the manifest CSV

`src/build_manifest.py` has a five-argument CLI (note: flags use
underscores, unlike `train.py`/`evaluate.py` which use hyphens):

```bash
python src/build_manifest.py --data_dir <raw_dataset_dir> --output_dir <standardized_output_dir> --dataset_name <name> --generator <name> --output_csv <manifest.csv>
```

(identical command on Windows PowerShell and Mac/Linux)

Example for CIFAKE:

```bash
python src/build_manifest.py --data_dir data/CIFAKE/train --output_dir data/CIFAKE_standardized --dataset_name CIFAKE --generator SD1.4 --output_csv data/cifake_manifest.csv
```

Arguments:

- `--data_dir` (required) — the raw downloaded images (the folder holding
  `REAL/`/`FAKE/`/`TAMPERED/`)
- `--output_dir` (required) — where standardized JPEGs are written
- `--dataset_name` (required) — recorded in the manifest's `dataset` column
- `--generator` (required) — recorded in the manifest's `generator` column
- `--output_csv` (required) — path for the manifest CSV
- `--force_split` (optional, one of `train`/`val`/`test`) — force every
  binary row into a single split. Intended for held-out sets such as
  WildFake, so they can never leak into training

What it does:

- Verifies each image isn't corrupted, hashes it (SHA-256) to
  de-duplicate, converts to RGB, and re-saves it as a standardized JPEG
  (quality 95) under `--output_dir`. This neutralizes format bias — see
  [results/decisions.md](results/decisions.md).
- Balances `REAL`/`FAKE` 50/50 and splits them 70/15/15 into
  train/val/test.
- Holds `TAMPERED` rows out of that balancing and out of the 70/15/15
  split, assigning them `split = "bonus"` — a value the binary splits never
  use, so `train`/`val`/`test` stay purely binary and a
  `split == "test"` filter cannot pick up 3-class rows. See
  [results/decisions.md](results/decisions.md).
- Writes `image_path` as the path of the **re-saved** file relative to
  `--output_dir`. Note the re-saved file keeps its **original extension**
  even though its bytes are JPEG — a `.png` input stays named `.png`.
  Loading is unaffected (Pillow reads by content), but the output
  directory is misleading to inspect by hand.

Run once per dataset. TODO: there's still no built-in way to merge
multiple datasets' manifests into one combined CSV for
`get_dataloaders()` — do that manually (e.g. `pandas.concat`) until
that's added.

### 5. Train the three experiments

`train.py` trains one experiment per run, selected by config file:

```bash
python train.py --config <config.yaml> --manifest <manifest.csv> --data-root <dataset_root>
```

- `--config` (required) — experiment YAML from `configs/`
- `--manifest` (required) — the CSV manifest built in step 4
- `--data-root` (required) — root directory the manifest's `image_path`
  values are relative to (note: hyphen, unlike step 4's underscore flags)

The three experiments:

```bash
# A -- clean baseline
python train.py --config configs/baseline.yaml --manifest <manifest.csv> --data-root <dataset_root>

# B -- robustness augmentation
python train.py --config configs/robustness.yaml --manifest <manifest.csv> --data-root <dataset_root>

# C -- consistency training (see known issue below)
python train.py --config configs/consistency.yaml --manifest <manifest.csv> --data-root <dataset_root>
```

> ⚠️ **Known issue — Experiment C does not currently run.** The training
> loop in `train.py` handles `train_mode: consistency`, but
> `get_dataloaders()` in [src/data.py](src/data.py) accepts only `clean`
> or `robust` and raises `ValueError: Unknown train_mode 'consistency'`
> before training starts. A and B run fine. The C results reported below
> came from a separate run, not from this command as written.

- **Experiment A** ([configs/baseline.yaml](configs/baseline.yaml)) — base
  preprocessing only, via `get_train_transform()` in
  [src/data.py](src/data.py) (`Resize(256) → CenterCrop(224) →
  RandomHorizontalFlip → Normalize`). No robustness damage.
- **Experiment B** ([configs/robustness.yaml](configs/robustness.yaml),
  `train_mode: robust`) — same base preprocessing plus `random_transform()`
  from [src/augmentations.py](src/augmentations.py), applying 0–3 of
  {JPEG, blur, resize, noise, colour, crop} per image at random strengths.
- **Experiment C** ([configs/consistency.yaml](configs/consistency.yaml),
  `train_mode: consistency`) — same augmentation as B, plus a consistency
  penalty between a clean and a damaged view of each image, implemented as
  `robustness_loss()` in [src/losses.py](src/losses.py). Weighted by
  `lambda_weight` (currently `0.5`).

All three configs hold the ablation variables identical — seed `42`,
`pretrained: true`, image size `224`, batch size `32`, `5` epochs, lr
`1e-4`, weight decay `1e-4`, AMP on — so the only thing that varies is the
training method. Checkpoints are written to `checkpoint_dir`/`checkpoint_name`
(`checkpoints/baseline.pt`, `robustness.pt`, `consistency.pt`); `checkpoints/`
and `*.pt` are gitignored.

Checkpoints are saved as a dict (`{epoch, model_state, optimizer_state}`)
for resumability. `predict.py` and `evaluate.py` both accept either that
form or a bare state dict.

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

### 6.1 Fixed chained robustness across Models A, B, and C

`evaluate_fixed_robustness_abc.py` is the explicit three-model runner. It
avoids generic spreadsheet headings such as `model_a_accuracy` when the first
slot actually contains Model B. The model roles are fixed and fully named:

- `model_a_clean_baseline` — Model A, trained on clean images.
- `model_b_robustness` — Model B, trained with robustness augmentation.
- `model_c_consistency` — Model C, trained with robustness augmentation plus
  consistency loss.

The default `all-fixed` condition set runs clean, all 15 official single
conditions, and these five deterministic chained conditions:

| Condition title | Ordered transforms |
|---|---|
| Fixed resize-half re-encode | Resize 0.50× → JPEG quality 70 |
| Fixed resize-quarter re-encode | Resize 0.25× → JPEG quality 50 |
| Fixed crop/colour/re-encode | Crop 0.80 → Colour +0.20 → JPEG quality 50 |
| Fixed screenshot resample | Resize 0.50× → Blur 0.50 → JPEG quality 70 |
| Fixed repeated JPEG | JPEG quality 90 → JPEG 70 → JPEG 50 |

Every condition starts again from the original image. Models A, B, and C see
the same manifest rows, transform order, transform parameters, threshold, and
seed. This runner performs no random condition selection.

```bash
python evaluate_fixed_robustness_abc.py \
  --data-root <dataset_root> \
  --manifest <manifest.csv> \
  --checkpoint-a-baseline <experiment_a_baseline_best.pt> \
  --checkpoint-b-robustness <experiment_b_robustness_best.pt> \
  --checkpoint-c-consistency <experiment_c_consistency_best.pt> \
  --output-dir results/fixed_robustness_abc_validation \
  --split val \
  --condition-set all-fixed \
  --device auto \
  --seed 42
```

Use `--condition-set fixed-chains-only` to run just clean plus the five chains.
The default full run writes filenames that state the exact models involved:

- `fixed_robustness__model_A_baseline_vs_model_B_robustness.csv`
- `fixed_robustness__model_B_robustness_vs_model_C_consistency.csv`
- `fixed_robustness__model_A_baseline_vs_model_C_consistency.csv`
- `fixed_robustness__models_A_B_C__combined_summary.csv`
- `fixed_robustness__models_A_B_C__full_metrics.csv`
- `fixed_robustness__models_A_B_C__per_image_predictions.csv`
- `fixed_robustness__models_A_B_C__run_config.json`

The metrics and comparison files include the confusion-matrix counts (true
negatives, false positives, false negatives, and true positives), accuracy,
balanced accuracy, precision, recall, F1, AUROC, AUPRC, false-positive rate,
false-negative rate, Brier score, changes from clean, pairwise differences,
and winners for both higher-is-better and lower-is-better metrics. Each row
also contains a human-readable condition title, the exact ordered chain, and
JSON-encoded transform parameter names and values.

`possible_in_b_c_training_sampler` means that the chain structure and values
could be generated by the B/C training sampler. It does not claim that the
exact transformed image was definitely shown during training.

### 6.2 Seeded random three-transform evaluation

`evaluate_random_robustness_abc.py` is the first random-chain milestone. For
each image and trial it selects three distinct transform types, randomises
their order, and samples each parameter from the official challenge grid.
The assignment is derived from `dataset_id + image_path + trial_seed`, so all
three models receive identical damaged pixels and any result can be recreated.

The default five trials use seeds `42 43 44 45 46`. CIFAKE and SID_Set should
be run separately with different `--dataset-id` and output directories rather
than mixing their rows into one accuracy figure.

```bash
python evaluate_random_robustness_abc.py \
  --dataset-id CIFAKE \
  --data-root <dataset_root> \
  --manifest <manifest.csv> \
  --checkpoint-a-baseline <experiment_a_baseline_best.pt> \
  --checkpoint-b-robustness <experiment_b_robustness_best.pt> \
  --checkpoint-c-consistency <experiment_c_consistency_best.pt> \
  --output-dir results/cifake_random_standard_3 \
  --split val \
  --trial-seeds 42 43 44 45 46 \
  --device auto
```

The primary result is pooled accuracy across all random trials, accompanied
by the mean and standard deviation across seeds and the drop from clean. The
runner also writes A/B, B/C, and A/C comparisons, ordered-pattern and
transform-inclusion breakdowns, every per-image chain assignment, all
per-image predictions, false positives and false negatives, clean-to-damaged
prediction changes, checkpoint hashes, and the complete random policy.

### 7. Run predict.py for the final submission

This step is real today — verified against the actual CLI:

```bash
python predict.py --input_dir <path_to_test_images> --checkpoint <path_to_checkpoint.pt> --output results/preds.json --batch_size 32
```

(identical command on Windows PowerShell and Mac/Linux; just make sure the
venv from step 1 is active)

Checkpoints written by `train.py` record which backbone they were trained
with, so the command above works for both EfficientNet-B0 and ConvNeXt-Tiny
without being told which is which. Only for a checkpoint saved before that
metadata existed do you need to name the architecture explicitly:

```bash
python predict.py --input_dir <path_to_test_images> --checkpoint <legacy_checkpoint.pt> --architecture convnext_tiny --output results/preds.json
```

TODO: fill in the real `--checkpoint` path once an experiment (A/B/C — TBD)
produces the checkpoint we're submitting.

## Results tables

> **Scope of every result below:** all of it was trained and evaluated on
> **CIFAKE** only — 32×32 images, **Stable Diffusion 1.4** as the sole
> generator. None of it touches **SID_Set** yet. These numbers say nothing
> about performance on high-resolution images or on generators the model
> hasn't seen — that generalization is untested.

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

Full clean-validation breakdown, from `results/clean_metrics_a_b.csv`
(14,724 validation images for both):

| Metric | Experiment A | Experiment B |
|---|---:|---:|
| Precision | 98.09% | 97.42% |
| Recall | 98.59% | 98.73% |
| F1 | 98.34% | 98.07% |
| AUROC | 99.84% | 99.80% |
| False positive rate | 1.97% | 2.69% |
| False negative rate | 1.41% | 1.27% |
| Brier score | 0.0133 | 0.0153 |

Confusion-matrix counts (out of 14,724): A — 7,118 true negatives, 143
false positives, 105 false negatives, 7,358 true positives. B — 7,066 true
negatives, **195 false positives**, **95 false negatives**, 7,368 true
positives.

**Observed trade-off:** Experiment B has more false positives than A (195
vs. 143) but fewer false negatives (95 vs. 105) — robustness augmentation
shifted the model's decision boundary slightly toward calling images
AI-generated. False positive rate rose from 1.97% (A) to 2.69% (B). This
is worth tracking rather than dismissing: the challenge specifically warns
against false positives on genuine photographs, and this shift moves in
that direction, even though it's a small one at this stage.

`clean_metrics_a_b.csv` also records both checkpoints' SHA-256 hashes
(`checkpoint_hash` column), which partially addresses the
`robustness_config.json` traceability TODO below — the clean-evaluation
checkpoints are identifiable now, though the robustness-grid run still
isn't.

### Robustness evaluation — Experiment A vs. Experiment B

![Accuracy under increasing transformation severity: six panels (JPEG, blur,
resize, noise, colour jitter, crop) each starting from the clean baseline. The
clean baseline collapses toward chance under blur, resize and noise, while the
augmented model stays above 91% throughout.](results/robustness_curves.png)

Regenerate with `python scripts/plot_robustness.py`.

Source: `results/robustness_comparison_a_vs_b.csv` (accuracy, per-condition)
and `results/robustness_metrics_a_b.csv` (AUROC and other per-condition
metrics).

| Condition | Exp A Acc | Exp B Acc | Exp A AUROC | Exp B AUROC | Exp A drop from clean | Exp B drop from clean |
|---|---:|---:|---:|---:|---:|---:|
| clean | 98.32% | 98.03% | 99.84% | 99.80% | — | — |
| jpeg q90 | 98.26% | 98.00% | 99.81% | 99.79% | 0.05pp | 0.03pp |
| jpeg q70 | 98.06% | 98.06% | 99.80% | 99.81% | 0.25pp | -0.03pp |
| jpeg q50 | 96.53% | 97.34% | 99.40% | 99.64% | 1.79pp | 0.69pp |
| jpeg q30 | 92.89% | 96.46% | 98.80% | 99.44% | 5.43pp | 1.57pp |
| blur σ0.5 | 85.33% | 97.53% | 99.36% | 99.72% | 12.99pp | 0.50pp |
| blur σ1.0 | 65.84% | 95.89% | 80.40% | 99.33% | 32.48pp | 2.14pp |
| blur σ2.0 | 57.75% | 91.77% | 77.07% | 97.68% | 40.57pp | 6.26pp |
| resize 0.5x | 62.33% | 95.35% | 84.88% | 99.32% | 35.98pp | 2.68pp |
| resize 0.25x | 60.74% | 91.37% | 74.78% | 97.54% | 37.57pp | 6.66pp |
| noise σ0.02 | 96.59% | 97.63% | 99.60% | 99.74% | 1.73pp | 0.40pp |
| noise σ0.05 | 88.16% | 96.84% | 96.46% | 99.60% | 10.15pp | 1.20pp |
| noise σ0.10 | 53.67% | 94.91% | 81.54% | 99.01% | 44.65pp | 3.12pp |
| colour -20% | 95.25% | 96.95% | 99.39% | 99.63% | 3.07pp | 1.08pp |
| colour +20% | 95.14% | 96.34% | 99.39% | 99.59% | 3.18pp | 1.69pp |
| crop 80% | 76.52% | 95.96% | 98.13% | 99.33% | 21.79pp | 2.07pp |

**Headline finding: the clean baseline collapses under degradation; the
augmented model holds.** At noise σ=0.10 — a level that's barely visible
to the eye — Experiment A's accuracy falls from 98.32% to 53.67% (a
44.65-point drop, barely above the 50% chance rate for this binary task),
while Experiment B falls only to 94.91% (a 3.12-point drop). The same
pattern repeats under blur (σ=2.0: A collapses to 57.75% vs. B at
91.77%), resize (0.25x: A at 60.74% vs. B at 91.37%), and centre crop
(80%: A at 76.52% vs. B at 95.96%). **The trade is a small one:**
robustness augmentation costs only 0.29 points of clean accuracy
(98.32% → 98.03%) in exchange for these large robustness gains.

### Robustness evaluation — Experiment B vs. Experiment C

Source: `results/robustness_comparison_b_vs_c.csv`. In that file's
`model_a`/`model_b` columns, "a" and "b" mean first/second comparison
slot, **not** Experiment A/B — for this run, slot **a is Experiment B**
(robustness augmentation) and slot **b is Experiment C** (consistency
training). The table below is relabeled with the actual experiment names
to avoid that confusion.

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

**Finding, stated plainly: adding consistency training on top of
augmentation does not move the needle.** C was marginally better than B
on clean and mild single-transform conditions (low JPEG compression,
light noise, colour jitter, crop), and marginally worse under heavy
single-transform degradation (blur, resize, aggressive JPEG/noise) and
under every chained transform but one. Every single-condition difference
between B and C is under 1.5 percentage points either way — a genuine tie
within noise, not evidence the consistency loss achieved its intended
goal.

**Put together with the A-vs-B result above, the full ablation tells one
story:** almost all of the robustness gain comes from augmentation itself
(A→B, tens of accuracy points under heavy degradation). Adding the
consistency penalty on top of that (B→C) adds nothing measurable — C is
statistically indistinguishable from B across every single-transform
condition.

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

### Data provenance

- Per-condition metrics used in both tables above —
  `results/robustness_comparison_a_vs_b.csv`,
  `results/robustness_comparison_b_vs_c.csv`,
  `results/robustness_metrics_a_b.csv` — are committed to the repo.
- Per-image predictions (`robustness_predictions.csv`, ~89MB) are
  **gitignored due to size** and are not in the repo. Full per-image error
  analysis (which specific images each model gets wrong) would need to be
  regenerated locally by re-running `evaluate.py`.
- TODO: `robustness_config.json` (the checkpoint SHA-256 hashes, split,
  seed, and condition grid `evaluate.py` writes automatically for each
  run) **has not been committed yet**. Until it is, these numbers aren't
  yet traceable to a specific checkpoint file with certainty.

TODO: error analysis (which images/generators/conditions each model gets
wrong) hasn't been done either.

## Limitations and what we'd improve with more time

Every figure below comes from
[results/error_analysis.md](results/error_analysis.md) and the CSVs in
`results/`.

**Everything was trained and evaluated on CIFAKE alone.** That means 32×32
images with Stable Diffusion 1.4 as the only generator, on a single
validation split of 14,724 images. We have **no evidence** about
high-resolution images or about generators the model never saw. SID_Set was
obtained but never trained on, and the WildFake cross-generator benchmark
was never run. Our robustness claim is therefore "robust to
transformations *within* CIFAKE", not "robust in general" — the headline
accuracies would very likely not survive a change of generator or
resolution.

**Consistency training (Experiment C) produced no measurable improvement.**
Across the 16 single-transform conditions, C beat B in 9 and lost in 7,
with a mean difference of −0.09 percentage points and a maximum absolute
difference of 0.64pp (1.39pp including chained conditions). On clean images
C is 98.21% against B's 98.03%. Differences that small on a single
validation split are indistinguishable from noise, so we cannot claim the
consistency loss did anything. Essentially all of the robustness gain comes
from augmentation (A→B); adding the consistency penalty on top (B→C) added
nothing we can measure.

**The clean baseline collapses under degradation.** Experiment A falls from
98.32% clean accuracy to **53.67% at noise σ=0.10** — barely above the 50%
chance rate for a balanced binary task — along with 57.75% at blur σ=2.0,
60.74% at resize 0.25×, and 76.52% at crop 80%. Experiment B holds at
94.91%, 91.77%, 91.37% and 95.96% on those same conditions. Across all 15
transformed conditions the baseline suffers 38,504 transformation flips
(images it got right clean and wrong after a transform) versus B's 5,611.
This is the single clearest result we have, and it is the reason
augmentation is in the final pipeline.

**Robustness costs a little clean-set precision.** On clean validation
images Experiment B has a higher false-positive rate than A: **2.69%
versus 1.97%** (195 versus 143 false positives out of 7,261 real images),
against a lower false-negative rate (1.27% versus 1.41%). Augmentation
shifts the decision boundary slightly toward calling images AI-generated.
The trade is favourable overall, but it moves in the direction the
challenge explicitly warns about — false accusations on genuine
photographs — and we did not tune the threshold to compensate.

**40 images defeat Experiment B under all 16 conditions.** No amount of
augmentation moves them, which suggests label noise or intrinsic ambiguity
rather than a robustness failure. We did not open these images to check.
They are a small fraction of 14,724, but they are the most likely place to
find a mislabelled subset of CIFAKE, and inspecting them is the cheapest
remaining diagnostic.

**We could not break errors down by source or generator.** The prediction
dump's `source` column is null in 100% of rows, and `dataset` and
`generator` are constant (`CIFAKE`, `SD1.4`) across every row — including
real images, where `generator` is a dataset-level tag rather than a
per-image attribute. The only available split was real versus AI. Any
claim about *which kinds* of images fail is therefore out of reach with
the data we logged.

**Model outputs are strongly bimodal, and only Experiment B's confidence
is trustworthy.** Just 8.3% (A) and 7.7% (B) of predictions land in the
middle 0.20–0.80 band; 12.1% (A) and 7.0% (B) sit at exactly 1.0. That
saturation is expected from a single-logit network trained with
`BCEWithLogitsLoss` to near-zero training loss. Bimodality alone is not a
defect — but calibration differs sharply between the two models. In B's
≥0.99 band, 99.0% of images really are AI; in A's >0.9 band, only 85.8%
are, and images A scores below 0.1 are still AI 11.9% of the time. So B's
`pred` values are usable as probabilities for thresholding or ranking;
A's are not. (Measured across all 16 conditions, so A's figure is dragged
down substantially by the conditions where it collapses.)

**The tampered holdout is correctly quarantined, but the analysis it
enables has never been run.** `src/build_manifest.py` routes tampered
SID_Set images (label `2.0`) to a dedicated `bonus` split, and
`get_evaluation_dataloader()` accepts that split, so the data is reachable
in principle. Nothing consumes it, though: `get_dataloaders()` builds only
`train` and `val`, and all three evaluation entry points (`evaluate.py`,
`evaluate_fixed_robustness_abc.py`, `evaluate_random_robustness_abc.py`)
restrict `--split` to `val` or `test`, so **no runnable command currently
targets the bonus split** and no bonus results exist under `results/`.
Labels are emitted as float32 for `BCEWithLogitsLoss`, so a `2.0` row would
also need explicit handling before any binary metric could be computed on
it. The quarantine is working as designed — no tampered data can leak into
binary training or evaluation — but the separate tampered analysis remains
future work. See [results/decisions.md](results/decisions.md).

**Checkpoint loading is not hardened against untrusted files.** Both
`predict.py` and `evaluate.py` call
`torch.load(..., weights_only=False)`, which permits arbitrary code
execution during unpickling. It is set that way so the loader can read the
`architecture` string stored alongside the weights, which
`weights_only=True` would reject. This is safe for checkpoints we produced
ourselves, but it means **neither script should be pointed at a checkpoint
from an untrusted source.** With more time we would either register the
needed types via `torch.serialization.add_safe_globals` and restore
`weights_only=True`, or store the architecture metadata outside the pickle
(a sidecar JSON, or the checkpoint filename) so the weights can be loaded
under the safe path.

**What we'd do with more time.** In priority order: retrain on SID_Set so
the detector sees high-resolution images and a second generator; run the
held-out WildFake subset (COCO val2017 + DALL·E Advanced) to get a real
cross-generator generalization number, which is currently our largest
unknown; and manually inspect the 40 persistent failures to establish
whether CIFAKE carries label noise. Beyond that: tune the decision
threshold to claw back Experiment B's false-positive rate, and re-run the
A/B/C ablation across multiple seeds so a 0.64pp difference between B and
C could actually be called significant or not.

## Team member contributions

**Isaac — Model Lead.** Owned the model architecture and training pipeline,
and conducted Experiments A/B/C covering clean training, robustness
augmentation, and consistency training. Main contributions include
`train.py`, the model definition in `src/models.py`, the consistency loss in
`src/losses.py`, the experiment configs in `configs/`, and the
experiment/checkpoint workflow.

**Jerry — Integration Lead.** Repo setup and initial structure; the
inference pipeline (`predict.py`) and its smoke tests; the shared
evaluation-time preprocessing (`get_eval_transform()` in `src/data.py`,
the single source of truth that keeps training and inference from
drifting apart) and the `build_model()` interface that `predict.py` and
`evaluate.py` both consume; error analysis (`scripts/error_analysis.py`,
`results/error_analysis.md`) and the robustness severity curves
(`scripts/plot_robustness.py`); and the README, reproduction steps, and
design-decision notes in `results/decisions.md`.

**Teoh Ke Yi — Data and Infrastructure Lead.** Prepared and published the
datasets the project runs on — CIFAKE (120,000 images; 100,000 train /
20,000 test) and SID_Set (35,000 images), both formatted and uploaded to
Kaggle — and established the tampered-image holdout that keeps 3-class data
out of binary training and evaluation. Owned the data pipeline end to end:
`src/build_manifest.py` (JPEG standardization to remove format bias,
SHA-256 de-duplication, 50/50 class balancing, and the
train/val/test/`bonus` split) and `src/data.py` (manifest-backed dataset
and split handling, aspect-ratio-preserving resize in the evaluation
transform, and `worker_init_fn` seeding so DataLoader workers do not draw
identical augmentations). Also contributed across the training and
inference paths: `configs/consistency.yaml` and the consistency
lambda-tuning wiring in `train.py`, epoch and optimizer state in
checkpoints for resumability, the mean / worst-case AUROC and
robustness-gap reporting in `evaluate_fixed_robustness_abc.py`, and the
CLI decision threshold plus the switch to `torch.inference_mode()` in
`predict.py`.

**Wei Jien — Evaluation and Benchmarking Lead.** Designed and implemented
the full model-evaluation workflow: checkpoint validation, clean
evaluation, binary metrics, fixed and seeded-random chained robustness
tests, fair A/B/C comparisons, and reproducible CIFAKE/SID Colab benchmarks
(`src/metrics.py`, `evaluate.py`, `evaluate_fixed_robustness_abc.py`,
`evaluate_random_robustness_abc.py`, `src/evaluation_conditions.py`,
`src/random_chain_conditions.py`). Validated dataset manifests, produced
per-image predictions, pairwise summaries and false-positive/false-negative
analysis, added evaluation tests and documentation, and interpreted results
to guide model selection and retraining.

**Tan Teck Heang — Augmentation and Consistency Training Lead.** Owned the
robustness augmentation pipeline and consistency-training objective,
including transform utilities, robustness-aware training support, and
consistency loss integration. Main contributions include
`src/augmentations.py`, `src/losses.py`, and related training/evaluation
configuration for robustness experiments.
