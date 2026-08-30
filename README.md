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

> ⚠️ **Known issue — this step does not currently run on `main`.**
> `src/build_manifest.py` raises `IndexError` at the 70/15/15 split
> (`np.split` returns plain numpy arrays rather than DataFrames with the
> installed pandas/numpy, so `train['split'] = ...` fails). Calling
> `build_and_split_manifest()` as a function fails earlier still, with
> `NameError: name 'args' is not defined`, because the body reads the
> module-level `args` for `--force_split` instead of taking it as a
> parameter. Both are being fixed — until then, treat the commands below
> as the intended interface rather than something that completes today.

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
- Holds `TAMPERED` rows out of that balancing and split entirely, giving
  them their own split value.
- Writes `image_path` as the path of the **re-saved** file relative to
  `--output_dir`.

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

TODO: known failure modes, transformation strengths not covered by
training/eval, and any planned improvements that didn't make the hackathon
deadline.

## Team member contributions

TODO: list each team member and what they owned (model/training, data
pipeline, inference/integration, evaluation, etc.).
