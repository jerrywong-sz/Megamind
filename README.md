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

`src/build_manifest.py` expects each dataset's raw images under `REAL/`
and `FAKE/` subfolders (searched recursively). A dataset with a tampered
holdout (SID_Set) also gets an optional `TAMPERED/` subfolder — the
crawler skips any of the three that doesn't exist, so a dataset without a
tampered set (CIFAKE) only needs `REAL/`/`FAKE/`:

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
    TAMPERED/
      ...
```

### 4. Build the manifest CSV

`src/build_manifest.py` now has a real CLI (verified against its current
`argparse` setup — note the flags use underscores, not hyphens):

```bash
python src/build_manifest.py --data_dir <raw_dataset_dir> --output_dir <standardized_output_dir> --dataset_name <name> --generator <name> --output_csv <manifest.csv>
```

(identical command on Windows PowerShell and Mac/Linux)

Example for CIFAKE:

```bash
python src/build_manifest.py --data_dir data/CIFAKE/train --output_dir data/CIFAKE_standardized --dataset_name CIFAKE --generator SD1.4 --output_csv data/cifake_manifest.csv
```

Example for SID_Set (has a `TAMPERED/` folder):

```bash
python src/build_manifest.py --data_dir data/SID_Set --output_dir data/SID_Set_standardized --dataset_name SID_Set --generator <sid_generator_name> --output_csv data/sidset_manifest.csv
```

What it actually does, read from the current source:

- Walks `REAL/` (label `0.0`), `FAKE/` (label `1.0`), and `TAMPERED/`
  (label `2.0`) under `--data_dir`; a missing subfolder is skipped, so
  CIFAKE (no tampered set) works with just `REAL/`/`FAKE/`.
- Verifies each image isn't corrupted, hashes it for de-duplication,
  converts it to RGB, and re-saves it as a standardized JPEG (quality 95)
  under `--output_dir` — this neutralizes format bias, so the model can't
  learn "file format" as a shortcut for real-vs-AI. Note: the re-saved
  file keeps its **original extension** even though its bytes are
  JPEG-encoded (the code computes a `.jpg`-suffixed path but doesn't use
  the result) — this doesn't break loading since Pillow reads by content,
  but don't be surprised to find JPEG data in a `.png`-named file if you
  inspect `--output_dir` by hand.
- `REAL`/`FAKE` rows are balanced 50/50 and split 70/15/15 into
  train/val/test, as before.
- **`TAMPERED` rows are excluded from that balancing/split** and instead
  get `split = "bonus"` — a separate holdout, never mixed into
  train/val/test.
- The manifest's `image_path` column is the **re-saved** file's path
  (`--output_dir` + the image's path relative to `--data_dir`), not the
  original raw path.

Run once per dataset. TODO: still no built-in way to merge multiple
datasets' manifests into one combined CSV for `get_dataloaders()` — do
that manually (e.g. `pandas.concat`) until that's added.

**Important for later steps:** since `image_path` already has
`--output_dir` baked into it, pass `--data-root .` to `train.py` /
`evaluate.py` (i.e. run them from the same working directory you ran this
command from) rather than pointing `--data-root` at `--output_dir` itself
— otherwise the path gets prefixed twice and image loading will fail.

### 5. Train the three experiments

`train.py` trains one experiment per run from a YAML config plus a
manifest/data-root pair. Verified CLI (from its `argparse` setup):

```bash
python train.py --config <config.yaml> --manifest <manifest.csv> --data-root <dataset_root>
```

- `--config` (required) — experiment YAML from `configs/`
- `--manifest` (required) — the CSV manifest built in step 4
- `--data-root` (required) — root directory the manifest's `image_path` values are relative to

(identical arguments on Windows PowerShell and Mac/Linux)

**Experiment A (clean baseline)** —
[configs/baseline.yaml](configs/baseline.yaml):

```bash
python train.py --config configs/baseline.yaml --manifest <manifest.csv> --data-root <dataset_root>
```

**Experiment B (robustness augmentation)** —
[configs/robustness.yaml](configs/robustness.yaml):

```bash
python train.py --config configs/robustness.yaml --manifest <manifest.csv> --data-root <dataset_root>
```

The two configs share every hyperparameter except `train_mode`, so the
ablation isolates the effect of robustness augmentation rather than a
confound:

| | A — `baseline.yaml` | B — `robustness.yaml` |
|---|---|---|
| seed | 42 | 42 |
| pretrained | true | true |
| image_size | 224 | 224 |
| batch_size | 32 | 32 |
| epochs | 5 | 5 |
| learning_rate | 1e-4 | 1e-4 |
| weight_decay | 1e-4 | 1e-4 |
| amp | true | true |
| train_mode | clean (default) | robust — applies `random_transform()` via `get_robust_train_transform()` in [src/data.py](src/data.py) |
| checkpoint_name | baseline.pt | robustness.pt |

Both A and B have already been trained; see **Results tables** below.

**Experiment C (consistency):** same robustness augmentation as B, plus a
consistency penalty between clean and damaged views of the same image
(design recorded in [results/decisions.md](results/decisions.md)). TODO —
not run yet: `robustness_loss()` in [src/losses.py](src/losses.py) already
implements the classification-plus-consistency loss, but `train.py` isn't
wired to use it yet (it only calls plain `BCEWithLogitsLoss`), and no
`configs/experiment_c.yaml` exists.

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

**Not yet run:** the robustness evaluation across the 15 fixed transform
conditions (JPEG 90/70/50/30, blur 0.5/1.0/2.0, resize 0.5x/0.25x, noise
0.02/0.05/0.10, colour ±20%, crop 80% — step 6 above) hasn't been executed;
no `robustness_metrics.csv` or `robustness_comparison.csv` exists in
`results/` yet. **The table above is clean performance only — it says
nothing about robustness, which is this project's actual target metric.**
TODO: fill in per-condition accuracy and the A-vs-B robustness comparison
once `evaluate.py --mode robustness` has been run. TODO: error analysis
(which images/generators/conditions each model gets wrong) hasn't been done
either.

## Limitations and what we'd improve with more time

TODO: known failure modes, transformation strengths not covered by
training/eval, and any planned improvements that didn't make the hackathon
deadline.

## Team member contributions

TODO: list each team member and what they owned (model/training, data
pipeline, inference/integration, evaluation, etc.).
