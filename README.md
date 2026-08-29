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

`evaluate.py` loads checkpoints A and B, evaluates both on the same manifest
rows in the same order, and records checkpoint hashes so a run can be traced
back to the exact model files.

First reproduce the clean validation results:

```bash
python evaluate.py --mode clean --data-root <dataset_root> --manifest <manifest.csv> --checkpoint-a <experiment_a.pt> --checkpoint-b <experiment_b.pt> --output-dir results/clean_validation --split val --device auto
```

After the clean integration check passes, run the fixed robustness grid:

```bash
python evaluate.py --mode robustness --data-root <dataset_root> --manifest <manifest.csv> --checkpoint-a <experiment_a.pt> --checkpoint-b <experiment_b.pt> --output-dir results/robustness_validation --split val --device auto --seed 42
```

The robustness runner uses:

- `exact_transform(img, name, param)` in
  [src/augmentations.py](src/augmentations.py) to apply each named transform.
  Names match the six challenge dimensions: `jpeg`, `blur`, `resize`, `noise`,
  `colour`, `crop`. Gaussian noise is seeded per image so A and B receive the
  same noisy pixels and later runs are reproducible.
- `exact_transform_chain(img, steps)` in
  [src/augmentations.py](src/augmentations.py) to apply realistic deployment
  chains such as resize followed by JPEG compression or repeated JPEG saves.
- `compute_binary_metrics(labels, probabilities, threshold=0.5)` in
  [src/metrics.py](src/metrics.py) — returns accuracy, balanced accuracy,
  precision/recall/F1, AUROC, AUPRC, FPR/FNR, and Brier score.

It evaluates these 15 individual transformed conditions, plus clean:

| Transform | Values |
|---|---|
| JPEG quality | 90, 70, 50, 30 |
| Blur (sigma) | 0.5, 1.0, 2.0 |
| Resize (scale) | 0.5x, 0.25x |
| Noise (sigma) | 0.02, 0.05, 0.10 |
| Colour jitter | ±20% |
| Crop (fraction) | 80% |

It also evaluates mixed robustness chains that better match deployment and
sharing pipelines:

| Chain | Seen in robustness training? |
|---|---|
| Resize 0.5x -> JPEG q70 | Yes |
| Resize 0.25x -> JPEG q50 | No |
| Crop 80% -> Colour +20% -> JPEG q50 | No |
| Resize 0.5x -> Blur 0.5 -> JPEG q70 | No |
| JPEG q90 -> JPEG q70 -> JPEG q50 | No |

For extra stress testing, append deterministic random corruption chains:

```bash
python evaluate.py --mode robustness --data-root <dataset_root> --manifest <manifest.csv> --checkpoint-a <experiment_a.pt> --checkpoint-b <experiment_b.pt> --output-dir results/robustness_stress --split val --device auto --seed 42 --stress-seed 42 --stress-count-mild 20 --stress-count-medium 20 --stress-count-strong 20
```

These sampled conditions are evaluation-only. They draw 1-4 realistic
transforms from severity-specific ranges, normally avoid duplicate transforms,
allow repeated JPEG only for recompression, and keep JPEG near the end of
mixed chains. The exact generated steps and parameters are recorded in
`robustness_config.json`.

The robustness run writes:

- `robustness_predictions.csv` — one row per image, model, and condition.
- `robustness_metrics.csv` — full metrics for each model and condition.
- `robustness_comparison.csv` — A/B accuracy, AUROC, drops from clean,
  seen/unseen labels, and the better model for every condition.
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

TODO: accuracy/AUC on clean test images, and accuracy under each
robustness transformation (compression, blur, resize, noise, colour jitter,
crop), once evaluation has been run.

## Limitations and what we'd improve with more time

TODO: known failure modes, transformation strengths not covered by
training/eval, and any planned improvements that didn't make the hackathon
deadline.

## Team member contributions

TODO: list each team member and what they owned (model/training, data
pipeline, inference/integration, evaluation, etc.).
