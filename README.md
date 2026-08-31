# techjam-aigc-detector

## Project overview

A binary image classifier that detects whether an image is AI-generated,
built for TikTok TechJam Track 5. The core challenge this project targets is
**robustness**: real-world images get compressed, blurred, resized, noised,
colour-jittered, and cropped before anyone ever sees them (e.g. after passing
through a social platform's upload pipeline), and a detector that only works
on pristine images isn't useful. The model is trained and evaluated to stay
accurate under these transformations, not just on clean inputs.

Architecture: **EfficientNet-B0** backbone with a single-logit head (sigmoid
-> probability the image is AI-generated). We also trained and evaluated
**ConvNeXt-Tiny** and **DINOv2 ViT-S/14** on the same data;
[src/models.py](src/models.py) builds all three, and
[src/data.py](src/data.py) holds the shared preprocessing pipeline used by
both training and inference.

EfficientNet-B0 is the architecture we recommend, and the comparison is part
of why: ConvNeXt-Tiny is statistically indistinguishable from it on SID
while costing 3.6× the inference time, and DINOv2 is materially weaker. The
more important result, though, is that **architecture matters far less than
what the model is trained on**. A model trained on one generator scores at
chance on another regardless of backbone — EfficientNet-B0 and ConvNeXt-Tiny
fail identically — while adding a second training source fixes it for about
half an accuracy point. See
[Cross-domain generalization](#cross-domain-generalization-what-sid-accuracy-does-not-tell-you)
for that arc; it is the strongest finding in this project.

**What it's for.** This is a lightweight **triage signal, not proof that an
image is synthetic**. A platform could run it over uploads after the usual
compression and resizing, and feed the probability into moderation alongside
provenance and other signals rather than treating it as a verdict. Keeping
EfficientNet-B0 instead of a larger backbone is what makes that affordable at
scale — **1.14 ms per image** — and our false-positive analysis (clean-set
FPR rises from 1.97% to 2.69% with robustness training) is why the score
should never stand alone as an accusation against a real photograph.

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
  omitted, `predict.py` **exits with an error** rather than guessing; see
  `--allow-random` below
- `--allow-random` (optional) — run without a checkpoint, emitting **random**
  `pred` values. Required as an explicit opt-in because the output file is
  valid JSON in the exact submission format, so nothing in the artifact
  distinguishes random numbers from real predictions. Use it to smoke-test the
  pipeline before a checkpoint exists; never to produce results.
- `--output` (required) — path to write the output JSON to
- `--batch_size` (optional, default `32`) — images per inference batch
- `--architecture` (optional) — which backbone to build. Listed here are
  the submission-supported backbones, `efficientnet_b0` and `convnext_tiny`;
  [src/models.py](src/models.py) also builds `dinov2_vits14`, which we
  evaluated and dropped, and which pulls its weights from `torch.hub` at
  build time rather than from `requirements.txt`. **Normally unnecessary:**
  checkpoints written by
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
  isn't instant, so request it early. The published dataset is ~300K images;
  **we did not use all of it.** After JPEG normalization, de-duplication and
  50/50 class balancing our binary manifest has a **5,099-image validation
  split** — the split every SID number in this README is measured on. With
  the 70/15/15 split in step 4 that implies a total on the order of 34,000
  images and a test split of similar size to validation; the exact train and
  test counts are not recorded in any committed file, so treat the 34,000 as
  derived rather than measured.
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

#### Known gap: the mixed SID+CIFAKE manifest is not reproducible from this repo

This matters more than the TODO above suggests, because the **mixed
SID+CIFAKE model is the checkpoint we recommend submitting** — and the
manifest behind it was built notebook-side, not by anything committed here.
What we can do is record exactly what was done, so the run can be recreated
by hand:

- The SID and CIFAKE manifests were combined with `pandas.concat` in a
  Kaggle notebook.
- The `path` and `source` fields were normalized so rows from the two
  datasets were addressable in one frame.
- The training set was balanced to **11,888 images per dataset-label group**
  — SID real, SID fake, CIFAKE real, CIFAKE fake — with **seed 42**.
- The validation set was balanced the same way.

No tooling was added to the repository for any of this. So the numbers are
recorded, but reproducing them means rebuilding the merge by hand and
trusting that you matched it, rather than running one command and getting
the same file.

**Future work:** a `--manifests a.csv b.csv` merge mode on
`src/build_manifest.py` that concatenates, normalizes the shared columns and
balances per dataset-label group under a given seed, would turn the above
into a single reproducible step and let the mixed model be rebuilt from the
repo alone.

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

Checkpoints are saved as a dict
(`{epoch, architecture, model_state, optimizer_state}`) for resumability.
The `architecture` field is what lets `predict.py` and `evaluate.py` rebuild
the right backbone without being told which one to use. Both accept either
that form or a bare state dict.

### 6. Evaluate against the challenge transform grid

All four evaluation modes now share one variable-length model interface. Each
run accepts two or more aligned checkpoint, ID, title, and architecture values:

- `--checkpoints` contains the model files.
- `--model-ids` contains unique lowercase IDs used in columns and filenames.
- `--model-titles` contains the human-readable names printed in every report.
- `--architectures` contains the matching architecture override for each
  checkpoint. Use `auto` only when the checkpoint records its architecture or
  is an EfficientNet-B0 checkpoint.

The four lists must contain the same number of values. Every model sees the
same manifest rows and transformed pixels, checkpoint hashes are recorded, and
all `n(n-1)/2` pairwise comparisons are generated. For example, four models
produce six pairwise comparisons. The older A/B and A/B/C flags remain
available so saved notebooks continue to run, but new runs should use the
plural interface.

First reproduce the clean validation results:

```bash
python evaluate.py \
  --mode clean \
  --data-root <sid_dataset_root> \
  --manifest <sid_manifest.csv> \
  --checkpoints <experiment_sid_a_clean_best.pt> <experiment_sid_b_robust_best.pt> \
  --model-ids sid_a_clean sid_b_robust \
  --model-titles "SID A — EfficientNet clean" "SID B — EfficientNet robust" \
  --architectures efficientnet_b0 efficientnet_b0 \
  --output-dir results/sid_a_vs_sid_b_clean \
  --split val \
  --device auto
```

After the clean integration check passes, run the basic single-transform grid:

```bash
python evaluate.py \
  --mode robustness \
  --data-root <sid_dataset_root> \
  --manifest <sid_manifest.csv> \
  --checkpoints <experiment_sid_a_clean_best.pt> <experiment_sid_b_robust_best.pt> \
  --model-ids sid_a_clean sid_b_robust \
  --model-titles "SID A — EfficientNet clean" "SID B — EfficientNet robust" \
  --architectures efficientnet_b0 efficientnet_b0 \
  --output-dir results/sid_a_vs_sid_b_basic \
  --split val \
  --device auto \
  --seed 42
```

To compare the three robustness-trained SID architectures, append each aligned
value. This pattern works identically in clean, basic, fixed, and random mode:

```bash
python evaluate.py \
  --mode robustness \
  --data-root <sid_dataset_root> \
  --manifest <sid_manifest.csv> \
  --checkpoints <experiment_sid_b_robust_best.pt> <experiment_sid_convnext_robust_best.pt> <experiment_sid_dinov2_robust_best.pt> \
  --model-ids sid_b_robust sid_convnext_robust sid_dino_robust \
  --model-titles "SID B — EfficientNet robust" "SID ConvNeXt — robust" "SID DINOv2 — robust" \
  --architectures efficientnet_b0 convnext_tiny dinov2_vits14 \
  --output-dir results/sid_robust_architectures_basic \
  --split val \
  --device auto \
  --seed 42
```

Keep each experiment in a separate output directory. Adding a model increases
inference work, but it does not require a new evaluator or code change.

The robustness runner uses:

- `exact_transform(img, name, param)` in
  [src/augmentations.py](src/augmentations.py) to apply each named transform.
  Names match the six challenge dimensions: `jpeg`, `blur`, `resize`, `noise`,
  `colour`, `crop`. Gaussian noise is seeded per image so every model receives
  the same noisy pixels and later runs are reproducible.
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
- `robustness_comparison.csv` — every model pair, IDs and titles, full metrics,
  candidate-minus-reference differences, clean drops, and winners for every
  condition.
- `robustness_config.json` — split, seed, preprocessing, condition grid, and
  checkpoint hashes used for the run.

### 6.1 Fixed chained robustness

`evaluate_fixed_robustness_abc.py` accepts the same variable list of two or
more models. The `_abc` filename is retained only because existing notebooks
import it; the evaluator is no longer restricted to Models A, B, and C.
Reports use the supplied IDs and titles, so a SID A/SID B comparison is never
presented as a generic A/B slot comparison.

The default `all-fixed` condition set runs clean, all 15 official single
conditions, and these five deterministic chained conditions:

| Condition title | Ordered transforms |
|---|---|
| Fixed resize-half re-encode | Resize 0.50× → JPEG quality 70 |
| Fixed resize-quarter re-encode | Resize 0.25× → JPEG quality 50 |
| Fixed crop/colour/re-encode | Crop 0.80 → Colour +0.20 → JPEG quality 50 |
| Fixed screenshot resample | Resize 0.50× → Blur 0.50 → JPEG quality 70 |
| Fixed repeated JPEG | JPEG quality 90 → JPEG 70 → JPEG 50 |

Every condition starts again from the original image. All supplied models see
the same manifest rows, transform order, transform parameters, threshold, and
seed. This runner performs no random condition selection.

```bash
python evaluate_fixed_robustness_abc.py \
  --data-root <sid_dataset_root> \
  --manifest <sid_manifest.csv> \
  --checkpoints <experiment_sid_a_clean_best.pt> <experiment_sid_b_robust_best.pt> \
  --model-ids sid_a_clean sid_b_robust \
  --model-titles "SID A — EfficientNet clean" "SID B — EfficientNet robust" \
  --architectures efficientnet_b0 efficientnet_b0 \
  --output-dir results/sid_a_vs_sid_b_fixed \
  --split val \
  --condition-set all-fixed \
  --device auto \
  --seed 42
```

Use `--condition-set fixed-chains-only` to run just clean plus the five chains.
New runs write one clearly named file per pair, such as
`fixed_robustness__sid_a_clean_vs_sid_b_robust.csv`, plus combined summary,
full-metric, per-image prediction, and configuration files containing the full
model-ID token. Supplying three, four, or five models automatically adds all
pair files and expands the combined summary.

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

`evaluate_random_robustness_abc.py` uses the same variable model interface.
The `_abc` filename remains for notebook compatibility. For each image and
trial it selects three distinct transform types, randomises their order, and
samples each parameter from the official challenge grid. The assignment is
derived from `dataset_id + image_path + trial_seed`, so every supplied model
receives identical damaged pixels and any result can be recreated.

The default five trials use seeds `42 43 44 45 46`. CIFAKE and SID_Set should
be run separately with different `--dataset-id` and output directories rather
than mixing their rows into one accuracy figure.

```bash
python evaluate_random_robustness_abc.py \
  --dataset-id SID_SET \
  --data-root <sid_dataset_root> \
  --manifest <sid_manifest.csv> \
  --checkpoints <experiment_sid_a_clean_best.pt> <experiment_sid_b_robust_best.pt> \
  --model-ids sid_a_clean sid_b_robust \
  --model-titles "SID A — EfficientNet clean" "SID B — EfficientNet robust" \
  --architectures efficientnet_b0 efficientnet_b0 \
  --output-dir results/sid_a_vs_sid_b_random_standard_3 \
  --split val \
  --trial-seeds 42 43 44 45 46 \
  --device auto
```

The primary result is pooled accuracy across all random trials, accompanied
by the mean and standard deviation across seeds and the drop from clean. The
runner also writes every pairwise comparison, ordered-pattern and
transform-inclusion breakdowns, every per-image chain assignment, all
per-image predictions, false positives and false negatives, clean-to-damaged
prediction changes, checkpoint hashes, and the complete random policy.

To run the three robustness-trained SID architectures, use:

```bash
python evaluate_random_robustness_abc.py \
  --dataset-id SID_SET \
  --data-root <sid_dataset_root> \
  --manifest <sid_manifest.csv> \
  --checkpoints <experiment_sid_b_robust_best.pt> <experiment_sid_convnext_robust_best.pt> <experiment_sid_dinov2_robust_best.pt> \
  --model-ids sid_b_robust sid_convnext_robust sid_dino_robust \
  --model-titles "SID B — EfficientNet robust" "SID ConvNeXt — robust" "SID DINOv2 — robust" \
  --architectures efficientnet_b0 convnext_tiny dinov2_vits14 \
  --output-dir results/sid_robust_architectures_random_standard_3 \
  --split val \
  --trial-seeds 42 43 44 45 46 \
  --device auto
```

Model IDs must be unique lowercase identifiers containing letters, numbers,
or underscores. They are used in column names and filenames; titles are the
human-readable descriptions shown in the results. Add or remove aligned list
items to evaluate two, three, four, five, or more models without changing the
Python source.

### Getting the checkpoints

Trained checkpoints are **not in this repository** — `*.pt` is gitignored
because the files are far too large for git. They are hosted on Google Drive.

**The one you want is `effnet_b0_sid_cifake_experiment_b_best.pt`**, the mixed
SID+CIFAKE model, which is what step 7 below and the submission both use:

<https://drive.google.com/file/d/1bz3KfWIPr422c7rGM9hYH3zs6wSr7pc7/view>

After downloading, verify the file is intact:

```bash
# Mac/Linux
shasum -a 256 effnet_b0_sid_cifake_experiment_b_best.pt
```

```powershell
# Windows (PowerShell)
Get-FileHash effnet_b0_sid_cifake_experiment_b_best.pt -Algorithm SHA256
```

Expected SHA-256:

```
9159a9d4ceb7fccd3eee24c3ddf9600c79c8abf1444cb0646b93ac66fd3b5c44
```

That hash is recorded in
[results/sid_mixed_model/robustness_config.json](results/sid_mixed_model/robustness_config.json),
alongside the run that produced every SID number in this README — so the file
you download is provably the file those results came from.

> **Do not skip this step.** Without `--checkpoint`, `predict.py` refuses to
> run and exits with an error telling you to download one. It *can* emit
> random values, but only if you explicitly pass `--allow-random`, because
> that output is well-formed JSON in the submission format and nothing in the
> file itself would reveal the numbers are meaningless.

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

**The checkpoint we recommend submitting is
`effnet_b0_sid_cifake_experiment_b_best.pt`** (EfficientNet-B0, SHA-256
`9159a9d4ceb7fccd…`, recorded in
[results/sid_mixed_model/robustness_config.json](results/sid_mixed_model/robustness_config.json)):

```bash
python predict.py --input_dir <path_to_test_images> --checkpoint effnet_b0_sid_cifake_experiment_b_best.pt --output results/preds.json
```

This is the SID+CIFAKE mixed model. It is not the highest-scoring checkpoint
on any single dataset — the SID-only model beats it by 0.47pp on SID — but
it is the only one that works on *both* datasets instead of collapsing to
chance on the one it was not trained on (99.31% on SID and 97.19% on CIFAKE,
against the SID-only model's 99.78% and 49.41%). A submission is scored on
images whose generator we do not know in advance, so the model that holds up
across sources is the right trade. The reasoning is in
[Synthesis](#synthesis-the-problem-is-the-training-distribution).

## Results tables

> **Scope note:** the results below come from two datasets and answer
> different questions. **SID_Set** (high-resolution, FLUX generator) carries
> the architecture comparison and the clean-vs-robust baseline. **CIFAKE**
> (32×32, Stable Diffusion 1.4) carries the A→B→C ablation and the
> transformation-flip result — still the clearest demonstration of what
> augmentation buys. Read both alongside the **cross-domain** result: SID
> models score near chance on CIFAKE, so high in-dataset accuracy does not
> transfer. Training on both sources fixes it: the mixed model reaches 97.19%
> on CIFAKE for a 0.47pp cost on SID. The WildFake benchmark remains held out
> and unrun, so generalization to a *third* generator is still untested.

### SID_Set results — architecture comparison

Three robustness-trained models evaluated on the **SID validation split**:
5,099 images per condition, 16 conditions (clean plus 15 isolated damages),
decision threshold 0.5, seed 42. Checkpoint SHA-256 hashes for all three are
recorded in [results/sid/run_config.json](results/sid/run_config.json), and
the full per-condition metrics — confusion matrices, precision, recall, F1,
AUPRC, Brier score — are in
[results/sid/basic_robustness_metrics.csv](results/sid/basic_robustness_metrics.csv).

#### Model comparison

| Model | `model_id` | Clean acc. | Mean damaged acc. | Worst damaged acc. | Mean damaged AUROC | Latency |
|---|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | `effnet_sid_b` | 99.78% | 99.73% | 99.43% (noise 0.1) | 0.99989 | 1.14 ms |
| ConvNeXt-Tiny | `convnext_sid_b` | 99.80% | 99.78% | 99.45% (noise 0.1) | 0.99996 | 4.12 ms |
| DINOv2 ViT-S/14 | `dino_sid_b` | 97.35% | 96.26% | 85.23% (crop 0.8) | 0.99373 | 4.34 ms |

> ⚠️ **Do not read 99.78% in isolation.** These numbers describe SID images
> only. The same models score **near chance on CIFAKE** — see
> [Cross-domain generalization](#cross-domain-generalization-what-sid-accuracy-does-not-tell-you)
> below. Robustness to transforms and generalization to an unseen generator
> are different problems, and this table measures only the first.

"Mean damaged" and "worst damaged" are computed across the 15 damaged
conditions, excluding clean; the worst condition for each model is named in
brackets. Latency is mean milliseconds per image as measured during this
evaluation run. Source:
[results/sid/model_summary.csv](results/sid/model_summary.csv).

#### Per-condition accuracy

| Condition | EfficientNet-B0 | ConvNeXt-Tiny | DINOv2 ViT-S/14 |
|---|---:|---:|---:|
| clean | 99.78% | 99.80% | 97.35% |
| jpeg 90 | 99.78% | 99.80% | 97.35% |
| jpeg 70 | 99.75% | 99.84% | 97.53% |
| jpeg 50 | 99.75% | 99.86% | 97.35% |
| jpeg 30 | 99.75% | 99.84% | 97.49% |
| blur 0.5 | 99.80% | 99.80% | 97.39% |
| blur 1 | 99.82% | 99.86% | 97.51% |
| blur 2 | 99.76% | 99.86% | 97.22% |
| resize 0.5 | 99.82% | 99.86% | 97.41% |
| resize 0.25 | 99.76% | 99.86% | 97.49% |
| noise 0.02 | 99.84% | 99.75% | 97.43% |
| noise 0.05 | 99.63% | 99.55% | 97.49% |
| noise 0.1 | 99.43% | 99.45% | 97.43% |
| colour -0.2 | 99.71% | 99.76% | 97.43% |
| colour 0.2 | 99.59% | 99.76% | 92.14% |
| crop 0.8 | 99.78% | 99.78% | 85.23% |

Source: [results/sid/accuracy_table.csv](results/sid/accuracy_table.csv);
the matching per-condition AUROC table is
[results/sid/auroc_table.csv](results/sid/auroc_table.csv).

One artifact worth naming: `jpeg 90` reproduces the clean confusion matrix
exactly for all three models. The manifest already stores every image as
JPEG quality 95, so re-encoding at quality 90 moves no image across the
threshold. AUROC does shift very slightly, so the transform is genuinely
being applied — it simply is not damage at this severity.

#### Architecture finding: EfficientNet-B0 is the better trade-off

ConvNeXt-Tiny is the numerically strongest model, but the margin is **0.05
percentage points** of mean damaged accuracy (99.78% against EfficientNet's
99.73%) and it costs **3.6× the inference time** — 4.12 ms versus 1.14 ms
per image.

The paired analysis in
[results/sid/effnet_vs_convnext_paired_analysis.csv](results/sid/effnet_vs_convnext_paired_analysis.csv)
shows that margin is not statistically significant. ConvNeXt wins 12 of the
16 conditions, EfficientNet wins 2, and 2 are tied; ConvNeXt is uniquely
correct on 170 images against EfficientNet's 134. But **no individual
condition reaches McNemar exact p < 0.05** — the smallest p-value across all
16 is 0.093, at colour +0.2. On 5,099 images per condition, a gap this small
is indistinguishable from noise.

We therefore keep **EfficientNet-B0** as the working architecture: it is
statistically indistinguishable from ConvNeXt-Tiny on this evaluation while
running 3.6× faster. ConvNeXt-Tiny is retained as a candidate for a later
cross-domain comparison rather than discarded — an architecture difference
that does not show up here might still show up on unseen generators.

**DINOv2 ViT-S/14 is dropped.** It reaches 96.26% mean damaged accuracy
against roughly 99.7% for the other two, and it fails specifically where
they do not: 92.14% under colour +0.2 and **85.23% under crop 0.8**, a
12-point fall from its own clean accuracy. It is also the slowest of the
three at 4.34 ms per image. Full reasoning is in
[results/sid/architecture_selection_notes.txt](results/sid/architecture_selection_notes.txt).

#### The SID A→B comparison: augmentation buys noise resistance

A clean-trained SID baseline exists — `experiment_sid_a_clean_best.pt`,
SHA-256 `fa2f40fd…` — and has been evaluated against the robustness-trained
model on the SID validation split, 5,099 images per condition, threshold
0.5, seed 42.

| Condition | SID-A (clean-trained) | SID-B (robustness-trained) | Difference |
|---|---:|---:|---:|
| clean | 99.86% | 99.78% | -0.08pp |
| jpeg 90 | 99.88% | 99.78% | -0.10pp |
| jpeg 70 | 99.80% | 99.75% | -0.06pp |
| jpeg 50 | 98.73% | 99.75% | +1.02pp |
| jpeg 30 | 96.35% | 99.75% | +3.39pp |
| blur 0.5 | 99.90% | 99.80% | -0.10pp |
| blur 1.0 | 99.90% | 99.82% | -0.08pp |
| blur 2.0 | 99.69% | 99.76% | +0.08pp |
| resize 0.5 | 99.88% | 99.82% | -0.06pp |
| resize 0.25 | 99.71% | 99.76% | +0.06pp |
| noise 0.02 | 99.04% | 99.84% | +0.80pp |
| noise 0.05 | 79.15% | 99.63% | +20.47pp |
| noise 0.10 | 56.85% | 99.43% | +42.58pp |
| colour -0.2 | 99.65% | 99.71% | +0.06pp |
| colour +0.2 | 99.29% | 99.59% | +0.29pp |
| crop 0.8 | 99.67% | 99.78% | +0.12pp |

Source:
[results/sid_a_vs_b/robustness_metrics.csv](results/sid_a_vs_b/robustness_metrics.csv);
aggregates in
[results/sid_a_vs_b/sid_a_vs_sid_b_basic_headline.csv](results/sid_a_vs_b/sid_a_vs_sid_b_basic_headline.csv).

**The average hides the result.** Across all 15 damaged conditions SID-A
averages 95.17% against SID-B's 99.73% — a gap that looks moderate. But on
13 of the 16 conditions the two models are within ±1pp of each other, and
the entire difference comes from three conditions:

- **noise σ0.10 — 56.85% vs 99.43%**, a 42.6pp gap. SID-A's recall falls to
  **12.11%**: it misses seven of every eight AI images.
- **noise σ0.05 — 79.15% vs 99.63%**, 20.5pp.
- **JPEG 30 — 96.35% vs 99.75%**, 3.4pp.

Everywhere else — blur, resize, colour, crop, mild JPEG — the clean-trained
model is fine, and on six conditions it is fractionally *ahead*. Augmentation
is not buying broad robustness on SID. It is buying resistance to additive
noise, which is the one degradation that clean SID training does not
incidentally cover.

One further detail worth stating: at noise σ0.10 SID-A's AUROC is still
**0.9972**. Its *ranking* is nearly intact — the scores are all there, the
decision boundary has simply moved. That is a calibration failure, not a
representation failure, and unlike the cross-domain collapse below it could
in principle be fixed by moving the threshold.

#### Chained damage: where the gap is unambiguous

Under randomly chained damage — 3 transforms per image, 5 trials over the
same 5,099 images, 25,495 pooled predictions — the two models separate
cleanly:

| | SID-A (clean-trained) | SID-B (robustness-trained) |
|---|---:|---:|
| Clean accuracy | 99.86% | 99.78% |
| Pooled chained accuracy | **82.997%** | **99.078%** |
| Trial std. dev. | 0.31pp | 0.07pp |
| Pooled AUROC | 0.9873 | 0.9998 |
| False-negative rate | **34.60%** | **1.72%** |
| Clean-correct retention | 83.04% | 99.14% |

"Clean-correct retention" is the share of images the model got right clean
that it still gets right after damage — the chained analogue of the
transformation-flip metric used on CIFAKE. SID-A loses **17%** of its
correct answers; SID-B loses 0.9%.

Both models fail asymmetrically, calling AI images real: SID-A's
false-negative rate is 34.60% against a false-positive rate of 0.04%. Under
stacked damage the clean-trained model misses roughly a third of all AI
images.

Across five trials the standard deviation is 0.31pp (A) and 0.07pp (B), so
the 16-point gap is far outside trial-to-trial variation. Source:
[results/sid_a_vs_b_chained/](results/sid_a_vs_b_chained/) — headline,
overall summary, per-trial metrics, and chain-pattern and
transform-inclusion breakdowns.

**Single transforms understate what augmentation is worth; chained damage
shows it.** Real redistribution stacks damage — an image is resized, re-
encoded, screenshotted, re-encoded again — and that is the regime where the
augmented model holds and the clean one does not.

### Cross-domain generalization: what SID accuracy does not tell you

Both EfficientNet-B0 SID models — the clean baseline `sid_a_clean` and the
robustness-trained `sid_b_robust` — were evaluated on the **CIFAKE
validation split**: 14,724 images (7,261 real, 7,463 AI), the same
16-condition grid, threshold 0.5, seed 42. Checkpoint hashes are in
[results/cross_domain_cifake/robustness_config.json](results/cross_domain_cifake/robustness_config.json).

**They collapse to chance.**

| | SID-A (clean-trained) | SID-B (robustness-trained) |
|---|---:|---:|
| Clean CIFAKE accuracy | 51.18% | 49.41% |
| Clean recall on AI images | 3.91% (292 / 7,463) | 0.21% (16 / 7,463) |
| Accuracy range over 16 conditions | 49.31% – 51.18% | 49.31% – 50.29% |
| AUROC range | 0.5079 – 0.6406 | 0.5015 – 0.6505 |

Predicting "real" for every image scores **49.31%** on this split. Both
models sit within two points of that floor, and the failure is almost
entirely false negatives: they call nearly everything real. Under noise 0.05
and noise 0.10 **both detect zero AI images** — not a low count, zero out of
7,463.

This is not a threshold artifact. AUROC starts around 0.635 on clean images
and falls to **0.5079** (SID-A) and **0.5015** (SID-B) under blur σ2.0 —
literally random ranking. A sweep over the clean predictions confirms it:
the best accuracy any threshold can buy is **59.6%** for SID-A (at ≈0.04)
and **59.9%** for SID-B (at ≈0.001). Recalibration cannot rescue a model
whose scores carry almost no signal.

**Transform robustness and cross-generator generalization are separate
problems.** A detector can be 99.8% accurate under every transform in the
grid on its own dataset and still be near-useless on images from a generator
it never saw. Nothing in the SID tables above predicts this result, and
nothing in our CIFAKE robustness work does either. For the challenge this is
the more consequential finding: a deployed detector meets generators that
were not in its training set, and our evidence says robustness training does
not prepare it for that. Robustness augmentation buys resilience *within* a
distribution, not across distributions.

Full per-condition metrics are in
[results/cross_domain_cifake/robustness_metrics.csv](results/cross_domain_cifake/robustness_metrics.csv);
the paired comparison is in
[results/cross_domain_cifake/robustness_comparison.csv](results/cross_domain_cifake/robustness_comparison.csv).
The 167MB per-image prediction dump is deliberately not committed (see
`.gitignore`).

#### A second architecture fails identically

The collapse is not an artifact of EfficientNet-B0. We ran the same CIFAKE
evaluation on **ConvNeXt-Tiny** (`sid_convnext_robust`, SHA-256
`758444e5…`) — a different backbone family, 27.8M parameters against
EfficientNet's 4.0M, and the model that scored *best of all three* on SID at
99.80% clean. Same 14,724 CIFAKE validation images, same 16 conditions,
threshold 0.5, seed 42.

| | EfficientNet-B0 | ConvNeXt-Tiny |
|---|---:|---:|
| SID clean accuracy (for reference) | 99.78% | 99.80% |
| CIFAKE clean accuracy | 49.41% | **49.38%** |
| CIFAKE clean recall on AI images | 0.21% (16 / 7,463) | **0.13% (10 / 7,463)** |
| Best recall across all 16 conditions | 7.84% | **0.46%** |
| Accuracy range over 16 conditions | 49.31% – 50.29% | **49.29% – 49.40%** |
| AUROC range | 0.5015 – 0.6505 | 0.5136 – 0.6407 |
| Conditions with zero AI detections | 2 (noise 0.05, 0.10) | **3 (noise 0.02, 0.05, 0.10)** |

ConvNeXt-Tiny is, if anything, *further* gone. It never exceeds 0.46% recall
under any condition, its accuracy never leaves a 0.10-point band around the
49.31% all-real floor, and it detects nothing at all under three noise
settings rather than two. Tripling the parameter count and changing backbone
family buys nothing.

**The cause is the training distribution, not model capacity.** Two
architectures with different inductive biases, trained on the same data,
fail in the same direction, to the same degree, with the same asymmetry —
both predict "real" almost universally. That rules out architecture as the
explanation and points at what the models were trained on. Scaling up or
swapping backbones is not a route out of this; changing the training data
is, which is what the mixed model below tests.

Source:
[results/cross_domain_convnext/robustness_metrics.csv](results/cross_domain_convnext/robustness_metrics.csv).
DINOv2 ViT-S/14 has not been evaluated cross-domain.

#### Mixed training closes the gap on CIFAKE

Training on SID **and** CIFAKE together recovers almost all of the loss. The
mixed model (`mixed_sid_cifake_b`, SHA-256 `9159a9d4…`) was evaluated on the
same CIFAKE validation split against the SID-only model it replaces:

| Condition | SID-only B | Mixed SID+CIFAKE B | Difference |
|---|---:|---:|---:|
| clean | 49.41% | 97.19% | +47.78pp |
| jpeg 90 | 49.40% | 97.11% | +47.70pp |
| jpeg 70 | 49.40% | 97.15% | +47.75pp |
| jpeg 50 | 49.36% | 95.81% | +46.45pp |
| jpeg 30 | 49.39% | 95.12% | +45.73pp |
| blur 0.5 | 49.37% | 96.00% | +46.63pp |
| blur 1.0 | 49.93% | 93.83% | +43.89pp |
| blur 2.0 | 50.04% | 87.75% | +37.71pp |
| resize 0.5 | 49.65% | 94.12% | +44.46pp |
| resize 0.25 | 50.29% | 88.28% | +38.00pp |
| noise 0.02 | 49.33% | 96.75% | +47.42pp |
| noise 0.05 | 49.31% | 95.63% | +46.31pp |
| noise 0.10 | 49.31% | 92.85% | +43.53pp |
| colour -0.2 | 49.33% | 95.68% | +46.35pp |
| colour +0.2 | 49.50% | 94.43% | +44.93pp |
| crop 0.8 | 49.37% | 94.88% | +45.51pp |

Clean accuracy goes from 49.41% to **97.19%**, mean damaged accuracy from
49.53% to **94.36%**, and AUROC from a 0.50–0.65 band to **0.958–0.996**.
Recall on AI images goes from 0.21% to 98.08%. The worst condition is blur
σ2.0 at 87.75%, still 38pp above the model it replaces. Source:
[results/cifake_mixed_model/robustness_metrics.csv](results/cifake_mixed_model/robustness_metrics.csv).

#### Mixed model under chained damage — and a column that misleads

The same two models under randomly chained damage on CIFAKE: 3 transforms
per image, 5 trials (seeds 42–46) over 14,724 images, 73,620 pooled
predictions.

| | Old SID-B (SID-only) | Mixed SID+CIFAKE B |
|---|---:|---:|
| Clean accuracy | 49.41% | 97.19% |
| Pooled chained accuracy | 49.66% | **84.33%** |
| Trial std. dev. | 0.12pp | 0.29pp |
| Pooled AUROC | 0.515 | **0.942** |
| False-negative rate | **97.74%** | 25.79% |
| False-positive rate | 1.61% | 5.27% |
| Recall on AI images | 2.26% | 74.21% |
| Accuracy drop from clean | **−0.26pp** | +12.86pp |
| Clean-correct retention | **98.22%** | 85.41% |

**Two columns in that table look like wins for the SID-only model and are
not.** Its clean-correct retention (98.22%) is *higher* than the mixed
model's (85.41%), and its accuracy under damage is marginally *higher* than
its clean accuracy — a negative drop. Neither is robustness.

Old SID-B predicts "real" for essentially every image: false-positive rate
1.61%, recall on AI images 2.26%, false-negative rate **97.74%**. Damage
cannot change an answer that does not depend on the input. It keeps getting
the real images right, keeps missing the AI images, and both behaviours
survive any transform you apply — so retention is near-perfect and the drop
is ~zero. **A constant classifier is trivially stable.** Retention and drop
measure *consistency*, not correctness, and they are only meaningful for a
model that is actually discriminating. Read them next to AUROC (0.515,
barely above random) and recall (2.26%), which are not flattering.

The mixed model's numbers are the honest kind. It loses 12.86 points from
clean to chained and retains 85.41% of its clean-correct answers, because it
is making real decisions that damage can genuinely disrupt. Its pooled AUROC
of 0.942 against 0.515 is the comparison that matters.

Source:
[results/cifake_mixed_chained/](results/cifake_mixed_chained/) — headline,
overall summary, per-trial metrics, and chain-pattern and
transform-inclusion breakdowns.

#### What mixing costs on SID: about half a point

The remaining question — whether adding CIFAKE damaged SID performance — has
now been measured. Both models on the SID validation split, 5,099 images per
condition, same 16 conditions, threshold 0.5, seed 42.

| Condition | SID-only B | Mixed SID+CIFAKE B | Cost |
|---|---:|---:|---:|
| clean | 99.78% | 99.31% | -0.47pp |
| jpeg 90 | 99.78% | 99.35% | -0.43pp |
| jpeg 70 | 99.75% | 99.43% | -0.31pp |
| jpeg 50 | 99.75% | 99.41% | -0.33pp |
| jpeg 30 | 99.75% | 99.49% | -0.25pp |
| blur 0.5 | 99.80% | 99.35% | -0.45pp |
| blur 1.0 | 99.82% | 99.39% | -0.43pp |
| blur 2.0 | 99.76% | 99.20% | -0.57pp |
| resize 0.5 | 99.82% | 99.41% | -0.41pp |
| resize 0.25 | 99.76% | 99.33% | -0.43pp |
| noise 0.02 | 99.84% | 99.29% | -0.55pp |
| noise 0.05 | 99.63% | 99.12% | -0.51pp |
| noise 0.10 | 99.43% | 98.51% | -0.92pp |
| colour -0.2 | 99.71% | 99.33% | -0.37pp |
| colour +0.2 | 99.59% | 99.24% | -0.35pp |
| crop 0.8 | 99.78% | 99.43% | -0.35pp |

**The cost is 0.47pp on clean images (99.78% → 99.31%) and 0.92pp at the
worst condition (99.43% → 98.51%, noise σ0.10).** No condition costs more
than 0.92pp; the median cost is 0.43pp. Clean AUROC is essentially unchanged
at 0.9998 against 0.9999, and mean AUROC across the damaged conditions is
0.9995 against 0.9999 — the ranking quality survives intact. Source:
[results/sid_mixed_model/robustness_metrics.csv](results/sid_mixed_model/robustness_metrics.csv).

#### Synthesis: the problem is the training distribution

Putting the three cross-domain results together:

| | SID | CIFAKE |
|---|---:|---:|
| SID-only B (EfficientNet-B0) | 99.78% | 49.41% |
| SID-only ConvNeXt-Tiny | 99.80% | 49.38% |
| **Mixed SID+CIFAKE B** | **99.31%** | **97.19%** |

Two conclusions follow, and they are the strongest results in this project.

**Backbone choice was not the dominant factor.** Two backbones from
different families — 4.0M and 27.8M parameters, convolutional and
modernized-ConvNeXt designs — trained on the same data collapse to within
0.03pp of each other on CIFAKE, both to a near-constant "real" prediction.
Tripling the parameter count changed nothing we could measure, so the
evidence points much more strongly to the training distribution than to
architecture or capacity. Two architectures is strong evidence rather than
proof: a third family might behave differently, and we did not test one.

**Training-source diversity was far more influential than scaling the
backbone in our experiments.** Adding a second training source costs
**0.47pp** on the original dataset and gains **47.78pp** on the previously
unseen one — roughly a hundred points gained for every point given up.
Neither a larger backbone nor heavier augmentation moved this failure mode
at all; a second source of images did.

The honest scope: this is one direction (SID → SID+CIFAKE), one seed, and
two datasets. We have shown that adding a source fixes performance on that
source at negligible cost to the first. We have **not** shown that a model
trained on two generators generalizes to a third — the held-out WildFake
benchmark would test exactly that, and it remains unrun. That is the natural
next experiment and the honest limit of the claim.

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

`clean_metrics_a_b.csv` also records both checkpoints' SHA-256 hashes in a
`checkpoint_hash` column, so these numbers are traceable to specific
checkpoint files. The same is true of every evaluation in this README — see
[Data provenance](#data-provenance) for the committed run configs.

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
- Run configs **are** committed, for every evaluation reported in this
  README. Each records the checkpoint SHA-256 hashes, architecture, split,
  seed, threshold and full condition grid, so every number above is
  traceable to a specific checkpoint file:
  [results/sid/run_config.json](results/sid/run_config.json),
  [results/sid_a_vs_b/](results/sid_a_vs_b/robustness_config.json),
  [results/sid_a_vs_b_chained/](results/sid_a_vs_b_chained/random_standard_3__run_config.json),
  [results/cross_domain_cifake/](results/cross_domain_cifake/robustness_config.json),
  [results/cross_domain_convnext/](results/cross_domain_convnext/robustness_config.json),
  [results/cifake_mixed_model/](results/cifake_mixed_model/robustness_config.json),
  [results/cifake_mixed_chained/](results/cifake_mixed_chained/random_standard_3__run_config.json)
  and [results/sid_mixed_model/](results/sid_mixed_model/robustness_config.json).
- Per-image error analysis **has** been done for the CIFAKE A/B comparison:
  [results/error_analysis.md](results/error_analysis.md), generated
  deterministically by [scripts/error_analysis.py](scripts/error_analysis.py)
  from the 471,168-row prediction dump. It covers false positives, false
  negatives, transformation flips, high-confidence mistakes, repeat failures
  and calibration bands. The script regenerates it from a local
  `robustness_predictions.csv`; the dump itself stays gitignored.

## Limitations and what we'd improve with more time

Every figure below comes from
[results/error_analysis.md](results/error_analysis.md) and the CSVs in
`results/`.

**Generalization to a third, unseen generator is unproven.** A model
trained on one generator fails completely on another: both SID-only models
score at chance on CIFAKE — 51.18% and 49.41% against a 49.31% all-real
floor, AUROC as low as 0.5015, zero AI images detected under noise, and no
threshold recovering more than ~60%. Robustness augmentation does not help,
and neither does changing architecture. Training on **both** sources fixes
it: the mixed model reaches 97.19% on CIFAKE for a 0.47pp cost on SID (see
[Synthesis](#synthesis-the-problem-is-the-training-distribution)).

What that does **not** establish is that a model trained on two generators
handles a third it has never seen. We have shown that adding a source fixes
that source, cheaply — one direction, one seed, two datasets. Whether source
diversity confers *general* robustness to unseen generators is exactly what
the held-out WildFake benchmark would test, and it remains unrun. Until it
is, treat any claim about a generator absent from training as unsupported,
with one demonstrated data point on how severe that failure can be.

**The augmentation effect is demonstrated on CIFAKE, and narrowly on SID.**
The A→B transformation-flip result (38,504 → 5,611) comes from CIFAKE: 32×32
images, Stable Diffusion 1.4, one validation split of 14,724 images, one
seed. The SID A→B comparison supports the same direction but more narrowly
than the average suggests: under single transforms the two models are within
±1pp on 13 of 16 conditions, and the whole gap sits in additive noise
(56.85% vs 99.43% at σ0.10). Under chained damage the separation is
unambiguous (82.997% vs 99.078%). So augmentation's value on SID is real but
concentrated — it is not broad robustness across every degradation type.

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

**That calibration result holds only in-distribution.** It was measured on
CIFAKE, where B was trained. Run a SID-trained model on CIFAKE instead and
its calibration collapses along with everything else: scores sit below the
threshold almost regardless of content, AUROC falls to 0.5015, and a
threshold sweep recovers at most ~60% accuracy — no threshold makes those
outputs useful, because the ranking itself carries almost no signal. A
`pred` value is interpretable as a probability only for images resembling
what the model was trained on.

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

**What we'd do with more time.** In priority order: run the held-out
WildFake subset (COCO val2017 + DALL·E Advanced) against the mixed model, to
find out whether training on two generators helps on a third — this is now
our largest unknown by a wide margin, and the one result that would turn the
source-diversity finding from a demonstration into a general claim; manually
inspect the 40 persistent failures to establish whether CIFAKE carries label
noise; tune the decision threshold to claw back Experiment B's
false-positive rate; and re-run the A/B/C ablation across multiple seeds so
a 0.64pp difference between B and C could actually be called significant or
not.

## Team member contributions

Note that some team members committed under more than one Git identity
(different machines, or a changed username mid-project). The repo's
`.mailmap` collapses these — `git shortlog -sn` shows the five
contributors listed below. GitHub's web contributor graph does not read
`.mailmap`, so it displays the raw identities.

**Isaac — Model Lead.** Owned the model architecture and the training
pipeline, and ran Experiments A, B and C covering clean training, robustness
augmentation and consistency training. Main contributions: `train.py`, the
model definitions in `src/models.py`, the experiment configs in `configs/`,
and the checkpoint workflow the evaluation scripts read from.

**Jerry — Integration Lead.** Repo setup and structure; the inference
pipeline (`predict.py`) and its smoke tests; the shared evaluation
preprocessing (`get_eval_transform()` in `src/data.py`) that keeps
validation, evaluation and inference from drifting apart; the error analysis
and robustness-curve
scripts in `scripts/`; and the README, reproduction steps and
`results/decisions.md`.

**Teoh Ke Yi — Data and Infrastructure Lead.** Prepared and published the
CIFAKE and SID_Set datasets the project runs on, and established the
tampered-image holdout that keeps 3-class data out of binary training and
evaluation. Owned the data pipeline end to end: `src/build_manifest.py`,
which produces the balanced, format-normalised manifest, and the
manifest-backed dataset and split handling in `src/data.py`.

**Wei Jien — Evaluation and Benchmarking Lead.** Designed and implemented the
evaluation workflow end to end: clean evaluation, binary metrics, and the
fixed and seeded-random chained robustness grids (`src/metrics.py`,
`evaluate.py`, `evaluate_fixed_robustness_abc.py`,
`evaluate_random_robustness_abc.py`). Produced the per-image predictions and
pairwise A/B/C summaries the results tables and error analysis are built
from.

**Tan Teck Heang — Augmentation and Consistency Training Lead.** Owned the
robustness augmentation pipeline and the consistency-training objective:
the challenge transform implementations in `src/augmentations.py` and the
paired clean/damaged consistency loss in `src/losses.py`. Between them these
two modules are what Experiments B and C are actually trained with, and the
transform set is reused unchanged by the evaluation grid.
