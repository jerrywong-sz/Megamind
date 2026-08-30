# Robust AI-Generated Image Detection — TikTok TechJam Track 5

**Demo video:** TODO — paste YouTube link here.

A binary detector for AI-generated images, built and measured around a single
question: does it still work after the image has been through a real upload
pipeline?

---

## 1. How the solution addresses the problem statement

A detector that only works on pristine files is not useful. Images that reach a
platform have been compressed, resized, blurred, noised, colour-shifted and
cropped first. So we measured the failure mode directly rather than reporting a
single clean-set accuracy.

Our headline measurement is **transformation flips**: images the model
classifies correctly when clean and incorrectly after a transform. Across 15
transformed conditions on 14,724 held-out CIFAKE validation images:

| | Clean baseline (A) | Robustness-augmented (B) |
|---|---:|---:|
| **Transformation flips** | **38,504** | **5,611** |
| Total misclassification events | 41,024 | 9,065 |
| High-confidence mistakes | 22,027 | 972 |
| Worst-condition accuracy | 53.67% | 91.37% |

**Augmentation removes 85.4% of the baseline's flips.** The baseline does not
degrade gracefully — it collapses. Under Gaussian noise σ=0.10 its accuracy
falls from 98.32% to **53.67%**, barely above the 50% chance rate for a balanced
binary task. Blur σ=2.0 takes it to 57.75% and resize 0.25× to 60.74%. The
augmented model stays at 94.91%, 91.77% and 91.37% on those same conditions, and
never drops below 91.37% on any of the 16 conditions.

The gap in *confidence* is larger than the gap in accuracy. Counting only errors
where the model was emphatic and wrong (an AI image scored ≤0.05, or a real
image scored ≥0.95), the baseline makes 22,027 and the augmented model 972 — a
23× difference. Calibration follows: in the augmented model's ≥0.99 score band,
99.0% of images really are AI-generated; in the baseline's >0.9 band, only 85.8%
are. The augmented model's output is usable as a probability. The baseline's is
not.

### Why the comparison is meaningful

We ran a three-arm ablation:

- **A — clean baseline.** Conventional preprocessing only: resize 256, centre
  crop 224, random horizontal flip, ImageNet normalization. A competent
  conventional classifier, deliberately not a strawman.
- **B — robustness augmentation.** Identical to A, plus randomized
  challenge-relevant degradation (JPEG, blur, resize, noise, colour jitter,
  crop) applied to 0–3 transforms per training image.
- **C — consistency training.** Identical to B, plus a penalty that pushes the
  model toward the same prediction on a clean and a damaged view of the same
  image.

**Backbone, seed (42), epochs (5), batch size (32), learning rate (1e-4), weight
decay (1e-4), mixed precision, and the train/val split are held identical across
all three.** Only the training method varies, so a difference between arms is
attributable to the method rather than to a hyperparameter confound. Evaluation
is equally controlled: both models see the same images in the same order under
each condition, Gaussian noise is seeded per image so both receive identical
noisy pixels, and checkpoint SHA-256 hashes are recorded with every run.

Training and inference share one preprocessing function
(`get_eval_transform()`), imported by both paths, so the two cannot silently
drift apart — a mismatch there produces no error, only quietly worse accuracy.

### The trade we are not hiding

Robustness is not free. On clean images the augmented model has a **higher
false-positive rate: 2.69% versus the baseline's 1.97%** (195 versus 143 false
positives out of 7,261 real images), against a slightly lower false-negative
rate (1.27% vs 1.41%). Augmentation shifts the decision boundary toward calling
images AI-generated. The trade is strongly favourable overall, but it moves in
the direction the challenge warns about — false accusations against genuine
photographs — and we did not tune the threshold to compensate.

## A negative result: consistency training did not help

Experiment C is reported as a **negative result**.

Across the 16 single-transform conditions, C beat B in **9** and lost in **7**,
with a **mean difference of −0.09 percentage points** and a maximum absolute
difference of 0.64pp. On the five chained conditions it lost in 4 of 5. On clean
images C is 98.21% against B's 98.03%.

Differences that small, on a single validation split with one seed, are
indistinguishable from noise. We cannot claim the consistency loss did anything.
Essentially all of the robustness gain comes from augmentation (A→B); adding the
consistency penalty on top (B→C) added nothing we can measure. Establishing
whether a 0.64pp difference is real would need multiple seeds, which we did not
have time to run.

## 2. Development tools used

- **VS Code** — primary editor
- **Google Colab** — GPU training (mixed precision on CUDA)
- **Kaggle** — CIFAKE dataset access
- **Git / GitHub** — version control, pull-request review across five people
- **Git Bash on Windows** — local shell
- **pytest** — automated tests (54 tests covering augmentations, metrics,
  losses, the data pipeline, the evaluation runners, and the inference CLI)
- **Claude Code** — development assistance

## 3. Models used

**EfficientNet-B0** from `torchvision.models`, initialized with ImageNet
pretrained weights, with the 1000-class classifier replaced by a single-logit
`nn.Linear(1280, 1)` head for binary classification. Sigmoid is applied outside
the model; training uses `BCEWithLogitsLoss`.

- **Measured parameter count: 4,008,829 (~4.0M).** This is **0.20% of the
  challenge's 2B parameter limit.** (Note: EfficientNet-B0 is commonly quoted at
  5.3M; the single-logit head removes ~1.28M parameters from the original
  classifier, hence 4.0M here.)
- **No external APIs.** All inference runs locally from a checkpoint file.

One backbone is shared across all three experiments so the ablation isolates
training method rather than architecture.

## 4. Libraries and frameworks

From `requirements.txt`, with the versions used:

| Library | Version | Role |
|---|---|---|
| torch | 2.13.0 | model, training loop, AMP |
| torchvision | 0.28.0 | EfficientNet-B0, transforms |
| pillow | 12.3.0 | image I/O, degradation transforms |
| numpy | 2.5.2 | numerics |
| scikit-learn | 1.9.0 | metrics (AUROC, AUPRC, F1, confusion matrix) |
| pandas | 3.0.5 | manifest and results handling |
| matplotlib | 3.11.1 | robustness severity curves |
| tqdm | 4.70.0 | progress reporting |
| pyyaml | 6.0.3 | experiment configs |
| pytest | 9.1.1 | test suite |

No other dependencies. The robustness transforms (JPEG re-encode, Gaussian blur,
resize-and-upscale, Gaussian noise, colour jitter, centre crop) are implemented
directly on Pillow and torch rather than pulled from an augmentation library, so
the exact challenge parameter values are reproduced rather than approximated.

## 5. Datasets

- **CIFAKE** (Kaggle) — 32×32 images, Stable Diffusion 1.4 as the generator.
  **All results reported above come from CIFAKE**: 68,712 training images and
  14,724 validation images after 50/50 class balancing and a 70/15/15 split.
- **SID_Set** (Hugging Face, CC BY 4.0) — ~300K images generated with FLUX,
  including a tampered class (real photographs with an AI-generated region
  inserted). **Training on SID_Set is in progress and is not reflected in any
  number in this submission.** Its tampered images are routed to a dedicated
  `bonus` split and are excluded from binary training and evaluation, because a
  tampered image is mostly authentic pixels and training on it as "AI" risks
  pushing the model toward false positives on genuine photographs.
- **WildFake** (COCO val2017 real images + DALL·E Advanced generated images) —
  **held out and untouched**, per the challenge rules, as a demonstration
  benchmark for cross-generator generalization. It has not been trained on and
  has not been evaluated yet.

To prevent the model learning file-format signatures rather than image content,
every image from every dataset is re-encoded to standard JPEG (quality 95, alpha
stripped) before training, and SHA-256 de-duplicated.

## Limitations

We would rather state these than have a judge find them.

1. **Everything reported is CIFAKE only** — 32×32 images, one generator (SD 1.4),
   one validation split. We have **no evidence** about high-resolution images or
   unseen generators. The robustness claim is "robust to transformations *within*
   CIFAKE", not "robust in general".
2. **Consistency training produced no measurable improvement** (above).
3. **The robustness gain costs clean-set precision** — false-positive rate rises
   from 1.97% to 2.69%.
4. **40 images are misclassified by the augmented model under all 16
   conditions.** No amount of augmentation moves them, which suggests label noise
   or intrinsic ambiguity in CIFAKE rather than a robustness failure. We did not
   open them to check.
5. **No per-source or per-generator error breakdown was possible.** The `source`
   column in our prediction logs is null in 100% of rows, and `dataset` and
   `generator` are constant, so we cannot say *which kinds* of image fail.
6. **Single seed, single split.** Small differences between arms cannot be called
   significant.
7. Model outputs are strongly bimodal — only 7.7–8.3% of predictions fall between
   0.20 and 0.80. This is expected from a single-logit network trained to
   near-zero loss, and is only a problem for the baseline, whose confidence is
   also poorly calibrated.

**With more time, in priority order:** finish training on SID_Set for
high-resolution and second-generator coverage; run the held-out WildFake
benchmark for a real cross-generator number, which is our largest unknown;
inspect the 40 persistent failures for label noise; tune the decision threshold
to claw back the false-positive rate; and re-run the A/B/C ablation across
multiple seeds.

## Reproducibility

The repository README documents the full pipeline end to end — dataset
acquisition, manifest construction, the three training commands, the evaluation
grid, and inference — with every command verified against the actual CLI.
Per-condition metrics, the error analysis, and the design-decision log are
committed under `results/`; the analysis and plotting scripts under `scripts/`
regenerate the figures deterministically from the committed CSVs.
