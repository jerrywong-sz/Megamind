# Robust AI-Generated Image Detection — TikTok TechJam Track 5

**Demo video:** TODO — paste YouTube link here.

A binary detector for AI-generated images, built and measured around two
questions: does it still work after the image has been through a real upload
pipeline, and does it still work on a generator it has never seen?

The first question has a good answer — robustness augmentation removes 85.4%
of the errors that transforms introduce. The second nearly did not. Our
best detector, 99.8% accurate on its own dataset, turned out to score **at
chance** on images from a different generator. Our evidence pointed strongly
to training-distribution shift rather than backbone choice; adding a second
training source removed the observed SID↔CIFAKE collapse for 0.47 accuracy
points, though generalization to a third unseen generator remains unproven.
That arc is
[the most important thing in this submission](#the-finding-that-matters-most-cross-domain-collapse-and-the-fix).

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
are. So the augmented model is **better calibrated than the baseline in our
in-distribution confidence analysis**. That analysis was run on CIFAKE, where
the augmented model was in-distribution; confidence bands on one dataset are
not a calibration guarantee, and cross-domain the same scores become
uninformative (see the collapse below).

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

Validation, evaluation and inference all share one preprocessing function,
`get_eval_transform()` in `src/data.py`, imported by every path that scores a
model, so those cannot silently drift apart — a mismatch there produces no
error, only quietly worse accuracy. Training deliberately uses a different
transform (`get_train_transform()`, or `get_robust_train_transform()` for the
augmented arms), since that is where the augmentation is applied.

### What this is for

The intended use is a **lightweight triage signal, not proof that an image is
synthetic**. A platform could run it over uploaded images after the normal
compression and resizing an upload pipeline applies — which is the regime the
robustness work targets — and feed the resulting probability into moderation
alongside provenance data, upload history and other signals, rather than
treating it as a verdict.

Keeping EfficientNet-B0 rather than a larger backbone is what makes that
feasible at platform scale: **1.14 ms per image**, against 4.12 ms for
ConvNeXt-Tiny for a difference in accuracy we could not distinguish from
noise. At upload volumes that is the difference between a signal you can
afford to compute on everything and one you cannot.

The false-positive analysis is the reason the output should not stand alone.
Robustness training raises the clean false-positive rate from 1.97% to 2.69%,
and every false positive is a real photograph flagged as synthetic. On a
large platform that is a lot of people wrongly accused if the score is used
as a decision rather than an input to one.

### The trade we are not hiding

Robustness is not free. On clean images the augmented model has a **higher
false-positive rate: 2.69% versus the baseline's 1.97%** (195 versus 143 false
positives out of 7,261 real images), against a slightly lower false-negative
rate (1.27% vs 1.41%). Augmentation shifts the decision boundary toward calling
images AI-generated. The trade is strongly favourable overall, but it moves in
the direction the challenge warns about — false accusations against genuine
photographs — and we did not tune the threshold to compensate.

### Beyond CIFAKE: SID_Set and the architecture choice

We also trained and evaluated on **SID_Set** — high-resolution images from a
different generator (FLUX). Three robustness-trained architectures were run
across the same 16-condition grid on the SID validation split: 5,099 images
per condition, threshold 0.5, seed 42, checkpoint SHA-256 hashes recorded.

| Model | Clean acc. | Mean damaged acc. | Worst damaged acc. | Latency |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 99.78% | 99.73% | 99.43% (noise 0.10) | 1.14 ms |
| ConvNeXt-Tiny | 99.80% | 99.78% | 99.45% (noise 0.10) | 4.12 ms |
| DINOv2 ViT-S/14 | 97.35% | 96.26% | 85.23% (crop 0.80) | 4.34 ms |

**ConvNeXt-Tiny came out marginally ahead and we did not choose it.** Its lead
is 0.05 percentage points of mean damaged accuracy, for **3.6× the inference
cost** (4.12 ms against 1.14 ms per image). A paired McNemar comparison across
all 16 conditions finds **no condition reaching p < 0.05** — the smallest
p-value is 0.093 — so on 5,099 images per condition the two models are
statistically indistinguishable. EfficientNet-B0 is the better trade-off and
stays as the working architecture; ConvNeXt-Tiny is retained as a candidate
for a later cross-generator comparison rather than discarded.

**DINOv2 ViT-S/14 is dropped.** Under crop 0.8 it falls **12 points below its
own clean accuracy** (85.23% against 97.35%), fails again under colour +0.2
(92.14%), and is the slowest of the three.

**The SID A→B comparison: augmentation buys noise resistance, and shows up
under chained damage.** A clean-trained SID baseline
(`experiment_sid_a_clean_best.pt`) has been evaluated against the
robustness-trained model on 5,099 SID validation images per condition.

Under *single* transforms the two are within ±1pp on 13 of 16 conditions.
The entire gap sits in three: **noise σ0.10 (56.85% vs 99.43%)**, noise
σ0.05 (79.15% vs 99.63%) and JPEG 30 (96.35% vs 99.75%). At σ0.10 the
clean-trained model's recall falls to 12.11% — it misses seven of every
eight AI images — though its AUROC is still 0.9972, so the ranking survives
and only the decision boundary has moved.

Under *randomly chained* damage (3 transforms per image, 5 trials, 25,495
pooled predictions) the separation is unambiguous: **82.997% against
99.078%** pooled accuracy, false-negative rate **34.60% against 1.72%**,
clean-correct retention **83.04% against 99.14%**, pooled AUROC 0.9873
against 0.9998. Trial standard deviation is 0.31pp and 0.07pp, so the
16-point gap is far outside run-to-run noise.

Single transforms understate what augmentation is worth; stacked damage —
which is how real redistribution works — shows it.

## The finding that matters most: cross-domain collapse, and the fix

**Our SID models score at chance on CIFAKE.** We took both EfficientNet-B0
SID models and ran them over the CIFAKE validation split — 14,724 images
(7,261 real, 7,463 AI), the same 16 conditions, threshold 0.5, seed 42.

| | SID-A (clean-trained) | SID-B (robustness-trained) |
|---|---:|---:|
| Clean CIFAKE accuracy | 51.18% | 49.41% |
| Clean recall on AI images | 3.91% (292 / 7,463) | 0.21% (16 / 7,463) |
| Accuracy range, 16 conditions | 49.31% – 51.18% | 49.31% – 50.29% |
| AUROC range | 0.5079 – 0.6406 | 0.5015 – 0.6505 |

Predicting "real" for everything scores **49.31%** here. Both models sit
within two points of that floor, failing almost entirely by false negative —
they call nearly every AI image real. Under noise 0.05 and 0.10 **both
detect zero AI images**, out of 7,463.

It is not a threshold problem. AUROC starts near 0.635 clean and reaches
**0.5015** under blur σ2.0 — random ranking. Sweeping the threshold over the
clean predictions recovers at most **59.6%** (SID-A) and **59.9%** (SID-B).
The scores carry almost no signal to recalibrate.

**Transform robustness and cross-generator generalization are separate
problems.** A model can be 99.8% accurate under every transform in the grid
on its own dataset and still be useless on an unseen generator. Nothing in
the SID results predicts this, and nothing in our CIFAKE robustness work
does either. For this challenge that is the consequential finding: a
deployed detector meets generators absent from its training set, and our
evidence says robustness augmentation buys resilience *within* a
distribution, not across distributions.

**It is not an architecture problem.** We ran the same CIFAKE evaluation on
**ConvNeXt-Tiny** — a different backbone family, 27.8M parameters against
EfficientNet's 4.0M, and the model that scored best of all three on SID
(99.80% clean). It collapses identically: **49.38%** clean CIFAKE accuracy,
**0.13%** recall on AI images (10 of 7,463), never above **0.46%** recall
under any of the 16 conditions, and zero AI detections under noise 0.02,
0.05 and 0.10. Its accuracy never leaves a 0.10-point band around the 49.31%
all-real floor — tighter to the floor than EfficientNet-B0.

Two architectures with different inductive biases, trained on the same data,
fail in the same direction and to the same degree. **That points at the
training distribution rather than model capacity as the cause.** Scaling the
backbone is not a route out; changing the training data is.

**Mixed training closes the gap on CIFAKE.** Training on SID and CIFAKE
together recovers almost all of the loss on CIFAKE: clean accuracy 49.41% →
**97.19%**, mean damaged accuracy 49.53% → **94.36%**, AUROC from a
0.50–0.65 band to **0.958–0.996**, recall on AI images 0.21% → 98.08%. Its
worst condition is blur σ2.0 at 87.75%.

Under *randomly chained* damage on CIFAKE (3 transforms per image, 5 trials,
73,620 pooled predictions) the mixed model holds **84.33%** pooled accuracy
with AUROC **0.942** and a 25.79% false-negative rate, against the SID-only
model's 49.66%, AUROC **0.515** and a **97.74%** false-negative rate.

**One caveat about how to read that table.** The SID-only model's
clean-correct retention (98.22%) is *higher* than the mixed model's
(85.41%), and its accuracy under damage is marginally *higher* than its
clean accuracy — a negative drop. Neither is robustness. It predicts "real"
for essentially every image (recall on AI images 2.26%, false-negative rate
97.74%), so damage cannot change an answer that never depended on the input:
it consistently keeps the real images it gets right and consistently misses
the AI images. A constant classifier is trivially stable. Retention and drop
measure consistency, not correctness. The mixed model loses 12.86 points
from clean to chained precisely because it is making real decisions that
damage can disrupt.

**And it costs almost nothing on SID.** The mixed model has now been
evaluated on the SID validation split as well — 5,099 images per condition,
the same 16 conditions. It gives up **0.47pp** on clean images (99.78% →
99.31%) and **0.92pp** at its worst condition (99.43% → 98.51% at noise
σ0.10). No condition costs more than 0.92pp, and AUROC is essentially
unchanged: 0.9998 against 0.9999.

### What the three results say together

| | SID | CIFAKE |
|---|---:|---:|
| SID-only B (EfficientNet-B0) | 99.78% | 49.41% |
| SID-only ConvNeXt-Tiny | 99.80% | 49.38% |
| **Mixed SID+CIFAKE B** | **99.31%** | **97.19%** |

**Backbone choice was not the dominant factor.** Two backbones from
different families — 4.0M and 27.8M parameters — trained on the same data
collapse to within 0.03pp of each other, both to a near-constant "real"
prediction. Tripling the parameter count changed nothing we could measure,
so the evidence points much more strongly to the training distribution than
to architecture or capacity. Two architectures is strong evidence, not
proof; a third family might behave differently, and we did not test one.

**Training-source diversity was far more influential than scaling the
backbone in our experiments.** Adding a second training source costs **0.47
points** on the original dataset and gains **47.78 points** on the
previously unseen one — roughly a hundred points gained for each one given
up. Neither a larger backbone nor heavier augmentation moved this failure
mode at all; a second source of images did.

The honest limit: this is one direction, one seed, two datasets. We have
shown that adding a source fixes performance on that source almost for free.
We have **not** shown that training on two generators generalizes to a
third — the held-out WildFake benchmark, still unrun, is what would test
that.

## A negative result: consistency training did not help

Experiment C is reported as a **negative result**.

> **Reproducibility caveat.** Experiment C does not run through the current
> CLI. `train.py` handles `train_mode: consistency`, but `get_dataloaders()`
> in `src/data.py` accepts only `clean` or `robust` and raises
> `ValueError: Unknown train_mode 'consistency'` before training starts. The
> C numbers below come from a separate, earlier run and cannot be regenerated
> with `python train.py --config configs/consistency.yaml` as the repository
> stands. Experiments A and B run as documented.

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
- **Google Colab** — GPU training and evaluation (mixed precision on CUDA);
  the CIFAKE experiments, the cross-domain runs and the mixed-model
  evaluations ran here
- **Kaggle** — dataset hosting for both CIFAKE and SID_Set, plus GPU
  notebooks: the SID architecture comparison (EfficientNet-B0 vs
  ConvNeXt-Tiny vs DINOv2) and the SID A-vs-B robustness evaluation were
  both trained and run on Kaggle
- **Git / GitHub** — version control, pull-request review across five people
- **Git Bash on Windows** — local shell
- **pytest** — automated tests (69 tests, all passing, covering the
  augmentation and consistency-loss modules, the metrics, the data pipeline,
  the model builder, the training loop, the evaluation runners and their
  multi-model checkpoint handling, and the inference CLI)
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

One backbone is shared across all three CIFAKE experiments, so that ablation
isolates training method rather than architecture.

Two further backbones were trained on SID_Set for the architecture comparison,
each with the same single-logit head: **ConvNeXt-Tiny** (torchvision,
27,820,897 parameters) and **DINOv2 ViT-S/14** (loaded via `torch.hub` from
`facebookresearch/dinov2`, fine-tuned end to end with a linear head on its
embedding output). DINOv2 was dropped on the results above. Every model is
orders of magnitude below the 2B parameter limit, and all inference is
local.

### The checkpoint we recommend

**`effnet_b0_sid_cifake_experiment_b_best.pt`** — EfficientNet-B0 trained on
SID_Set and CIFAKE together, SHA-256 `9159a9d4ceb7fccd…`.

Checkpoints are gitignored for size and hosted on Google Drive — download it
at <https://drive.google.com/file/d/1bz3KfWIPr422c7rGM9hYH3zs6wSr7pc7/view>
and verify against the full SHA-256 in the repo's README; without
`--checkpoint`, `predict.py` falls back to random output with a stderr
warning.

It is deliberately **not** the best checkpoint on either dataset taken
alone: the SID-only model beats it by 0.47pp on SID (99.78% against 99.31%).
But it is the only checkpoint that does not collapse on a dataset it was not
trained on — 99.31% on SID and 97.19% on CIFAKE, against the SID-only
model's 99.78% and **49.41%**. A submission is scored on images whose
generator is not known in advance, so the model that holds up across sources
is the right trade, and half a point on one dataset is a small price for
forty-eight on another.

## 4. Libraries and frameworks

From `requirements.txt`, which pins each package to the version our final
environment ran:

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
  68,712 training images and 14,724 validation images after 50/50 class
  balancing and a 70/15/15 split. **The A/B/C ablation — and therefore the
  headline transformation-flip result — comes entirely from CIFAKE.** The
  CIFAKE validation split also serves as the unseen-generator test set for
  the SID models, which is where the cross-domain collapse above shows up.
- **SID_Set** (Hugging Face, CC BY 4.0) — ~300K images generated with FLUX,
  including a tampered class (real photographs with an AI-generated region
  inserted). **We did not use the full dataset.** Our balanced binary
  manifest has a **5,099-image validation split**, which every SID number in
  this submission is measured on; under the 70/15/15 split that implies a
  total on the order of 34,000 images, not 300K. **SID_Set results are
  included in this submission**: three
  robustness-trained architectures evaluated across the 16-condition grid on
  5,099 validation images per condition, reported above, plus a clean-trained
  SID baseline compared against the robustness-trained model. Its tampered
  images are routed to a dedicated
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

1. **Generalization to an *unseen* generator remains unproven.** Training on
   a generator fixes that generator (SID-only models score 49.4% on CIFAKE;
   the mixed model scores 97.19%). What we have not shown is that training on
   two generators helps on a third. The held-out WildFake benchmark would
   test this and is still unrun, so for a generator absent from training we
   have **no positive evidence** — only the demonstrated failure mode, which
   was severe: at chance, AUROC 0.5015, zero AI images detected under noise,
   and no threshold recovering more than ~60%.
2. **The augmentation effect is broad on CIFAKE but narrow on SID.** The A→B
   flip result comes from CIFAKE alone: 32×32 images, one generator (SD 1.4),
   one validation split, one seed. On SID the two models are within ±1pp on
   13 of 16 single transforms — the whole gap is additive noise — and clear
   separation only appears under chained damage (82.997% vs 99.078%).
3. **Consistency training produced no measurable improvement** (above).
4. **The robustness gain costs clean-set precision** — false-positive rate rises
   from 1.97% to 2.69%.
5. **40 images are misclassified by the augmented model under all 16
   conditions.** No amount of augmentation moves them, which suggests label noise
   or intrinsic ambiguity in CIFAKE rather than a robustness failure. We did not
   open them to check.
6. **No per-source or per-generator error breakdown was possible.** The `source`
   column in our prediction logs is null in 100% of rows, and `dataset` and
   `generator` are constant, so we cannot say *which kinds* of image fail.
7. **Single seed, single split.** Small differences between arms cannot be called
   significant.
8. Model outputs are strongly bimodal — only 7.7–8.3% of predictions fall between
   0.20 and 0.80. This is expected from a single-logit network trained to
   near-zero loss, and is only a problem for the baseline, whose confidence is
   also poorly calibrated.
9. **The tampered holdout has never been analysed.** Tampered SID_Set images are
   correctly routed to a dedicated `bonus` split and kept out of binary training
   and evaluation, but no evaluation entry point currently exposes that split
   (`--split` accepts only `val` or `test`), and labels are emitted as float32
   for `BCEWithLogitsLoss`, so a 3-class row would need explicit handling. The
   quarantine works; the analysis it enables is future work.

**With more time, in priority order:** run WildFake against the mixed model,
which would show whether source diversity generalizes to a *third* generator
or only fixes the two it saw; inspect the 40 persistent failures for label
noise; tune the decision threshold to claw back the false-positive rate; and
re-run the A/B/C ablation across multiple seeds.

## Reproducibility

The repository README documents the full pipeline end to end — dataset
acquisition, manifest construction, the three training commands, the evaluation
grid, and inference — with every command verified against the actual CLI.
Per-condition metrics, the error analysis, and the design-decision log are
committed under `results/`; the analysis and plotting scripts under `scripts/`
regenerate the figures deterministically from the committed CSVs.
