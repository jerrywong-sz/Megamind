# Experiment Decisions

## Experiment A — Conventional Baseline

EfficientNet-B0 trained using standard image preprocessing and conventional augmentation:

- Resize to 256 × 256
- Center crop to 224 × 224
- Random horizontal flip
- ImageNet normalization

Experiment A does **not** use any robustness-specific degradation such as
JPEG compression, blur, resize degradation, Gaussian noise, colour jitter,
or robustness crop.

This represents a competent conventional image-classification baseline rather
than an intentionally weak baseline.

## Experiment B — Robustness Augmentation

Experiment B uses the same base pipeline, backbone, and training
hyperparameters as Experiment A, but additionally applies `random_transform()`
using challenge-relevant degradations:

- JPEG compression
- Gaussian blur
- Resize and upscale
- Gaussian noise
- Colour jitter
- Crop

## Experiment C — Consistency Training

Experiment C uses the same robustness augmentation as Experiment B.

For each source image, the model receives both a clean view and a damaged view.
Training includes the normal classification loss plus a consistency penalty
that encourages the model to produce similar predictions for both views.

## Fair Ablation

To make A/B/C directly comparable, the following should remain constant:

- EfficientNet-B0 backbone
- Random seed
- Number of epochs
- Batch size
- Learning rate
- Weight decay
- Train/validation split
- Other shared training settings

The intended comparison is:

A → conventional detector

B → effect of robustness augmentation

C → additional effect of explicit consistency training

## Format Normalization

`src/build_manifest.py` does not use the downloaded files in place. Every
image is re-saved as a standard JPEG (quality 95, converted to RGB so any
alpha channel is stripped) into a separate `--output_dir`, and the
manifest's `image_path` points at that re-saved copy.

Reasoning: if real and AI images came from sources with different file
formats (e.g. real images predominantly JPEG, fake images predominantly
PNG), the model could learn to read the file format as a shortcut instead
of the image content — scoring well on our data while being useless in
the real world. Forcing every image through identical JPEG re-encoding
removes that shortcut. It also means every image carries the same JPEG
compression artifacts going in, so compression history can't act as a
class signal either.

Note: the re-saved file keeps its **original extension**. The code
computes `save_path.with_suffix('.jpg')` but doesn't assign the result,
so a `.png` input stays named `.png` while containing JPEG bytes. Loading
is unaffected (Pillow reads by content, not extension), but the output
directory is misleading to inspect by hand.

## Tampered Image Holdout (SID_Set)

SID_Set has three classes: `0` real, `1` fully synthetic, `2` tampered
(a real photo with an AI-generated region inserted). The challenge is
binary, so tampered images are excluded from the 50/50 real/fake class
balancing and from the 70/15/15 split, then reattached with their own
split value.

Reasoning: a tampered image is mostly authentic pixels, so training on it
as "AI" risks pushing the model toward false positives on genuine
photographs — exactly what the challenge warns against. Tampered images
are held back for separate analysis instead of being folded into the main
real/fake split.

**Open question — which split value.** The original intent was a
dedicated `bonus` split, keeping tampered rows clearly separate from
every binary split. The current code assigns `split_override = "test"`
instead, which places tampered rows in the same split as binary test
rows — meaning a plain `split == "test"` filter would silently mix
3-class tampered data into a binary test set. This is under discussion
and not yet settled.
