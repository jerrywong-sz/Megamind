"""Run the AI-generated-image detector over a directory of images.

Usage:
    python predict.py --input_dir test_images --checkpoint model.pt --output results/preds.json

If --checkpoint is omitted, predictions fall back to random values (with a
warning) so the rest of the pipeline (this script, downstream tooling, CI)
stays runnable before a trained checkpoint exists.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from src.data import get_eval_transform
from src.models import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(input_dir: Path) -> list[Path]:
    """Recursively collect image files under input_dir, extension match case-insensitive."""
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_model(checkpoint: Path | None, device: torch.device):
    """Load a trained model from checkpoint, or None to signal random fallback."""
    if checkpoint is None:
        print(
            "WARNING: no --checkpoint provided. Predictions will be RANDOM, "
            "not real model output. Pass --checkpoint <path> once a trained "
            "checkpoint is available.",
            file=sys.stderr,
        )
        return None

    model = build_model(pretrained=False)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory of images (searched recursively)")
    parser.add_argument("--checkpoint", default=None, type=Path, help="Path to a trained model checkpoint (.pt)")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the output JSON to")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of images per inference batch")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: --input_dir '{args.input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    transform = get_eval_transform()

    image_paths = find_images(args.input_dir)
    preds: list[float] = [0.5] * len(image_paths)
    skipped = 0

    batch_tensors: list[torch.Tensor] = []
    batch_indices: list[int] = []

    def flush_batch():
        """Run the accumulated batch through the model (or random fallback) and store results."""
        nonlocal batch_tensors, batch_indices
        if not batch_tensors:
            return

        batch = torch.stack(batch_tensors).to(device)
        if model is None:
            probs = torch.rand(batch.size(0))
        else:
            with torch.no_grad():
                logits = model(batch).squeeze(1)
                probs = torch.sigmoid(logits)

        for idx, prob in zip(batch_indices, probs.tolist()):
            preds[idx] = prob

        batch_tensors = []
        batch_indices = []

    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                tensor = transform(img.convert("RGB"))
        except Exception as exc:
            print(f"WARNING: skipping unreadable image '{path}': {exc}", file=sys.stderr)
            skipped += 1
            continue

        batch_tensors.append(tensor)
        batch_indices.append(idx)
        if len(batch_tensors) >= args.batch_size:
            flush_batch()

    flush_batch()

    output_rows = [
        {
            "image_path": path.relative_to(args.input_dir).as_posix(),
            "pred": round(float(preds[idx]), 3),
        }
        for idx, path in enumerate(image_paths)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_rows, f, indent=2)

    found = len(image_paths)
    succeeded = found - skipped
    print(f"Done. Images found: {found}, succeeded: {succeeded}, skipped: {skipped}.")


if __name__ == "__main__":
    main()
