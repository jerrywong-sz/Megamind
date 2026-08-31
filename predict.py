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


def load_model(
    checkpoint: Path | None,
    device: torch.device,
    architecture: str | None = None,
):
    """Load a trained model, supporting architecture-aware checkpoints."""
    if checkpoint is None:
        print(
            "WARNING: running with --allow-random and no checkpoint. "
            "Every 'pred' value below is RANDOM, not model output. "
            "Do not report these numbers as results.",
            file=sys.stderr,
        )
        return None

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint_data, dict):
        checkpoint_architecture = checkpoint_data.get(
            "architecture"
        )
        state_dict = checkpoint_data.get(
            "model_state",
            checkpoint_data,
        )
    else:
        checkpoint_architecture = None
        state_dict = checkpoint_data

    if (
        checkpoint_architecture is not None
        and architecture is not None
        and checkpoint_architecture != architecture
    ):
        raise ValueError(
            "Checkpoint architecture "
            f"'{checkpoint_architecture}' conflicts with "
            f"requested architecture '{architecture}'."
        )

    resolved_architecture = (
        checkpoint_architecture
        or architecture
        or "efficientnet_b0"
    )

    model = build_model(
        pretrained=False,
        architecture=resolved_architecture,
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint does not match architecture "
            f"'{resolved_architecture}': {checkpoint}"
        ) from error

    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory of images (searched recursively)")
    parser.add_argument("--checkpoint", default=None, type=Path, help="Path to a trained model checkpoint (.pt)")
    parser.add_argument(
        "--architecture",
        default=None,
        help=(
            "Model architecture override for legacy checkpoints "
            "without architecture metadata."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the output JSON to")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of images per inference batch")
    parser.add_argument("--threshold", default=0.5, type=float, help="Probability threshold for the positive class (only used with --include-label)")
    parser.add_argument(
        "--include-label",
        action="store_true",
        help=(
            "Add a 'predicted_label' key (0/1, thresholded at --threshold) to each "
            "output row. Off by default: the submission format is exactly "
            "image_path + pred."
        ),
    )
    parser.add_argument(
        "--allow-random",
        action="store_true",
        help=(
            "Permit running without --checkpoint, producing RANDOM predictions. "
            "Required to opt in, because the output JSON is indistinguishable "
            "from real output. For smoke-testing the pipeline only."
        ),
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: --input_dir '{args.input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Refuse to silently emit random numbers. The output file is valid JSON in
    # the submission format either way, so a reader who missed a stderr warning
    # would have no way to tell real predictions from noise. Make it explicit.
    if args.checkpoint is None and not args.allow_random:
        print(
            "ERROR: no --checkpoint provided.\n"
            "  Without a checkpoint this script can only emit RANDOM predictions, "
            "which look exactly like real output.\n"
            "  Pass --checkpoint <path_to_checkpoint.pt> to run the detector, or "
            "--allow-random to deliberately generate random values for a smoke test.\n"
            "  See the 'Getting the checkpoints' section of the README for the "
            "download link.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.checkpoint,
        device,
        architecture=args.architecture,
    )
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
            with torch.inference_mode():
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

    # The challenge submission format is exactly image_path + pred. Anything
    # extra is opt-in so the default output can never fail a strict schema check.
    output_rows = []
    for idx, path in enumerate(image_paths):
        row = {
            "image_path": path.relative_to(args.input_dir).as_posix(),
            "pred": round(float(preds[idx]), 3),
        }
        if args.include_label:
            row["predicted_label"] = 1 if preds[idx] >= args.threshold else 0
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_rows, f, indent=2)

    found = len(image_paths)
    succeeded = found - skipped
    print(f"Done. Images found: {found}, succeeded: {succeeded}, skipped: {skipped}.")


if __name__ == "__main__":
    main()
