"""Plot severity curves for Experiment A vs Experiment B.

Reads results/robustness_comparison_a_vs_b.csv and writes a small-multiples
figure to results/robustness_curves.png -- one panel per transform type, with
accuracy on the y-axis and increasing damage along the x-axis.

    python scripts/plot_robustness.py

Each panel starts from the clean baseline so the degradation trend is visible,
and every panel shares the same y-scale so the panels can be compared directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
import pandas as pd

# Categorical slots 1 and 2 from the validated palette. Colour follows the
# entity: A is always blue, B is always orange, in every panel.
COLOUR_A = "#2a78d6"
COLOUR_B = "#eb6834"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Panel order, and the severity order within each panel: always weakest damage
# first. Note this is NOT numeric order for every transform -- lower JPEG
# quality and smaller resize scale both mean *more* damage.
PANELS = [
    ("jpeg", "JPEG compression", [90.0, 70.0, 50.0, 30.0], lambda s: f"q{int(s)}"),
    ("blur", "Gaussian blur", [0.5, 1.0, 2.0], lambda s: f"σ{s:g}"),
    ("resize", "Resize + upscale", [0.5, 0.25], lambda s: f"{s:g}×"),
    ("noise", "Gaussian noise", [0.02, 0.05, 0.10], lambda s: f"σ{s:g}"),
    ("colour", "Colour jitter", [-0.20, 0.20], lambda s: f"{s:+.0%}"),
    ("crop", "Centre crop", [0.80], lambda s: f"{s:.0%}"),
]

CHANCE_LEVEL = 0.5  # balanced binary task


def load_comparison(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"transform", "severity", "model_a_accuracy", "model_b_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {csv_path} is missing columns: {sorted(missing)}")
    return df


def accuracy_at(df: pd.DataFrame, transform: str, severity: float | None) -> tuple[float, float]:
    """Look up (A, B) accuracy for one condition."""
    if severity is None:
        row = df[df["transform"] == "clean"]
    else:
        row = df[(df["transform"] == transform) & (df["severity"] == severity)]
    if row.empty:
        raise SystemExit(f"ERROR: no row for transform={transform!r} severity={severity!r}")
    return float(row["model_a_accuracy"].iloc[0]), float(row["model_b_accuracy"].iloc[0])


def build_figure(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    clean_a, clean_b = accuracy_at(df, "clean", None)

    for ax, (transform, title, severities, fmt) in zip(axes.flat, PANELS):
        # Every panel opens at the clean baseline, so the drop is visible.
        xs = list(range(len(severities) + 1))
        labels = ["clean"] + [fmt(s) for s in severities]

        ys_a = [clean_a]
        ys_b = [clean_b]
        for s in severities:
            a, b = accuracy_at(df, transform, s)
            ys_a.append(a)
            ys_b.append(b)

        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        # Chance level: below this the model is worse than guessing.
        ax.axhline(CHANCE_LEVEL, color=BASELINE, linewidth=0.8, zorder=1)

        for ys, colour, label in ((ys_a, COLOUR_A, "A"), (ys_b, COLOUR_B, "B")):
            ax.plot(
                xs, ys,
                color=colour, linewidth=2, zorder=3,
                marker="o", markersize=6,
                markerfacecolor=colour,
                markeredgecolor=SURFACE, markeredgewidth=2,  # 2px surface ring
                label=label,
            )

        ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=10, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9, color=INK_MUTED)
        ax.set_xlim(-0.35, len(severities) + 0.35)
        ax.tick_params(axis="y", labelsize=9, colors=INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0)

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.spines["bottom"].set_linewidth(0.8)

    # Shared y-scale: floor just below chance so a collapse to ~0.54 is legible
    # without the axis exaggerating the drop.
    axes.flat[0].set_ylim(0.45, 1.02)
    axes.flat[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes.flat[0].set_yticklabels([f"{v:.0%}" for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])

    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("Accuracy", color=INK_SECONDARY, fontsize=10)

    # Selective direct labels: one panel only, placed where the two lines are
    # furthest apart (blur σ1) so neither label sits on a line or its twin.
    blur_panel = axes[0][1]
    blur_a, blur_b = accuracy_at(df, "blur", 1.0)
    blur_panel.annotate(
        "B — augmented",
        xy=(2, blur_b), xytext=(0, -16), textcoords="offset points",
        ha="center", fontsize=9, color=INK_SECONDARY, fontweight="bold",
    )
    blur_panel.annotate(
        "A — clean baseline",
        xy=(2, blur_a), xytext=(0, -18), textcoords="offset points",
        ha="center", fontsize=9, color=INK_SECONDARY, fontweight="bold",
    )

    handles = [
        plt.Line2D([], [], color=COLOUR_A, linewidth=2, marker="o", markersize=6,
                   markeredgecolor=SURFACE, markeredgewidth=2,
                   label="Experiment A — clean baseline"),
        plt.Line2D([], [], color=COLOUR_B, linewidth=2, marker="o", markersize=6,
                   markeredgecolor=SURFACE, markeredgewidth=2,
                   label="Experiment B — robustness augmented"),
        plt.Line2D([], [], color=BASELINE, linewidth=0.8, label="Chance (50%)"),
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.945),
        frameon=False, fontsize=10, labelcolor=INK_SECONDARY, ncol=3,
        handletextpad=0.6, columnspacing=2.0,
    )

    fig.suptitle(
        "Accuracy under increasing transformation severity",
        x=0.062, y=0.985, ha="left", fontsize=14, color=INK_PRIMARY,
    )
    fig.text(
        0.062, 0.022,
        "CIFAKE validation split (14,724 images). Each panel starts at the clean "
        "baseline; x-axis runs weakest to strongest damage.\n"
        "Source: results/robustness_comparison_a_vs_b.csv",
        ha="left", fontsize=8.5, color=INK_MUTED,
    )

    fig.tight_layout(rect=(0.045, 0.075, 0.985, 0.90))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        default=Path("results/robustness_comparison_a_vs_b.csv"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("results/robustness_curves.png"),
        type=Path,
    )
    args = parser.parse_args()

    if not args.comparison.is_file():
        raise SystemExit(f"ERROR: comparison CSV not found: {args.comparison}")

    df = load_comparison(args.comparison)
    fig = build_figure(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
