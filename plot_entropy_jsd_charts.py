#!/usr/bin/env python3
"""Create charts for entropy/JSD distribution comparison results.

Outputs charts into outputs/plots by default:
- entropy_by_range_normalized.png
- jsd_by_range.png
- distribution_overlay_probabilities.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot entropy/JSD comparison charts.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/plots/distribution_metrics_entropy_jsd.csv"),
        help="Metrics CSV produced by compare_distributions.py",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory holding mft_scores_0_<range>_... CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory for chart PNG files",
    )
    parser.add_argument(
        "--ranges",
        type=int,
        nargs="+",
        default=[5, 20, 100],
        help="Ranges to plot as 1..N distributions",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen08b", "qwen2b_awq"],
        help="Model suffixes in filenames",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="mft_scores",
        help="Filename prefix before _0_<range>_...",
    )
    parser.add_argument(
        "--run-tag",
        default="1m_seed20260417",
        help="Filename middle tag after range and before model suffix",
    )
    return parser.parse_args()


def read_metrics(metrics_csv: Path) -> tuple[dict[tuple[int, str], float], dict[int, float]]:
    entropy_norm: dict[tuple[int, str], float] = {}
    jsd_bits: dict[int, float] = {}

    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            section = row.get("section", "")
            range_max = int(row["range_max"])
            if section == "distribution":
                model = row["model"]
                entropy_norm[(range_max, model)] = float(row["normalized_entropy"])
            elif section == "pairwise":
                jsd_bits[range_max] = float(row["jsd_bits"])

    return entropy_norm, jsd_bits


def histogram_probs(csv_path: Path, range_max: int) -> list[float]:
    counts = [0] * (range_max + 1)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("parse_ok", "")).strip() not in {"1", "true", "True", "TRUE"}:
                continue
            raw = row.get("scores_json", "")
            if not raw:
                continue
            try:
                score_map = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(score_map, dict):
                continue
            for v in score_map.values():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and int(v) == float(v):
                    iv = int(v)
                    if 1 <= iv <= range_max:
                        counts[iv] += 1

    total = sum(counts[1:])
    if total == 0:
        return [0.0] * range_max
    return [c / total for c in counts[1:]]


def build_input_path(input_dir: Path, prompt_prefix: str, range_max: int, run_tag: str, model: str) -> Path:
    return input_dir / f"{prompt_prefix}_0_{range_max}_{run_tag}_{model}.csv"


def plot_entropy(entropy_norm: dict[tuple[int, str], float], ranges: list[int], models: list[str], output_dir: Path) -> Path:
    x = list(range(len(ranges)))
    width = 0.38 if len(models) == 2 else max(0.15, 0.8 / max(1, len(models)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, model in enumerate(models):
        vals = [entropy_norm.get((r, model), 0.0) for r in ranges]
        offsets = [v + (idx - (len(models) - 1) / 2) * width for v in x]
        ax.bar(offsets, vals, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([f"1-{r}" for r in ranges])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized entropy")
    ax.set_xlabel("Score range")
    ax.set_title("Distribution spread by model (normalized entropy)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = output_dir / "entropy_by_range_normalized.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_jsd(jsd_bits: dict[int, float], ranges: list[int], output_dir: Path) -> Path:
    vals = [jsd_bits.get(r, 0.0) for r in ranges]
    labels = [f"1-{r}" for r in ranges]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, vals)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("JSD (bits)")
    ax.set_xlabel("Score range")
    ax.set_title("Model divergence by range (JSD)")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    out = output_dir / "jsd_by_range.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_overlays(args: argparse.Namespace, output_dir: Path) -> Path:
    fig, axes = plt.subplots(nrows=len(args.ranges), ncols=1, figsize=(10, 3.2 * len(args.ranges)))
    if len(args.ranges) == 1:
        axes = [axes]

    for ax, range_max in zip(axes, args.ranges):
        x = list(range(1, range_max + 1))
        for model in args.models:
            src = build_input_path(args.input_dir, args.prompt_prefix, range_max, args.run_tag, model)
            probs = histogram_probs(src, range_max)
            ax.plot(x, probs, marker="o", linewidth=1.5, markersize=2.5, label=model)

        ax.set_title(f"Non-zero score probability by bin (1-{range_max})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    out = output_dir / "distribution_overlay_probabilities.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entropy_norm, jsd_bits = read_metrics(args.metrics_csv)

    chart1 = plot_entropy(entropy_norm, args.ranges, args.models, args.output_dir)
    chart2 = plot_jsd(jsd_bits, args.ranges, args.output_dir)
    chart3 = plot_overlays(args, args.output_dir)

    print(f"WROTE {chart1}")
    print(f"WROTE {chart2}")
    print(f"WROTE {chart3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
