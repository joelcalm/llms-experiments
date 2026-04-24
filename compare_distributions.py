#!/usr/bin/env python3
"""Compare score distributions with entropy and Jensen-Shannon divergence.

Reads MFT score CSV files, drops zero score values, builds score histograms on 1..N,
and reports:
- Entropy per model per range
- JSD between model pairs per range

Default inputs match this repository's naming pattern.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DistStats:
    model: str
    range_max: int
    total_nonzero: int
    entropy_bits: float
    normalized_entropy: float


@dataclass
class PairStats:
    model_a: str
    model_b: str
    range_max: int
    jsd_bits: float
    js_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute entropy and JSD for score distributions.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory holding mft_scores_0_<range>_... CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/plots/distribution_metrics_entropy_jsd.csv"),
        help="Output CSV path for metrics summary.",
    )
    parser.add_argument(
        "--ranges",
        type=int,
        nargs="+",
        default=[5, 20, 100],
        help="Score ranges to process (interpreted as 1..N after dropping zeros).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen08b", "qwen2b_awq"],
        help="Model suffixes in filenames.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="mft_scores",
        help="Filename prefix before _0_<range>_...",
    )
    parser.add_argument(
        "--run-tag",
        default="1m_seed20260417",
        help="Filename middle tag after range and before model suffix.",
    )
    return parser.parse_args()


def is_parse_ok(value: str) -> bool:
    return str(value).strip() in {"1", "true", "True", "TRUE"}


def iter_nonzero_scores(csv_path: Path) -> Iterable[int]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_parse_ok(row.get("parse_ok", "")):
                continue
            raw = row.get("scores_json", "")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            for value in parsed.values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and int(value) == float(value):
                    score = int(value)
                    if score != 0:
                        yield score


def histogram_for_range(csv_path: Path, range_max: int) -> list[int]:
    counts = [0] * (range_max + 1)
    for score in iter_nonzero_scores(csv_path):
        if 1 <= score <= range_max:
            counts[score] += 1
    return counts


def to_probs(counts: list[int]) -> list[float]:
    total = float(sum(counts))
    if total == 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def entropy_bits(probabilities: Iterable[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def jsd_bits(p: list[float], q: list[float]) -> float:
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kld_bits(a_probs: list[float], b_probs: list[float]) -> float:
        total = 0.0
        for a_i, b_i in zip(a_probs, b_probs):
            if a_i > 0.0:
                total += a_i * math.log2(a_i / b_i)
        return total

    return 0.5 * kld_bits(p, m) + 0.5 * kld_bits(q, m)


def build_input_path(input_dir: Path, prompt_prefix: str, range_max: int, run_tag: str, model: str) -> Path:
    return input_dir / f"{prompt_prefix}_0_{range_max}_{run_tag}_{model}.csv"


def main() -> int:
    args = parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    distributions: dict[tuple[int, str], list[float]] = {}
    dist_rows: list[DistStats] = []
    pair_rows: list[PairStats] = []

    for range_max in args.ranges:
        for model in args.models:
            csv_path = build_input_path(args.input_dir, args.prompt_prefix, range_max, args.run_tag, model)
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing input file: {csv_path}")

            counts = histogram_for_range(csv_path, range_max)
            probs = to_probs(counts[1:])
            total_nonzero = sum(counts[1:])
            h_bits = entropy_bits(probs)
            max_h = math.log2(range_max) if range_max > 1 else 1.0
            h_norm = (h_bits / max_h) if total_nonzero > 0 else 0.0

            distributions[(range_max, model)] = probs
            dist_rows.append(
                DistStats(
                    model=model,
                    range_max=range_max,
                    total_nonzero=total_nonzero,
                    entropy_bits=h_bits,
                    normalized_entropy=h_norm,
                )
            )

        for model_a, model_b in itertools.combinations(args.models, 2):
            p = distributions[(range_max, model_a)]
            q = distributions[(range_max, model_b)]
            jsd = jsd_bits(p, q)
            pair_rows.append(
                PairStats(
                    model_a=model_a,
                    model_b=model_b,
                    range_max=range_max,
                    jsd_bits=jsd,
                    js_distance=math.sqrt(jsd),
                )
            )

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "section",
                "range_max",
                "model",
                "model_a",
                "model_b",
                "total_nonzero",
                "entropy_bits",
                "normalized_entropy",
                "jsd_bits",
                "js_distance",
            ]
        )
        for row in dist_rows:
            writer.writerow(
                [
                    "distribution",
                    row.range_max,
                    row.model,
                    "",
                    "",
                    row.total_nonzero,
                    f"{row.entropy_bits:.8f}",
                    f"{row.normalized_entropy:.8f}",
                    "",
                    "",
                ]
            )
        for row in pair_rows:
            writer.writerow(
                [
                    "pairwise",
                    row.range_max,
                    "",
                    row.model_a,
                    row.model_b,
                    "",
                    "",
                    "",
                    f"{row.jsd_bits:.8f}",
                    f"{row.js_distance:.8f}",
                ]
            )

    print(f"Wrote metrics: {args.output_csv}")

    print("\nEntropy by model:")
    for row in sorted(dist_rows, key=lambda x: (x.range_max, x.model)):
        print(
            f"  range=1-{row.range_max:>3} model={row.model:<11} "
            f"n={row.total_nonzero:<8} H={row.entropy_bits:.6f} H_norm={row.normalized_entropy:.6f}"
        )

    print("\nJSD by range:")
    for row in sorted(pair_rows, key=lambda x: (x.range_max, x.model_a, x.model_b)):
        print(
            f"  range=1-{row.range_max:>3} {row.model_a} vs {row.model_b} "
            f"JSD={row.jsd_bits:.6f} JS_distance={row.js_distance:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
