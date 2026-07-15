#!/usr/bin/env python3
"""Simple local GPU one-label classification over a CSV using vLLM.

Supports both low-VRAM and datacenter GPUs (e.g., A100) with runtime presets.
- Streams input rows (does not load entire CSV into RAM)
- Processes rows in batches with one model load
- Uses a system prompt from a markdown or plain text file
- Appends results to an output CSV so runs can be resumed
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

MODEL_PRESETS = {
    "qwen-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen-2b-awq": "cyankiwi/Qwen3.5-2B-AWQ-4bit",
    "qwen-27b": "Qwen/Qwen3.5-27B",
    "gemma-31b": "google/gemma-4-31B-it",
    "gemma-31b-awq": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "ministral-14b": "mistralai/Ministral-3-14B-Instruct-2512",
}

RUNTIME_PRESETS: dict[str, dict[str, Any]] = {
    "low-vram": {
        "gpu_mem_util": 0.82,
        "max_model_len": 512,
        "max_num_seqs": 8,
        "batch_size": 16,
        "max_tokens": 192,
        "context_reserve_tokens": 32,
        "log_every": 2000,
        "flush_every": 1,
        "dtype": "auto",
    },
    "a100": {
        "gpu_mem_util": 0.94,
        "max_model_len": 1024,
        "max_num_seqs": 64,
        "batch_size": 64,
        "max_tokens": 96,
        "context_reserve_tokens": 128,
        "log_every": 5000,
        "flush_every": 8,
        "dtype": "bfloat16",
    },
}

MFT_LABELS = [
    "care",
    "harm",
    "fairness",
    "cheating",
    "loyalty",
    "betrayal",
    "authority",
    "subversion",
    "sanctity",
    "degradation",
]

LABEL_ALIASES = {
    "purity": "sanctity",
}

SHVT_LABELS = [
    "self_direction_thought",
    "self_direction_action",
    "stimulation",
    "hedonism",
    "achievement",
    "power_dominance",
    "power_resources",
    "face",
    "security_personal",
    "security_societal",
    "tradition",
    "conformity_rules",
    "conformity_interpersonal",
    "humility",
    "benevolence_caring",
    "benevolence_dependability",
    "universalism_concern",
    "universalism_nature",
    "universalism_tolerance",
    "universalism_objectivity",
]


def extract_prompt_from_markdown(md_text: str, prompt_type: str) -> str:
    """Extract prompt text nested under markdown headings for MFT or SHVT."""
    prompt_type = prompt_type.upper()

    if prompt_type == "MFT":
        pattern = r"## MFT.*?```text\n(.*?)```"
    elif prompt_type == "SHVT":
        pattern = r"## SHVT.*?```text\n(.*?)```"
    else:
        raise ValueError("prompt_type must be MFT or SHVT")

    match = re.search(pattern, md_text, flags=re.DOTALL)
    if not match:
        if "```" not in md_text:
            return md_text.strip()
        raise ValueError(f"Could not find {prompt_type} prompt in markdown file")
    return match.group(1).strip()


def parse_one_label_json(raw_text: str, valid_labels: set[str]) -> tuple[bool, str, str]:
    """Parse JSON containing a value_id key and validate it against the allowed label set."""
    text = (raw_text or "").strip()

    if not text:
        return False, "parse_error", "empty_response"

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return False, "parse_error", "no_json_object"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False, "parse_error", "json_parse_error"

    if not isinstance(obj, dict):
        return False, "parse_error", "json_not_object"
    if set(obj.keys()) != {"value_id"}:
        return False, "parse_error", "schema_mismatch"

    value_id = obj.get("value_id")
    if not isinstance(value_id, str):
        return False, "parse_error", "invalid_value_id_type"

    normalized = canonicalize_label(value_id)
    if normalized not in valid_labels:
        return False, "invalid_label", f"invalid_label:{normalized}"

    return True, normalized, ""


def build_one_label_json_schema(labels: list[str]) -> dict[str, Any]:
    """Build a JSON schema constraint for structured outputs requesting a single value_id."""
    return {
        "type": "object",
        "properties": {
            "value_id": {
                "type": "string",
                "enum": labels + ["none"],
            }
        },
        "required": ["value_id"],
        "additionalProperties": False,
    }


def canonicalize_label(label: str) -> str:
    """Normalize a label string and map it to its canonical form via LABEL_ALIASES."""
    normalized = label.strip()

    return LABEL_ALIASES.get(normalized, normalized)


def safe_divide(numerator: float, denominator: float) -> float:
    """Perform division returning 0.0 if the denominator is zero."""
    return numerator / denominator if denominator else 0.0


def write_one_label_metrics(output_csv: Path, labels: list[str], run_name: str) -> None:
    """Calculate and write accuracy, macro/weighted F1, confusion matrix, and classification report."""
    y_true: list[str] = []

    y_pred: list[str] = []

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = canonicalize_label(row.get("value_id") or "")
            pred_label = canonicalize_label(row.get("predicted_value_id") or "")
            if true_label and pred_label:
                y_true.append(true_label)
                y_pred.append(pred_label)

    metrics_dir = output_csv.parent
    metrics_dir.mkdir(parents=True, exist_ok=True)
    base = metrics_dir / run_name

    ordered_labels: list[str] = []
    appearing = set(y_true) | set(y_pred)
    for label in labels:
        if label not in ordered_labels:
            ordered_labels.append(label)
    for label in ["none", "parse_error", "invalid_label"]:
        if label in appearing and label not in ordered_labels:
            ordered_labels.append(label)
    ordered_labels.extend(sorted(appearing - set(ordered_labels)))

    counts: dict[tuple[str, str], int] = {}
    for true_label, pred_label in zip(y_true, y_pred):
        counts[(true_label, pred_label)] = counts.get((true_label, pred_label), 0) + 1

    report_rows: list[dict[str, Any]] = []
    total = len(y_true)
    correct = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label)

    for label in ordered_labels:
        tp = counts.get((label, label), 0)
        fp = sum(counts.get((true_label, label), 0) for true_label in ordered_labels if true_label != label)
        fn = sum(counts.get((label, pred_label), 0) for pred_label in ordered_labels if pred_label != label)
        support = sum(counts.get((label, pred_label), 0) for pred_label in ordered_labels)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        report_rows.append(
            {
                "class": label,
                "precision": f"{precision:.10f}",
                "recall": f"{recall:.10f}",
                "f1": f"{f1:.10f}",
                "support": support,
            }
        )

    macro_precision = safe_divide(sum(float(row["precision"]) for row in report_rows), len(report_rows))
    macro_recall = safe_divide(sum(float(row["recall"]) for row in report_rows), len(report_rows))
    macro_f1 = safe_divide(sum(float(row["f1"]) for row in report_rows), len(report_rows))
    weighted_precision = safe_divide(sum(float(row["precision"]) * int(row["support"]) for row in report_rows), total)
    weighted_recall = safe_divide(sum(float(row["recall"]) * int(row["support"]) for row in report_rows), total)
    weighted_f1 = safe_divide(sum(float(row["f1"]) * int(row["support"]) for row in report_rows), total)

    with (base.with_name(f"{run_name}_metrics.csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in [
            ("accuracy", safe_divide(correct, total)),
            ("macro_precision", macro_precision),
            ("macro_recall", macro_recall),
            ("macro_f1", macro_f1),
            ("weighted_precision", weighted_precision),
            ("weighted_recall", weighted_recall),
            ("weighted_f1", weighted_f1),
        ]:
            writer.writerow({"metric": metric, "value": f"{value:.10f}"})

    with (base.with_name(f"{run_name}_classification_report.csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(report_rows)

    with (base.with_name(f"{run_name}_confusion_matrix.csv")).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", *ordered_labels])
        for true_label in ordered_labels:
            writer.writerow([true_label, *[counts.get((true_label, pred_label), 0) for pred_label in ordered_labels]])

    with (base.with_name(f"{run_name}_summary.md")).open("w", encoding="utf-8") as f:
        f.write(f"# {run_name}\n\n")
        f.write(f"- Rows: {total}\n")
        f.write(f"- Accuracy: {safe_divide(correct, total):.6f}\n")
        f.write(f"- Macro F1: {macro_f1:.6f}\n")
        f.write(f"- Weighted F1: {weighted_f1:.6f}\n")


def load_structured_outputs_params_class() -> Any | None:
    """Dynamically import and return the StructuredOutputsParams class from vllm if available."""
    try:
        module = importlib.import_module("vllm.sampling_params")
        return getattr(module, "StructuredOutputsParams", None)
    except Exception:
        return None


def detect_csv_dialect(csv_path: Path) -> csv.Dialect:
    """Sniff a sample of the CSV file to detect its dialect and delimiter."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def count_completed_rows(out_csv: Path) -> int:
    """Return the number of data rows already written to the output CSV file."""
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def is_context_overflow_error(exc: Exception) -> bool:
    """Check if an exception indicates a context length/token limit overflow error."""
    msg = str(exc).lower()

    return (
        "maximum context length" in msg
        or "parameter=input_tokens" in msg
        or "vllmvalidationerror" in exc.__class__.__name__.lower()
    )


def chat_token_len(tokenizer: Any, system_prompt: str, sentence: str) -> int:
    """Compute the exact token length of the chat conversation formatted with the system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]
    tok_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(tok_ids, "tolist"):
        tok_ids = tok_ids.tolist()
    if isinstance(tok_ids, list):
        return len(tok_ids)
    return int(tok_ids.shape[-1])


def truncate_sentence_to_fit(
    tokenizer: Any,
    system_prompt: str,
    sentence: str,
    max_model_len: int,
    reserve_tokens: int,
) -> tuple[str, bool]:
    """Truncate a user sentence using binary search to fit within the model's max token length."""
    target_len = max(1, int(max_model_len) - max(0, int(reserve_tokens)))

    current_len = chat_token_len(tokenizer, system_prompt, sentence)
    if current_len <= target_len:
        return sentence, False

    # If even an empty user message is too long, no truncation can recover this prompt.
    if chat_token_len(tokenizer, system_prompt, "") > target_len:
        return "", False

    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if not sentence_ids:
        return "", False

    lo, hi = 0, len(sentence_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = tokenizer.decode(
            sentence_ids[:mid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        cand_len = chat_token_len(tokenizer, system_prompt, candidate)
        if cand_len <= target_len:
            lo = mid
        else:
            hi = mid - 1

    if lo <= 0:
        return "", False

    truncated = tokenizer.decode(
        sentence_ids[:lo],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return truncated, True


def apply_runtime_preset(args: argparse.Namespace) -> None:
    """Overlay settings from the chosen runtime profile onto the parsed arguments."""
    preset = RUNTIME_PRESETS[args.runtime_profile]

    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def main() -> int:
    """Parse command line arguments, load the local vLLM model, and stream batch inference over the CSV."""
    ap = argparse.ArgumentParser(description="Local GPU one-label CSV classification with vLLM")

    ap.add_argument("--input-csv", default="protoethosv2_3m_finalqc_20260512.csv", help="Input CSV path")
    ap.add_argument("--text-column", default="sentence", help="CSV column containing the sentence text")
    ap.add_argument("--output-csv", required=True, help="Output CSV path")
    ap.add_argument("--prompt-md", required=True, help="Markdown or plain text prompt file")
    ap.add_argument("--prompt-type", choices=["MFT", "SHVT"], required=True, help="Prompt type to apply")
    ap.add_argument("--id-column", default="id", help="CSV id column")
    ap.add_argument("--label-column", default="value_id", help="Ground-truth label column")
    ap.add_argument("--sample-index-column", default="sample_index", help="CSV sample index column")
    ap.add_argument(
        "--metrics-run-name",
        default="",
        help="Run name prefix for metrics files (default: output CSV stem)",
    )
    ap.add_argument(
        "--runtime-profile",
        choices=sorted(RUNTIME_PRESETS),
        default="a100",
        help="Performance preset for hardware class",
    )

    ap.add_argument("--model", choices=sorted(MODEL_PRESETS), default="qwen-0.8b", help="Model preset")
    ap.add_argument("--model-id", default="", help="Explicit HF model id override")
    ap.add_argument("--gpu-mem-util", type=float, default=None, help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=None, help="vLLM max_model_len")
    ap.add_argument("--max-num-seqs", type=int, default=None, help="vLLM max_num_seqs")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size")
    ap.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32", "half"],
        default=None,
        help="vLLM dtype",
    )
    ap.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (often safer on low VRAM)")
    ap.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM prefix caching for repeated system prompts",
    )
    ap.add_argument(
        "--language-model-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable multimodal inputs for VL models",
    )
    ap.add_argument(
        "--skip-mm-profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip multimodal profiling during engine init",
    )

    ap.add_argument("--batch-size", type=int, default=None, help="Rows per llm.chat() call")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--disable-structured-output",
        action="store_true",
        help="Disable guided JSON output constraints",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process at most N non-empty rows after resume skip (0 = all)")
    ap.add_argument(
        "--truncate-to-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-truncate long inputs when prompt exceeds max_model_len",
    )
    ap.add_argument(
        "--context-reserve-tokens",
        type=int,
        default=None,
        help="Token budget reserved for generation when truncating",
    )
    ap.add_argument("--resume", action="store_true", help="Resume by skipping rows already present in output")
    ap.add_argument("--log-every", type=int, default=None, help="Progress print frequency in rows")
    ap.add_argument("--flush-every", type=int, default=None, help="Flush output every N completed batches")

    args = ap.parse_args()
    apply_runtime_preset(args)

    input_csv = Path(args.input_csv)
    prompt_md = Path(args.prompt_md)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        print(f"ERROR: input CSV not found: {input_csv}", file=sys.stderr)
        return 1
    if not prompt_md.exists():
        print(f"ERROR: prompt markdown not found: {prompt_md}", file=sys.stderr)
        return 1

    model_id = args.model_id.strip() or MODEL_PRESETS[args.model]
    labels = MFT_LABELS if args.prompt_type == "MFT" else SHVT_LABELS
    one_label_valid_labels = set(labels) | {"none"}

    with prompt_md.open("r", encoding="utf-8") as f:
        md_text = f.read()
    system_prompt = extract_prompt_from_markdown(md_text, args.prompt_type)

    print(f"Loading model: {model_id}")
    print(
        "Runtime config: "
        f"profile={args.runtime_profile} "
        f"gpu_mem_util={args.gpu_mem_util} "
        f"max_model_len={args.max_model_len} "
        f"max_num_seqs={args.max_num_seqs} "
        f"batch_size={args.batch_size} "
        f"dtype={args.dtype}"
    )
    runtime_max_model_len = max(128, int(args.max_model_len))
    llm = LLM(
        model=model_id,
        max_model_len=runtime_max_model_len,
        gpu_memory_utilization=float(args.gpu_mem_util),
        max_num_seqs=max(1, int(args.max_num_seqs)),
        tensor_parallel_size=max(1, int(args.tensor_parallel_size)),
        dtype=str(args.dtype),
        enable_prefix_caching=bool(args.enable_prefix_caching),
        enforce_eager=bool(args.enforce_eager),
        language_model_only=bool(args.language_model_only),
        skip_mm_profiling=bool(args.skip_mm_profiling),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    try:
        base_prompt_tokens = chat_token_len(tokenizer, system_prompt, "")
    except Exception:
        base_prompt_tokens = -1
    if base_prompt_tokens > 0:
        print(f"Prompt token count (without sentence): {base_prompt_tokens}")
        if base_prompt_tokens >= runtime_max_model_len:
            print(
                "ERROR: system prompt alone exceeds --max-model-len. Increase --max-model-len or use a shorter prompt.",
                file=sys.stderr,
            )
            return 1

    effective_max_tokens = max(1, int(args.max_tokens))
    runtime_reserve_tokens = max(1, int(args.context_reserve_tokens))
    if base_prompt_tokens > 0:
        max_tokens_from_base = max(1, runtime_max_model_len - base_prompt_tokens - 8)
        if effective_max_tokens > max_tokens_from_base:
            print(
                f"WARNING: reducing max_tokens {effective_max_tokens} -> {max_tokens_from_base} to fit context budget."
            )
            effective_max_tokens = max_tokens_from_base
        runtime_reserve_tokens = max(runtime_reserve_tokens, effective_max_tokens)

    sampling_kwargs: dict[str, Any] = {
        "temperature": float(args.temperature),
        "max_tokens": effective_max_tokens,
    }

    if not args.disable_structured_output:
        StructuredOutputsParams = load_structured_outputs_params_class()
        if StructuredOutputsParams is not None:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=build_one_label_json_schema(labels),
                disable_any_whitespace=True,
                disable_additional_properties=True,
            )
            print("Structured JSON output: enabled")
        else:
            print("WARNING: StructuredOutputsParams not available; continuing without guided JSON.")
    else:
        print("Structured JSON output: disabled")

    sampling = SamplingParams(**sampling_kwargs)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_exists = output_csv.exists() and output_csv.stat().st_size > 0
    out_fields = [
        "sample_index",
        "id",
        "sentence",
        "value_id",
        "predicted_value_id",
        "parse_ok",
        "correct",
    ]

    completed_rows = count_completed_rows(output_csv) if args.resume else 0
    if args.resume and completed_rows > 0:
        print(f"Resume enabled: skipping first {completed_rows} data rows already in output")

    dialect = detect_csv_dialect(input_csv)
    started = time.time()
    total_seen = 0
    total_written = 0
    total_truncated = 0
    total_context_errors = 0
    accepted_rows = 0
    batch_rows: list[tuple[int, str, dict[str, str]]] = []
    flush_every = max(1, int(args.flush_every))

    def write_result_row(
        writer: csv.DictWriter,
        source_row_number: int,
        source_sentence: str,
        source_row: dict[str, str],
        raw_text: str,
        *,
        truncated_input: bool,
        forced_error: str = "",
    ) -> None:
        nonlocal total_written
        true_value_id = (source_row.get(args.label_column) or "").strip()
        sample_index = (source_row.get(args.sample_index_column) or "").strip()
        source_id = (source_row.get(args.id_column) or "").strip()

        if forced_error:
            parse_ok = False
            predicted_value_id = "parse_error"
        else:
            parse_ok, predicted_value_id, _ = parse_one_label_json(raw_text, one_label_valid_labels)

        row_out = {
            "sample_index": sample_index,
            "id": source_id,
            "sentence": source_sentence,
            "value_id": true_value_id,
            "predicted_value_id": predicted_value_id,
            "parse_ok": "1" if parse_ok else "0",
            "correct": "1" if canonicalize_label(true_value_id) == canonicalize_label(predicted_value_id) else "0",
        }
        writer.writerow(row_out)
        total_written += 1

    def process_single_row(row_item: tuple[int, str, dict[str, str]], writer: csv.DictWriter) -> None:
        nonlocal total_truncated, total_context_errors
        source_row_number, sentence_text, source_row = row_item
        model_sentence = sentence_text
        was_truncated = False

        if args.truncate_to_fit:
            model_sentence, was_truncated = truncate_sentence_to_fit(
                tokenizer,
                system_prompt,
                sentence_text,
                runtime_max_model_len,
                runtime_reserve_tokens,
            )
            if was_truncated:
                total_truncated += 1

        if not model_sentence:
            total_context_errors += 1
            write_result_row(
                writer,
                source_row_number,
                sentence_text,
                source_row,
                "",
                truncated_input=was_truncated,
                forced_error="context_overflow_unrecoverable",
            )
            return

        one_conv = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": model_sentence},
            ]
        ]
        try:
            out = llm.chat(one_conv, sampling)[0]
            raw = out.outputs[0].text if out.outputs else ""
            write_result_row(
                writer,
                source_row_number,
                sentence_text,
                source_row,
                raw,
                truncated_input=was_truncated,
            )
        except Exception as single_exc:
            total_context_errors += 1
            if is_context_overflow_error(single_exc):
                error_label = "context_overflow_after_truncate" if was_truncated else "context_overflow"
            else:
                error_label = f"single_row_error:{single_exc.__class__.__name__}"
            write_result_row(
                writer,
                source_row_number,
                sentence_text,
                source_row,
                "",
                truncated_input=was_truncated,
                forced_error=error_label,
            )

    def process_batch(rows: list[tuple[int, str, dict[str, str]]], writer: csv.DictWriter) -> None:
        if not rows:
            return

        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": s},
            ]
            for _, s, _ in rows
        ]
        try:
            outputs = llm.chat(conversations, sampling)
            for (source_row_number, sentence_text, source_row), out in zip(rows, outputs):
                raw = out.outputs[0].text if out.outputs else ""
                write_result_row(
                    writer,
                    source_row_number,
                    sentence_text,
                    source_row,
                    raw,
                    truncated_input=False,
                )
            return
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise

        # Split on overflow to retain batching for normal rows while isolating problematic long rows.
        if len(rows) == 1:
            process_single_row(rows[0], writer)
            return

        mid = len(rows) // 2
        process_batch(rows[:mid], writer)
        process_batch(rows[mid:], writer)

    with (
        input_csv.open("r", encoding="utf-8", newline="") as in_f,
        output_csv.open("a", encoding="utf-8", newline="") as out_f,
    ):
        reader = csv.DictReader(in_f, dialect=dialect)
        if not reader.fieldnames:
            print("ERROR: input CSV has no header", file=sys.stderr)
            return 1
        if args.text_column not in reader.fieldnames:
            print(f"ERROR: missing text column '{args.text_column}' in input CSV", file=sys.stderr)
            return 1
        for required_column in [args.sample_index_column, args.id_column, args.label_column]:
            if required_column not in reader.fieldnames:
                print(f"ERROR: missing required column '{required_column}' in input CSV", file=sys.stderr)
                return 1

        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        if not out_exists:
            writer.writeheader()

        batches_since_flush = 0

        for row_idx, row in enumerate(reader, start=1):
            total_seen += 1
            if row_idx <= completed_rows:
                continue

            sentence = (row.get(args.text_column) or "").strip()
            if not sentence:
                continue

            if args.limit > 0 and accepted_rows >= args.limit:
                break

            accepted_rows += 1

            batch_rows.append((row_idx, sentence, row))
            if len(batch_rows) < max(1, int(args.batch_size)):
                continue

            process_batch(batch_rows, writer)
            batches_since_flush += 1
            if batches_since_flush >= flush_every:
                out_f.flush()
                batches_since_flush = 0
            batch_rows.clear()

            if total_written > 0 and total_written % max(1, int(args.log_every)) == 0:
                elapsed = max(1e-6, time.time() - started)
                rate = total_written / elapsed
                print(f"Progress: written={total_written} seen={total_seen} rate={rate:.2f} rows/s")

        if batch_rows:
            process_batch(batch_rows, writer)
            batches_since_flush += 1
            if batches_since_flush >= flush_every:
                out_f.flush()
                batches_since_flush = 0

        if batches_since_flush > 0:
            out_f.flush()

    elapsed = max(1e-6, time.time() - started)
    print(f"Done. Wrote {total_written} rows in {elapsed:.1f}s ({total_written / elapsed:.2f} rows/s)")
    print(f"Truncated rows: {total_truncated} | Context-overflow failures: {total_context_errors}")
    print(f"Output: {output_csv}")
    run_name = args.metrics_run_name.strip() or output_csv.stem
    write_one_label_metrics(output_csv, labels, run_name)
    print(f"Metrics prefix: {output_csv.parent / run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
