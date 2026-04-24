#!/usr/bin/env python3
"""Simple local GPU batch inference over a CSV using vLLM.

Supports both low-VRAM and datacenter GPUs (e.g., A100) with runtime presets.
- Streams input rows (does not load entire CSV into RAM)
- Processes rows in batches with one model load
- Uses modular system prompts built from components in prompt_builder.py
- Appends results to an output CSV so runs can be resumed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from prompt_builder import build_prompt, load_prompt_components, get_labels

MODEL_PRESETS = {
    "qwen-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen-2b-awq": "cyankiwi/Qwen3.5-2B-AWQ-4bit",
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


def parse_model_json(raw_text: str, score_max: int) -> tuple[bool, dict[str, int] | None, str]:
    text = (raw_text or "").strip()
    if not text:
        return False, None, "empty_response"

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return False, None, "no_json_object"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False, None, "json_parse_error"

    if not isinstance(obj, dict):
        return False, None, "json_not_object"

    scores = obj.get("scores")
    if not isinstance(scores, dict):
        return False, None, "missing_scores"

    normalized: dict[str, int] = {}
    for key, value in scores.items():
        if isinstance(value, bool):
            return False, None, f"invalid_score_type:{key}"
        if isinstance(value, (int, float)) and int(value) == float(value):
            iv = int(value)
            if iv < 0 or iv > score_max:
                return False, None, f"score_out_of_range:{key}"
            normalized[key] = iv
        else:
            return False, None, f"invalid_score_type:{key}"

    return True, normalized, ""


def build_json_schema(labels: list[str], score_max: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    label: {"type": "integer", "minimum": 0, "maximum": score_max}
                    for label in labels
                },
                "required": labels,
                "additionalProperties": False,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def load_structured_outputs_params_class() -> Any | None:
    try:
        module = importlib.import_module("vllm.sampling_params")
        return getattr(module, "StructuredOutputsParams", None)
    except Exception:
        return None


def detect_csv_dialect(csv_path: Path) -> csv.Dialect:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def count_completed_rows(out_csv: Path) -> int:
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_context_overflow_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "maximum context length" in msg
        or "parameter=input_tokens" in msg
        or "vllmvalidationerror" in exc.__class__.__name__.lower()
    )


def chat_token_len(tokenizer: Any, system_prompt: str, sentence: str) -> int:
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
    preset = RUNTIME_PRESETS[args.runtime_profile]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple local GPU CSV batch inference with vLLM")

    ap.add_argument("--input-csv", default="v2_3m_final_clean_text.csv", help="Input CSV path")
    ap.add_argument("--text-column", default="text", help="CSV column containing the sentence text")
    ap.add_argument("--output-csv", required=True, help="Output CSV path")
    ap.add_argument("--prompt-md", default="prompt_examples.md", help="Markdown file with prompts")
    ap.add_argument("--prompt-type", choices=["MFT", "SHVT"], required=True, help="Prompt type to apply")
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
        "--score-max",
        type=int,
        default=100,
        help="Maximum allowed integer score per label (default: 100)",
    )
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
    ap.add_argument("--include-text", action="store_true", help="Include original text in output CSV")
    ap.add_argument("--log-every", type=int, default=None, help="Progress print frequency in rows")
    ap.add_argument("--flush-every", type=int, default=None, help="Flush output every N completed batches")

    args = ap.parse_args()
    apply_runtime_preset(args)

    if args.score_max <= 0:
        print("ERROR: --score-max must be > 0", file=sys.stderr)
        return 1

    input_csv = Path(args.input_csv)
    prompt_md = Path(args.prompt_md)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        print(f"ERROR: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    # Load prompt components from the markdown file
    if not prompt_md.exists():
        print(f"ERROR: prompt markdown not found: {prompt_md}", file=sys.stderr)
        return 1
    load_prompt_components(str(prompt_md))

    model_id = args.model_id.strip() or MODEL_PRESETS[args.model]
    labels = get_labels(args.prompt_type)

    # Use modular prompt builder
    system_prompt = build_prompt(args.prompt_type)

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
                f"WARNING: reducing max_tokens {effective_max_tokens} -> {max_tokens_from_base} to fit context budget.")
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
                json=build_json_schema(labels, int(args.score_max)),
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
        "row_number",
        "text_hash",
        "prompt_type",
        "truncated_input",
        "parse_ok",
        "scores_json",
        "error",
    ]
    if args.include_text:
        out_fields.append("text")

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
    batch_rows: list[tuple[int, str]] = []
    expected_labels = set(labels)
    flush_every = max(1, int(args.flush_every))

    def write_result_row(
        writer: csv.DictWriter,
        source_row_number: int,
        source_sentence: str,
        raw_text: str,
        *,
        truncated_input: bool,
        forced_error: str = "",
    ) -> None:
        nonlocal total_written
        parse_ok = False
        scores_obj: dict[str, int] | None = None
        error = forced_error
        if not forced_error:
            parse_ok, scores_obj, error = parse_model_json(raw_text, int(args.score_max))
            if scores_obj is not None:
                actual = set(scores_obj.keys())
                if actual != expected_labels:
                    parse_ok = False
                    missing = sorted(expected_labels - actual)
                    extras = sorted(actual - expected_labels)
                    details = []
                    if missing:
                        details.append(f"missing={','.join(missing[:5])}")
                    if extras:
                        details.append(f"extras={','.join(extras[:5])}")
                    error = "label_mismatch:" + ";".join(details)

        row_out: dict[str, Any] = {
            "row_number": source_row_number,
            "text_hash": text_hash(source_sentence),
            "prompt_type": args.prompt_type,
            "truncated_input": "1" if truncated_input else "0",
            "parse_ok": "1" if parse_ok else "0",
            "scores_json": json.dumps(scores_obj, ensure_ascii=False, separators=(",", ":")) if scores_obj else "",
            "error": error,
        }
        if args.include_text:
            row_out["text"] = source_sentence
        writer.writerow(row_out)
        total_written += 1

    def process_single_row(row_item: tuple[int, str], writer: csv.DictWriter) -> None:
        nonlocal total_truncated, total_context_errors
        source_row_number, sentence_text = row_item
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
                "",
                truncated_input=was_truncated,
                forced_error=error_label,
            )

    def process_batch(rows: list[tuple[int, str]], writer: csv.DictWriter) -> None:
        if not rows:
            return

        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": s},
            ]
            for _, s in rows
        ]
        try:
            outputs = llm.chat(conversations, sampling)
            for (source_row_number, sentence_text), out in zip(rows, outputs):
                raw = out.outputs[0].text if out.outputs else ""
                write_result_row(
                    writer,
                    source_row_number,
                    sentence_text,
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

    with input_csv.open("r", encoding="utf-8", newline="") as in_f, output_csv.open("a", encoding="utf-8", newline="") as out_f:
        reader = csv.DictReader(in_f, dialect=dialect)
        if not reader.fieldnames:
            print("ERROR: input CSV has no header", file=sys.stderr)
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

            batch_rows.append((row_idx, sentence))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
