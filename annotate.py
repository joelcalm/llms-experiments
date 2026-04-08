"""Annotate sentences from CSV with moral/value scores using a local vLLM model."""

import argparse
import csv
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b": "cyankiwi/Qwen3.5-2B-AWQ-4bit",
}

MFT_LABELS = ["care", "harm", "fairness", "cheating", "loyalty", "betrayal",
               "authority", "subversion", "sanctity", "degradation"]

SHVT_LABELS = ["self_direction_thought", "self_direction_action", "stimulation",
               "hedonism", "achievement", "power_dominance", "power_resources",
               "face", "security_personal", "security_societal", "tradition",
               "conformity_rules", "conformity_interpersonal", "humility",
               "benevolence_caring", "benevolence_dependability",
               "universalism_concern", "universalism_nature",
               "universalism_tolerance", "universalism_objectivity"]

SYSTEM_TEMPLATE = """\
You are an expert annotator of moral/value content in text.

Label set: {label_set}

Task:
- Input is ONE sentence.
- Identify which labels in the provided label set are conveyed/invoked by the sentence.
- Invocation includes praise, blame, appeal, critique, or framing.
- Use ONLY what is in the sentence; do not assume external context.

Scoring:
- For each label in the label set, assign an integer 0..100 for strength of evidence.
- Output ALL labels exactly once; do not omit labels.
- If none apply, return all labels with score 0.
- Integers only. No extra keys. No explanations. JSON only.
Format:
- Output must be a single-line JSON object (no newlines/indentation).
IMPORTANT: Provide only the final answer. Do not include reasoning steps, chain-of-thought, hidden analysis, or <think> tags.
"""


def build_json_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {l: {"type": "integer", "minimum": 0, "maximum": 100} for l in labels},
                "required": labels,
                "additionalProperties": False,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def load_sentences(path: str, n: int | None = None) -> list[str]:
    sentences = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            sentences.append(row[0])
            if n and len(sentences) >= n:
                break
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Annotate sentences with MFT/SHVT scores via vLLM.")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="0.8b",
                        help="Model size to use (default: 0.8b)")
    parser.add_argument("--task", choices=["mft", "shvt"], default="mft",
                        help="Label set (default: mft)")
    parser.add_argument("--csv", default="v2_3m_final_clean_text.csv",
                        help="Input CSV path")
    parser.add_argument("-n", type=int, default=10,
                        help="Number of sentences to process (default: 10)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSONL path (default: results_<task>_<model>.jsonl)")
    args = parser.parse_args()

    labels = MFT_LABELS if args.task == "mft" else SHVT_LABELS
    system_prompt = SYSTEM_TEMPLATE.format(label_set=args.task.upper())
    schema = build_json_schema(labels)
    model_name = MODELS[args.model]
    output_path = args.output or f"results_{args.task}_{args.model}.jsonl"

    print(f"Loading model: {model_name}")
    # WSL + 4GB VRAM: keep this conservative so KV cache can allocate.
    llm = LLM(
        model=model_name,
        max_model_len=256,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    sentences = load_sentences(args.csv, args.n)
    print(f"Processing {len(sentences)} sentences...")

    structured = StructuredOutputsParams(
        json=schema,
        disable_any_whitespace=True,
        disable_additional_properties=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=128, structured_outputs=structured)

    conversations = [
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": s}]
        for s in sentences
    ]

    outputs = llm.chat(conversations, sampling)

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence, output in zip(sentences, outputs):
            text = output.outputs[0].text
            try:
                scores = json.loads(text)
            except json.JSONDecodeError:
                scores = {"raw": text}
            record = {"sentence": sentence, **scores}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
