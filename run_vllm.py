import argparse
import csv
import json
import re
from pathlib import Path

from openai import OpenAI

"""
Run the following command to start the vLLM server:
vllm serve Qwen/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 8001 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --enforce-eager

Run the following command to annotate a sentence:
python run_vllm.py   --input_csv v2_3m_final_clean_text.csv   --row_number 20000   --prompt_md prompt_examples.md   --promp
t_type MFT   --output_file outputs/row20000_mft.json   --model_name Qwen/Qwen3.5-0.8B  
 --base_url http://127.0.0.1:8001/v1   --max_tokens 128

"""


MFT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "care": {"type": "integer", "minimum": 0, "maximum": 100},
                "harm": {"type": "integer", "minimum": 0, "maximum": 100},
                "fairness": {"type": "integer", "minimum": 0, "maximum": 100},
                "cheating": {"type": "integer", "minimum": 0, "maximum": 100},
                "loyalty": {"type": "integer", "minimum": 0, "maximum": 100},
                "betrayal": {"type": "integer", "minimum": 0, "maximum": 100},
                "authority": {"type": "integer", "minimum": 0, "maximum": 100},
                "subversion": {"type": "integer", "minimum": 0, "maximum": 100},
                "sanctity": {"type": "integer", "minimum": 0, "maximum": 100},
                "degradation": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": [
                "care", "harm", "fairness", "cheating", "loyalty",
                "betrayal", "authority", "subversion", "sanctity", "degradation"
            ],
            "additionalProperties": False,
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

SHVT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "self_direction_thought": {"type": "integer", "minimum": 0, "maximum": 100},
                "self_direction_action": {"type": "integer", "minimum": 0, "maximum": 100},
                "stimulation": {"type": "integer", "minimum": 0, "maximum": 100},
                "hedonism": {"type": "integer", "minimum": 0, "maximum": 100},
                "achievement": {"type": "integer", "minimum": 0, "maximum": 100},
                "power_dominance": {"type": "integer", "minimum": 0, "maximum": 100},
                "power_resources": {"type": "integer", "minimum": 0, "maximum": 100},
                "face": {"type": "integer", "minimum": 0, "maximum": 100},
                "security_personal": {"type": "integer", "minimum": 0, "maximum": 100},
                "security_societal": {"type": "integer", "minimum": 0, "maximum": 100},
                "tradition": {"type": "integer", "minimum": 0, "maximum": 100},
                "conformity_rules": {"type": "integer", "minimum": 0, "maximum": 100},
                "conformity_interpersonal": {"type": "integer", "minimum": 0, "maximum": 100},
                "humility": {"type": "integer", "minimum": 0, "maximum": 100},
                "benevolence_caring": {"type": "integer", "minimum": 0, "maximum": 100},
                "benevolence_dependability": {"type": "integer", "minimum": 0, "maximum": 100},
                "universalism_concern": {"type": "integer", "minimum": 0, "maximum": 100},
                "universalism_nature": {"type": "integer", "minimum": 0, "maximum": 100},
                "universalism_tolerance": {"type": "integer", "minimum": 0, "maximum": 100},
                "universalism_objectivity": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": [
                "self_direction_thought", "self_direction_action", "stimulation",
                "hedonism", "achievement", "power_dominance", "power_resources",
                "face", "security_personal", "security_societal", "tradition",
                "conformity_rules", "conformity_interpersonal", "humility",
                "benevolence_caring", "benevolence_dependability",
                "universalism_concern", "universalism_nature",
                "universalism_tolerance", "universalism_objectivity"
            ],
            "additionalProperties": False,
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def extract_prompt_from_markdown(md_text: str, prompt_type: str) -> str:
    """
    Extract the code block under either:
    - '## MFT ...'
    - '## SHVT ...'
    """
    prompt_type = prompt_type.upper()

    if prompt_type == "MFT":
        pattern = r"## MFT.*?```text\n(.*?)```"
    elif prompt_type == "SHVT":
        pattern = r"## SHVT.*?```text\n(.*?)```"
    else:
        raise ValueError("prompt_type must be either 'MFT' or 'SHVT'")

    match = re.search(pattern, md_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {prompt_type} prompt inside markdown file.")

    return match.group(1).strip()


def get_schema(prompt_type: str) -> dict:
    prompt_type = prompt_type.upper()
    if prompt_type == "MFT":
        return MFT_SCHEMA
    if prompt_type == "SHVT":
        return SHVT_SCHEMA
    raise ValueError("prompt_type must be either 'MFT' or 'SHVT'")


def get_sentence_from_csv(input_csv: str, row_number: int) -> str:
    """
    row_number is 1-based over DATA rows after skipping the first row.
    Example:
      - row_number=1 -> first sentence after the header/first row
      - row_number=2 -> second sentence after the header/first row
    """
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        # Skip first row
        next(reader, None)

        for i, row in enumerate(reader, start=1):
            if i == row_number:
                if not row:
                    raise ValueError(f"Row {row_number} is empty.")
                sentence = row[0].strip()
                if not sentence:
                    raise ValueError(f"Row {row_number} contains an empty sentence.")
                return sentence

    raise ValueError(f"Requested row_number={row_number} but file has fewer data rows.")


def call_model(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    sentence: str,
    max_tokens: int,
    temperature: float,
    guided_json_schema: dict | None = None,
) -> str:
    extra_body = {}
    if guided_json_schema is not None:
        extra_body["structured_outputs"] = {"json": guided_json_schema}

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body if extra_body else None,
    )

    return response.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV")
    parser.add_argument("--row_number", type=int, required=True,
                        help="1-based row among data rows AFTER skipping the first row")
    parser.add_argument("--prompt_md", required=True, help="Path to prompt_examples.md")
    parser.add_argument("--prompt_type", choices=["MFT", "SHVT"], default="MFT")
    parser.add_argument("--output_file", required=True, help="Path to output JSON file")
    parser.add_argument("--model_name", required=True, help="Model name/path served by vLLM")
    parser.add_argument("--base_url", default="http://localhost:8001/v1",
                        help="vLLM OpenAI-compatible server URL")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable_guided_json", action="store_true",
                        help="Disable vLLM guided_json structured output")

    args = parser.parse_args()

    with open(args.prompt_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    system_prompt = extract_prompt_from_markdown(md_text, args.prompt_type)
    sentence = get_sentence_from_csv(args.input_csv, args.row_number)

    guided_schema = None if args.disable_guided_json else get_schema(args.prompt_type)

    client = OpenAI(
        api_key="EMPTY",
        base_url=args.base_url,
    )

    response_text = call_model(
        client=client,
        model_name=args.model_name,
        system_prompt=system_prompt,
        sentence=sentence,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        guided_json_schema=guided_schema,
    )

    output = {
        "row_number": args.row_number,
        "prompt_type": args.prompt_type,
        "sentence": sentence,
        "response_raw": response_text,
    }

    # Try to parse the model output as JSON, since the prompt expects JSON-only
    try:
        output["response_json"] = json.loads(response_text)
    except Exception:
        output["response_json"] = None

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()