"""Modular prompt builder for experiments with different prompt components."""

import re
from pathlib import Path
from typing import Dict, List


def parse_prompt_markdown(md_file_path: str, prompt_type: str) -> Dict[str, str]:
    """Parse the markdown file and extract prompt components for the given prompt_type."""
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Find the section for the prompt type
    pattern = rf"## {prompt_type.upper()}.*?```text\n(.*?)```"
    match = re.search(pattern, md_content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {prompt_type} prompt in markdown file")

    prompt_content = match.group(1).strip()

    # Split into sections
    sections = {}
    current_section = None
    current_lines = []

    lines = prompt_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Task:'):
            if current_section:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = 'task_instructions'
            current_lines = [line]
        elif line.startswith('Scoring:'):
            if current_section:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = 'scoring_instructions'
            current_lines = [line]
        elif line.startswith('Labels and definitions:'):
            if current_section:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = 'labels_definitions'
            current_lines = [line]
        elif current_section:
            current_lines.append(line)
        i += 1

    if current_section:
        sections[current_section] = '\n'.join(current_lines).strip()

    # Extract introduction (everything before Task:)
    intro_end = prompt_content.find('Task:')
    if intro_end != -1:
        sections['introduction'] = prompt_content[:intro_end].strip()
    else:
        sections['introduction'] = prompt_content

    return sections


def parse_labels_definitions(labels_text: str) -> Dict[str, str]:
    """Parse the labels and definitions section into a dictionary."""
    labels = {}
    lines = labels_text.split('\n')
    for line in lines:
        if line.startswith('- '):
            parts = line[2:].split(': ', 1)
            if len(parts) == 2:
                label, definition = parts
                labels[label.strip()] = definition.strip()
    return labels


# Load components from the markdown file
PROMPT_COMPONENTS = {}

def load_prompt_components(md_file_path: str):
    """Load and parse prompt components from the markdown file."""
    global PROMPT_COMPONENTS
    PROMPT_COMPONENTS.clear()  # Clear previous components
    for prompt_type in ["MFT", "SHVT"]:
        components = parse_prompt_markdown(md_file_path, prompt_type)
        PROMPT_COMPONENTS[prompt_type] = components
        PROMPT_COMPONENTS[prompt_type]['labels_definitions_dict'] = parse_labels_definitions(components.get('labels_definitions', ''))


def get_labels(prompt_type: str) -> List[str]:
    """Get the list of labels for a prompt type."""
    if not PROMPT_COMPONENTS:
        raise RuntimeError("Prompt components not loaded. Call load_prompt_components() first.")
    return list(PROMPT_COMPONENTS[prompt_type]['labels_definitions_dict'].keys())


def build_prompt(
    prompt_type: str,
    include_introduction: bool = True,
    include_task: bool = True,
    include_scoring: bool = True,
    include_labels_definitions: bool = True,
    custom_task_instructions: str = None,
    custom_scoring_instructions: str = None,
) -> str:
    """Build a prompt from modular components."""

    if prompt_type not in PROMPT_COMPONENTS:
        raise ValueError("prompt_type must be MFT or SHVT")

    components = PROMPT_COMPONENTS[prompt_type]

    parts = []

    if include_introduction:
        intro = components.get('introduction', '')
        parts.append(intro)

    if include_task:
        task = custom_task_instructions if custom_task_instructions else components.get('task_instructions', '')
        parts.append(task)

    if include_scoring:
        scoring = custom_scoring_instructions if custom_scoring_instructions else components.get('scoring_instructions', '')
        parts.append(scoring)

    if include_labels_definitions:
        labels_text = components.get('labels_definitions', '')
        parts.append(labels_text)
    else:
        labels = list(components['labels_definitions_dict'].keys())
        labels_text = "Labels:\n" + "\n".join(f"- {label}" for label in labels)
        parts.append(labels_text)

    return '\n\n'.join(part.strip() for part in parts if part.strip())


# Labels will be set after load_prompt_components() is called
MFT_LABELS = []
SHVT_LABELS = []


# Example usage for experiments
if __name__ == "__main__":

    import os
    prompt = os.path.join(os.getcwd(),'prompts', "prompt_examples.md")
    load_prompt_components(md_file_path=prompt)

    # Full prompt
    print("Full MFT prompt:")
    print(build_prompt("MFT"))
    print("\n" + "="*50 + "\n")

    # Without definitions
    print("MFT prompt without definitions:")
    print(build_prompt("MFT", include_labels_definitions=False))
    print("\n" + "="*50 + "\n")

    # Custom scoring (e.g., binary instead of 0-100)
    custom_scoring = """Scoring:
- For each label, assign 1 if present, 0 if not.
- Output ALL labels exactly once.
{"Output schema (STRICT)": "{\\"scores\\": {\\"care\\": <0 or 1>, \\"harm\\": <0 or 1>, ...}}"}
- JSON only."""
    print("MFT prompt with binary scoring:")
    print(build_prompt("MFT", custom_scoring_instructions=custom_scoring, include_labels_definitions=False))