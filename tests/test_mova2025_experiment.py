from pathlib import Path

from llms_experiments.configuration import load_config, validate_config
from llms_experiments.orchestration import run_matrix


def test_mova2025_config_validation():
    config_path = Path("experiments/mova2025.yaml")
    assert config_path.exists(), "experiments/mova2025.yaml must exist"
    config = load_config(config_path, check_files=False)
    validate_config(config, check_files=False)
    assert config["run"]["id"] == "mova2025"
    assert len(config["datasets"]) == 5


def test_mova2025_dataset_ids_and_prompt_assets():
    config_path = Path("experiments/mova2025.yaml")
    config = load_config(config_path, check_files=False)
    expected_ids = {"mova_mft", "mova_mac", "mova_values10", "mova_values20", "mova_common_morality"}
    dataset_ids = {ds["id"] for ds in config["datasets"]}
    assert expected_ids == dataset_ids

    for ds in config["datasets"]:
        source = ds["input"]
        prompt_parts = source.get("prompt_parts", {})
        for name, rel_path in prompt_parts.items():
            path = Path(rel_path)
            assert path.exists(), f"Prompt part {name} ({rel_path}) for {ds['id']} must exist"


def test_mova_full_matrix_config_validation():
    config = load_config("experiments/mova_full_matrix.yaml", check_files=False)
    validate_config(config, check_files=False)
    assert config["run"]["id"] == "mova_full_matrix"
    assert len(config["datasets"]) == 11


def test_mova2025_run_smoke(tmp_path):
    config_path = Path("experiments/mova2025.yaml")
    data_path = tmp_path / "demo.csv"
    data_path.write_text("paragraph_id,vtext\nexample,Helping a neighbour is good.\n", encoding="utf-8")
    output_dir = tmp_path / "mova2025_results"
    overrides = [
        "model.backend=fake",
        "model.name=fake",
        f"output.directory={output_dir}",
    ]
    overrides.extend(f"datasets.{index}.input.path={data_path}" for index in range(5))
    config = load_config(config_path, overrides=overrides)
    summary = run_matrix(config, row_limit=1)
    assert len(summary["datasets"]) == 5
