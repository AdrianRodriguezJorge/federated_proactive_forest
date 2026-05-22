import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def save_experiment_results_json(
    output_path: str,
    experiment_metadata: Dict[str, Any],
    records: Iterable[Dict[str, Any]],
) -> None:
    """Save experiment results to a structured JSON file.

    The output JSON is designed to preserve metadata and make experiments
    comparable across runs.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "experiment": {
            "script_name": experiment_metadata.get("script_name", "unknown"),
            "experiment_type": experiment_metadata.get("experiment_type", "unknown"),
            "objective": experiment_metadata.get("objective", "unknown"),
            "dataset_config": experiment_metadata.get("dataset_config", {}),
            "common_config": experiment_metadata.get("common_config", {}),
            "created_by": experiment_metadata.get("created_by", "script"),
        },
        "records": list(records),
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
