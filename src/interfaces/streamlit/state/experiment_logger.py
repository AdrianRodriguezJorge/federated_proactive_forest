"""Experiment result logger for Streamlit UI.

Saves evaluation metrics (global + per-client) to CSV files organized by
dataset × strategy combination. Each combination overwrites its previous log.

Directory: logs/streamlit/
Filename:  {dataset_type}_{strategy_key}.csv
Max files: len(datasets) × 8 strategies
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Project root → logs/streamlit/
_LOG_DIR = Path(__file__).resolve().parents[4] / "logs" / "streamlit"


def _sanitize(name: str) -> str:
    """Lowercase and replace non-alphanumeric chars with underscores."""
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _build_log_path(dataset_type: str, strategy_key: str) -> Path:
    """Build the CSV path for a dataset/strategy combination."""
    ds = _sanitize(dataset_type)
    st = _sanitize(strategy_key)
    return _LOG_DIR / f"{ds}_{st}.csv"


def save_experiment_log(
    results: Any,
    config: Dict[str, Any],
) -> Optional[Path]:
    """
    Save evaluation results to a CSV log file.

    The CSV mirrors the comparison table from page_metrics:
    Modelo | Accuracy | Macro-F1 | Macro Precision | Macro Recall | PCD | Tamaño Bosque

    Plus experiment metadata columns:
    timestamp | dataset | strategy | n_clients | use_weighted | local_weight |
    n_estimators | seed | convergence_round

    Args:
        results: FLResults from the federated round.
        config: Full experiment configuration dict.

    Returns:
        Path to the saved CSV, or None on error.
    """
    try:
        dataset_type = config.get("dataset", {}).get("type", "unknown")
        strategy_key = config.get("aggregation", {}).get("strategy", "unknown")
        log_path = _build_log_path(dataset_type, strategy_key)

        # Ensure directory exists
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

        # ── Build rows (same logic as page_metrics comparison table) ──────
        rows: List[Dict[str, Any]] = []

        # Shared metadata for every row
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_cols = {
            "timestamp": ts,
            "dataset": dataset_type,
            "strategy": strategy_key,
            "n_clients": config.get("federation", {}).get("n_clients", ""),
            "use_weighted": config.get("prediction", {}).get("use_weighted", True),
            "local_weight": config.get("prediction", {}).get("local_weight", 0.4),
            "n_estimators": config.get("model", {}).get("n_estimators", ""),
            "seed": config.get("seed", ""),
            "convergence_round": getattr(results, "convergence_round", None),
        }

        # 1. Global model
        global_report = results.global_report
        rows.append({
            **meta_cols,
            "modelo": "Global",
            "accuracy": round(global_report.accuracy, 4),
            "macro_f1": round(global_report.macro_f1, 4),
            "macro_precision": round(global_report.macro_precision, 4),
            "macro_recall": round(global_report.macro_recall, 4),
            "pcd": round(global_report.pcd, 4),
            "forest_size": global_report.forest_size,
        })

        # 2. Per-client (hybrid predictions)
        for cid in results.client_ids:
            report = results.client_reports.get(cid) or results.client_reports.get(str(cid))
            meta = results.client_metadata.get(cid) or results.client_metadata.get(str(cid))

            if report:
                if meta:
                    if isinstance(meta, dict):
                        pcd_val = round(meta.get('pcd', 0.0), 4)
                    else:
                        pcd_val = round(getattr(meta, 'pcd', 0.0), 4)
                else:
                    pcd_val = 0.0
                rows.append({
                    **meta_cols,
                    "modelo": f"Client_{cid}",
                    "accuracy": round(report.accuracy, 4),
                    "macro_f1": round(report.macro_f1, 4),
                    "macro_precision": round(report.macro_precision, 4),
                    "macro_recall": round(report.macro_recall, 4),
                    "pcd": pcd_val,
                    "forest_size": report.forest_size,
                })

        # ── Write CSV (overwrite) ─────────────────────────────────────────
        df = pd.DataFrame(rows)

        # Column order: metadata first, then metrics
        col_order = [
            "timestamp", "dataset", "strategy", "modelo",
            "accuracy", "macro_f1", "macro_precision", "macro_recall",
            "pcd", "forest_size",
            "n_clients", "use_weighted", "local_weight",
            "n_estimators", "seed", "convergence_round",
        ]
        # Only keep columns that exist
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]

        df.to_csv(log_path, index=False, encoding="utf-8")
        logger.info("Experiment log saved to %s (%d rows)", log_path, len(df))
        return log_path

    except Exception as e:
        logger.error("Failed to save experiment log: %s", e, exc_info=True)
        return None
