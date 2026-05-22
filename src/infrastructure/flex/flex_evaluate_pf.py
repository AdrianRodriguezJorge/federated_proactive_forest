"""FLEX Evaluation primitive procedures for Proactive Forest.

Supports server-side global model assessment and client-side testing of both
local models and deployed global ensembles.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from flex.model import FlexModel
from flex.pool.decorators import evaluate_server_model


def _align_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to ensure y_true and y_pred are comparable.

    Both arrays are mapped to standard label indices or string categories
    to prevent alignment failures.

    Args:
        y_true (np.ndarray): True target labels.
        y_pred (np.ndarray): Predicted target labels.
        class_names (Optional[List[str]]): Class names vocabulary.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Aligned true and predicted label arrays.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if class_names is None or len(class_names) == 0:
        return y_true_arr, y_pred_arr

    class_to_idx = {str(name): i for i, name in enumerate(class_names)}

    # Process y_true
    try:
        if y_true_arr.size > 0 and isinstance(y_true_arr.flat[0], (str, np.str_)):
            y_true_arr = np.array(
                [class_to_idx.get(str(y), 0) for y in y_true_arr]
            )
    except Exception as e:
        logging.warning(f"Failed to align true labels: {e}")

    # Process y_pred
    try:
        if y_pred_arr.size > 0 and isinstance(y_pred_arr.flat[0], (str, np.str_)):
            y_pred_arr = np.array(
                [class_to_idx.get(str(p), 0) for p in y_pred_arr]
            )
    except Exception as e:
        logging.warning(f"Failed to align predicted labels: {e}")

    return y_true_arr, y_pred_arr


@evaluate_server_model
def evaluate_global_pf_model(
    server_flex_model: FlexModel,
    test_data: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate global model on server side.

    FLEX Primitive for evaluate_server_model.

    Args:
        server_flex_model (FlexModel): Server model state wrapper.
        test_data (Any): Dataset locally loaded or fed from server.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, Any]: Performance report with accuracy, f1, and PCD.
    """
    model = server_flex_model.get("model")
    if model is None:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "pcd": 0.0,
            "per_class_metrics": {},
            "confusion_matrix": [],
        }

    try:
        # Extract X and y from various possible formats
        X_test, y_test = None, None

        if hasattr(test_data, "to_numpy"):
            X_test, y_test = test_data.to_numpy()
        elif test_data is not None and isinstance(test_data, np.ndarray):
            X_test = test_data
            y_test = kwargs.get("y_test")

        # Fallback to kwargs
        if X_test is None:
            X_test = kwargs.get("X_test")
        if y_test is None:
            y_test = kwargs.get("y_test")

        if X_test is None or y_test is None:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "pcd": 0.0,
                "per_class_metrics": {},
                "confusion_matrix": [],
            }

        # 1. Predictions
        predictions = model.predict(X_test)

        # Robust label alignment
        class_names = getattr(model, "class_names", None)
        if not class_names:
            config = server_flex_model.get("config", {})
            class_names = config.get("class_names") or config.get(
                "model",
            ).get("class_names")

        y_test_arr, predictions_arr = _align_labels(
            y_test, predictions, class_names
        )

        # Basic metrics
        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(
            f1_score(
                y_test_arr, predictions_arr, average="macro", zero_division=0
            )
        )

        # Detailed metrics
        labels = list(range(len(class_names))) if class_names else None
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_arr,
            predictions_arr,
            average=None,
            labels=labels,
            zero_division=0,
        )
        conf_mat = confusion_matrix(y_test_arr, predictions_arr, labels=labels)

        per_class_metrics = {}
        if class_names:
            for i, name in enumerate(class_names):
                if i < len(prec):
                    per_class_metrics[name] = {
                        "precision": float(prec[i]),
                        "recall": float(rec[i]),
                        "f1": float(f1[i]),
                    }

        # 2. Diversity Metric (PCD)
        pcd = 0.0
        if hasattr(model, "diversity_measure"):
            try:
                pcd = float(
                    model.diversity_measure(
                        X_test, y_test, diversity="pcd"
                    )
                )
            except Exception as div_exc:
                warnings.warn(f"PCD calculation failed: {div_exc}")

        # Store in server model state for persistence
        server_flex_model["global_accuracy"] = accuracy
        server_flex_model["global_f1"] = macro_f1
        server_flex_model["global_pcd"] = pcd

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "pcd": pcd,
            "forest_size": len(server_flex_model.get("trees", [])),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": conf_mat.tolist(),
        }
    except Exception as e:
        logger = logging.getLogger("FLEX_Eval")
        logger.error(f"CRITICAL: evaluate_global_pf_model failed: {e}")
        raise


def evaluate_global_pf_model_at_clients(
    client_flex_model: FlexModel, client_data: Any, *args: Any, **kwargs: Any
) -> Dict[str, float]:
    """Evaluate global model on client side.

    Args:
        client_flex_model (FlexModel): FLEX client model state wrapper.
        client_data (Any): Dataset local to client.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, float]: Accuracy and F1 scores.
    """
    global_model = client_flex_model.get("global_model")
    if global_model is None or client_data is None:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    try:
        X_test, y_test = client_data.to_numpy()
        predictions = global_model.predict(X_test)
        class_names = getattr(global_model, "class_names", None)
        y_test_arr, predictions_arr = _align_labels(
            y_test, predictions, class_names
        )

        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(
            f1_score(
                y_test_arr, predictions_arr, average="macro", zero_division=0
            )
        )

        client_flex_model["global_accuracy"] = accuracy
        client_flex_model["global_f1"] = macro_f1
        return {"accuracy": accuracy, "macro_f1": macro_f1}
    except Exception as e:
        logger = logging.getLogger("FLEX_Eval_Client")
        logger.error(f"Error evaluating global model at client: {e}")
        raise


def evaluate_local_pf_model_at_clients(
    client_flex_model: FlexModel, client_data: Any, *args: Any, **kwargs: Any
) -> Dict[str, float]:
    """Evaluate local model on client side.

    Args:
        client_flex_model (FlexModel): FLEX client model state wrapper.
        client_data (Any): Dataset local to client.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, float]: Accuracy and F1 scores.
    """
    local_model = client_flex_model.get("model")
    if local_model is None or client_data is None:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    try:
        X_test, y_test = client_data.to_numpy()
        predictions = local_model.predict(X_test)
        class_names = getattr(local_model, "class_names", None)
        y_test_arr, predictions_arr = _align_labels(
            y_test, predictions, class_names
        )

        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(
            f1_score(
                y_test_arr, predictions_arr, average="macro", zero_division=0
            )
        )

        client_flex_model["local_accuracy"] = accuracy
        client_flex_model["local_f1"] = macro_f1
        return {"accuracy": accuracy, "macro_f1": macro_f1}
    except Exception as e:
        logger = logging.getLogger("FLEX_Eval_Client")
        logger.error(f"Error evaluating local model at client: {e}")
        raise


__all__ = [
    "evaluate_global_pf_model",
    "evaluate_global_pf_model_at_clients",
    "evaluate_local_pf_model_at_clients",
]