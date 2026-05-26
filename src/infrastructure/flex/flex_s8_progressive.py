"""FLEX primitives for S8 (Roulette-based) progressive window training.

Handles episodic fitting, convergence tracking, and feature probabilities
routing on clients under FLEX S8 strategy.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from flex.model import FlexModel
from flex.pool.decorators import collect_clients_weights


def train_window_pf_s8(
    client_flex_model: FlexModel,
    client_data: Any,
    active_ids: Optional[List[str]] = None,
) -> FlexModel:
    """Trains a window of W trees on a client, preserving Proactive Forest state.

    Skips training if the client is not in the active_ids list.

    Args:
        client_flex_model (FlexModel): FLEX client model state.
        client_data (Any): Dataset local to client.
        active_ids (Optional[List[str]]): Active client IDs in this round.

    Returns:
        FlexModel: Updated client model state.
    """
    actor_id = str(getattr(client_flex_model, "actor_id", "unknown"))
    if active_ids is not None:
        active_ids_str = [str(aid) for aid in active_ids]
        if actor_id not in active_ids_str:
            return client_flex_model

    from sklearn.model_selection import train_test_split
    from src.domain.metrics.forest_evaluator import ForestEvaluator
    from src.domain.model.proactive_forest import ProactiveForest
    from src.domain.model.progressive_forest import (
        ComparativeProgressiveForest,
    )
    from src.infrastructure.metrics.sklearn_metrics_service import (
        SklearnMetricsService,
    )

    X, y = client_data.to_numpy()
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    except Exception as e:
        import logging
        logging.warning(f"Stratified split failed, falling back to non-stratified: {e}")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    config = client_flex_model.get("config", {})
    window_size = config.get("aggregation", {}).get("window_size", 5)

    pf = client_flex_model.get("model")
    if pf is None:
        class_names = config.get("model", {}).get("class_names", [])
        pf = ProactiveForest(
            n_estimators=config.get("n_estimators", 100),
            alpha_pf=config.get("model", {}).get("alpha_pf", config.get("alpha_pf", 0.1)),
            class_names=class_names,
            local_convergence_threshold=config.get("model", {}).get(
                "local_convergence_threshold", 0.002
            ),
        )
        pf._is_fitted = True
        pf._cpf = ComparativeProgressiveForest(
            pf._classifier, local_convergence_threshold=pf.local_convergence_threshold
        )
        pf._classifier._n_instances, pf._classifier._n_features = X_train.shape
        all_labels = np.unique(np.concatenate([y_train, y_val]))
        pf._classifier.classes_ = all_labels
        pf._classifier._encoder_dict = {
            val: idx for idx, val in enumerate(all_labels)
        }
        pf._classifier._decoder_dict = {
            idx: val for idx, val in enumerate(all_labels)
        }
        pf._classifier._n_classes = len(all_labels)

    classifier = pf._classifier

    prev_meta = client_flex_model.get("metadata", {})
    if not isinstance(prev_meta, dict):
        prev_meta = {}

    stop_counter = prev_meta.get("stop_counter", 0)
    prev_episode_acc = prev_meta.get("prev_episode_acc")
    rounds_completed = prev_meta.get("rounds_completed", 0) + 1
    min_rounds = config.get("aggregation", {}).get("min_rounds", 5)

    # Build trees
    classifier.buildEpisode(X_train, y_train, X_val, y_val, window_size)

    # Calculate convergence using CPF logic
    min_acc = (
        min(classifier._m_progressive_accuracy)
        if classifier._m_progressive_accuracy
        else 0
    )
    max_acc = (
        max(classifier._m_progressive_accuracy)
        if classifier._m_progressive_accuracy
        else 0
    )
    episode_acc = max_acc - min_acc

    has_converged = False
    acc_diff = 0.002
    if prev_episode_acc is not None:
        acc_diff = episode_acc - prev_episode_acc

    if (
        acc_diff < pf.local_convergence_threshold
        or episode_acc < pf.local_convergence_threshold
    ):
        stop_counter += 1
        if stop_counter >= 2 and rounds_completed >= min_rounds:
            has_converged = True
    else:
        stop_counter = 0

    client_flex_model.update(
        {
            "model": pf,
            "metadata": {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "n_trees": len(pf.get_trees()),
                "pcd": 0.0,
                "has_converged": has_converged,
                "stop_counter": stop_counter,
                "prev_episode_acc": episode_acc,
                "rounds_completed": rounds_completed,
            },
            "X_train": X_train,
        }
    )

    return client_flex_model


def check_convergence_s8(
    client_flex_model: FlexModel, *args: Any, **kwargs: Any
) -> bool:
    """Returns True if the client model has converged.

    Args:
        client_flex_model (FlexModel): Client model state wrapper.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        bool: True if converged, False otherwise.
    """
    meta = client_flex_model.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get("has_converged", False)
    return False
