"""Progressive tree selector service.

Implements sequential validation checks on subset candidate entries, checking
improvement convergence limits to perform early stopping and dynamic proactive
disagreement re-ranking.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.domain.aggregation.tree_ranker import TreeEntry, TreeRanker
from src.domain.metrics.metrics_service import IDiversityService, IMetricsService
from src.domain.prediction.voting import calculate_mode
from src.domain.services.label_service import SimpleLabelService


class ProgressiveSelector:
    """Handles progressive tree selection with early stopping (convergence).

    Supports proactive re-ranking based on Marginal PCD (Cepero 2023).
    """

    def __init__(
        self,
        metrics_service: Optional[IMetricsService] = None,
        diversity_service: Optional[IDiversityService] = None,
    ):
        """Initializes ProgressiveSelector.

        Args:
            metrics_service (Optional[IMetricsService]): Performance metrics.
            diversity_service (Optional[IDiversityService]): Diversity metrics.
        """
        self.metrics_svc = metrics_service
        self.diversity_svc = diversity_service

    def select(
        self,
        candidate_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val_norm: np.ndarray,
        episode_size: int,
        t_max: int,
        convergence_threshold: float,
        label_service: Optional[SimpleLabelService] = None,
        ranker: Optional[TreeRanker] = None,
    ) -> Tuple[
        List[Any], List[TreeEntry], Optional[int], List[Dict[str, Any]]
    ]:
        """Progressively selects trees evaluating against validation data.

        If a ranker is provided, it performs proactive re-ranking at each
        episode using dynamic marginal PCD.

        Args:
            candidate_entries (List[TreeEntry]): Candidate tree wrappers.
            X_val (np.ndarray): Validation features.
            y_val_norm (np.ndarray): Normalised true target validation labels.
            episode_size (int): Tree count to add in each episode.
            t_max (int): Max ensemble size.
            convergence_threshold (float): Required accuracy improvement.
            label_service (Optional[SimpleLabelService]): Transform encoder.
            ranker (Optional[TreeRanker]): Optional proactive re-sorter.

        Returns:
            Tuple: A tuple containing:
                - List[Any]: Trees in selected ensemble.
                - List[TreeEntry]: Selected TreeEntry instances.
                - Optional[int]: Episode index when convergence was reached.
                - List[Dict[str, Any]]: Performance logs per episode.
        """
        stop_counter = 0
        selected_entries: List[TreeEntry] = []
        episode_accuracies = []
        round_logs = []
        convergence_round = None
        prediction_cache: Dict[int, np.ndarray] = {}

        n_val_samples = X_val.shape[0]
        global_hits_per_sample = np.zeros(n_val_samples, dtype=int)

        remaining_candidates = candidate_entries.copy()
        episode_idx = 0

        while len(selected_entries) < t_max and remaining_candidates:
            episode_idx += 1

            # PROACTIVE RE-RANKING (marginal PCD)
            if (
                ranker
                and self.diversity_svc
                and ranker.criterion.value == "f1_pcd"
            ):
                for entry in remaining_candidates:
                    tree_id = id(entry.tree)
                    if tree_id not in prediction_cache:
                        prediction_cache[tree_id] = entry.tree.predict(X_val)

                    preds = prediction_cache[tree_id]
                    entry.pcd = self.diversity_svc.calculate_marginal_pcd(
                        candidate_predictions=preds,
                        current_hits_per_sample=global_hits_per_sample,
                        n_existing_trees=len(selected_entries),
                        y_true=y_val_norm,
                    )
                remaining_candidates = ranker.rank(remaining_candidates)

            # Pick next episode
            current_episode = remaining_candidates[:episode_size]
            remaining_candidates = remaining_candidates[episode_size:]

            if not current_episode:
                break

            for entry in current_episode:
                selected_entries.append(entry)
                tree_id = id(entry.tree)
                if tree_id not in prediction_cache:
                    prediction_cache[tree_id] = entry.tree.predict(X_val)

                preds_raw = prediction_cache[tree_id]
                if label_service:
                    preds = label_service.transform(preds_raw)
                else:
                    preds = preds_raw
                global_hits_per_sample += (preds == y_val_norm).astype(int)

            # Evaluate current ensemble
            raw_predictions = self._predict_ensemble(
                selected_entries, X_val, cache=prediction_cache
            )
            if label_service:
                predictions = label_service.transform(raw_predictions)
            else:
                predictions = raw_predictions

            if self.metrics_svc:
                acc = float(
                    self.metrics_svc.accuracy_score(y_val_norm, predictions)
                )
                f1 = float(
                    self.metrics_svc.f1_score(
                        y_val_norm, predictions, average="macro"
                    )
                )
            else:
                acc = float(np.mean(predictions == y_val_norm))
                f1 = 0.0

            episode_accuracies.append(acc)
            round_logs.append(
                {
                    "episode": episode_idx,
                    "n_trees": len(selected_entries),
                    "accuracy": acc,
                    "macro_f1": f1,
                }
            )

            # Check convergence
            if len(episode_accuracies) >= 2:
                improvement = episode_accuracies[-1] - episode_accuracies[-2]
                if improvement < convergence_threshold:
                    stop_counter += 1
                    if stop_counter >= 2:
                        convergence_round = episode_idx
                        break
                else:
                    stop_counter = 0

        selected_trees = [e.tree for e in selected_entries]
        return (
            selected_trees,
            selected_entries,
            convergence_round,
            round_logs,
        )

    def _predict_ensemble(
        self,
        selected_entries: List[TreeEntry],
        X: np.ndarray,
        cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Helper to get ensemble predictions using majority voting mode."""
        if not selected_entries:
            return np.zeros(X.shape[0], dtype=int)

        n_samples = X.shape[0]
        n_trees = len(selected_entries)
        all_predictions = np.empty((n_samples, n_trees), dtype=object)
        for j, entry in enumerate(selected_entries):
            tree_id = id(entry.tree)
            if cache is not None and tree_id in cache:
                all_predictions[:, j] = cache[tree_id]
            else:
                preds = entry.tree.predict(X)
                all_predictions[:, j] = preds
                if cache is not None:
                    cache[tree_id] = preds

        return calculate_mode(all_predictions, axis=1)
