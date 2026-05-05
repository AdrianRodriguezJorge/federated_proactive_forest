from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from src.domain.aggregation.tree_ranker import TreeEntry
from src.domain.metrics.metrics_service import IMetricsService
from src.domain.services.label_service import SimpleLabelService
from src.domain.prediction.voting import calculate_mode

class ProgressiveSelector:
    """
    Handles progressive tree selection with early stopping (convergence).
    Used by S2-S7 and Progressive Windows to reduce duplication.
    """
    def __init__(self, metrics_service: Optional[IMetricsService] = None):
        self.metrics_svc = metrics_service

    def select(
        self,
        candidate_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val_norm: np.ndarray,
        episode_size: int,
        t_max: int,
        convergence_threshold: float,
        label_service: Optional[SimpleLabelService] = None
    ) -> Tuple[List[Any], List[TreeEntry], Optional[int], List[Dict]]:
        """
        Progressively selects trees evaluating against validation data.
        
        Returns:
            - global_trees: Selected trees.
            - selected_entries: Selected entries.
            - convergence_round: Episode index where convergence occurred.
            - round_logs: Evolution logs.
        """
        models_built = 0
        stop_counter = 0
        
        selected_entries: List[TreeEntry] = []
        episode_accuracies = []
        round_logs = []
        convergence_round = None
        prediction_cache: Dict[int, np.ndarray] = {}

        episode_idx = 0
        while models_built < min(len(candidate_entries), t_max):
            episode_idx += 1
            episode_entries = candidate_entries[models_built:models_built + episode_size]
            if not episode_entries:
                break

            for entry in episode_entries:
                selected_entries.append(entry)

            models_built = len(selected_entries)
            predictions_raw = self._predict_ensemble(selected_entries, X_val, cache=prediction_cache)
            predictions = label_service.transform(predictions_raw) if label_service else predictions_raw
            
            if self.metrics_svc:
                acc = float(self.metrics_svc.accuracy_score(y_val_norm, predictions))
                f1 = float(self.metrics_svc.f1_score(y_val_norm, predictions, average='macro'))
            else:
                acc = float(np.mean(predictions == y_val_norm))
                f1 = 0.0
            
            episode_accuracies.append(acc)
            round_logs.append({
                'episode': episode_idx,
                'n_trees': models_built,
                'accuracy': acc,
                'macro_f1': f1
            })

            if len(episode_accuracies) >= 2:
                improvement = episode_accuracies[-1] - episode_accuracies[-2]
                if improvement < convergence_threshold:
                    stop_counter += 1
                    if stop_counter >= 2:
                        convergence_round = episode_idx
                        break
                else:
                    stop_counter = 0

        return [e.tree for e in selected_entries], selected_entries, convergence_round, round_logs

    def _predict_ensemble(self, selected_entries: List[TreeEntry], X: np.ndarray, 
                         cache: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        if not selected_entries:
            return np.zeros(X.shape[0], dtype=int)
        
        n_samples = X.shape[0]
        n_trees = len(selected_entries)
        # Use object dtype to handle both string and numeric predictions before mode calculation
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
