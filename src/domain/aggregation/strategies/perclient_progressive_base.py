"""Base class for per-client strategies (S5-S7) with Progressive Forest aggregation.

These strategies rank trees within each client independently and incorporate them
using round-robin scheduling with Progressive Forest early stopping.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry
from ...prediction.voting import calculate_mode
from ...model.cpf_implementation.estimator import ProactiveForestClassifier
from src.domain.metrics.metrics_service import IMetricsService, IDiversityService
from src.domain.services.label_service import SimpleLabelService


class PerClientProgressiveStrategy(ABC):
    """Base class for per-client ranking strategies with Progressive Forest.

    Each client's trees are ranked independently, then incorporated via round-robin.
    EPISODE size is fixed to the number of clients (W), adding one tree per client each time.
    
    Aggregation stops by:
    1. Convergence (2 consecutive episodes with improvement < CONVERGENCE)
    2. T_MAX trees reached
    3. All trees exhausted
    """

    CONVERGENCE = 0.002
    T_MAX = 100

    def __init__(self, 
                 metrics_service: Optional[IMetricsService] = None,
                 diversity_service: Optional[IDiversityService] = None):
        self.metrics_svc = metrics_service
        self.diversity_svc = diversity_service
    
    @property
    @abstractmethod
    def ranking_criterion(self) -> RankingCriterion:
        """Return the ranking criterion for this strategy."""
        pass
    
    @property
    def strategy_id(self) -> str:
        return self.__class__.__name__.replace('Strategy', '').replace('PerClient', 'S')
    
    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: int = None,
        max_trees_per_client: int = None,
        t_max: int = None,
        **kwargs
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry], Optional[int], List[Dict]]:
        """Aggregate trees using per-client ranking with round-robin and Progressive Forest."""
        client_ids = list(client_trees.keys())
        if not client_ids:
            return [], {}, [], None, []
        
        if max_trees_per_client is None and max_trees is not None:
            max_trees_per_client = max(1, max_trees // len(client_ids))
        
        # Step 1: Rank trees within each client independently
        client_ranked_entries: Dict[str, List[TreeEntry]] = {}
        criterion = self._get_ranking_criterion(**kwargs)
        diversity_svc = self.diversity_svc or kwargs.get('diversity_service')
        ranker = TreeRanker(criterion=criterion,
                           f1_weight=kwargs.get('f1_weight', 0.5),
                           pcd_weight=kwargs.get('pcd_weight', 0.5),
                           diversity_service=diversity_svc)
        
        for client_id, trees in client_trees.items():
            meta = client_metadata[client_id]
            entries = TreeRanker.build_entries({client_id: trees}, {client_id: meta}, X_val=X_val, diversity_service=diversity_svc)
            ranked = ranker.rank(entries)
            if max_trees_per_client is not None:
                ranked = ranked[:max_trees_per_client]
            client_ranked_entries[client_id] = ranked
        
        # Step 2: Interleave trees from all clients using round-robin
        round_robin_entries = self._interleave_round_robin(client_ranked_entries, client_ids)
        
        if X_val is None or y_val is None:
            global_trees = [e.tree for e in round_robin_entries]
            selected_ids = {cid: [] for cid in client_trees.keys()}
            for entry in round_robin_entries:
                selected_ids[entry.client_id].append(entry.tree_local_id)
            return global_trees, selected_ids, round_robin_entries, None, []

        # Normalize y_val to numeric indices if class_names available
        class_names = kwargs.get('class_names', [])
        label_svc = SimpleLabelService(class_names) if class_names else None
        y_val_norm = label_svc.transform(y_val) if label_svc else np.asarray(y_val)
        
        # Ensure y_val_norm is not object dtype
        if y_val_norm.dtype == object:
            try:
                y_val_norm = np.array(y_val_norm.tolist())
            except:
                pass
        
        # Step 3: Apply Progressive Forest with early stopping
        global_trees, selected_entries, conv_round, logs = self._progressive_selection_with_convergence(
            round_robin_entries, X_val, y_val_norm, t_max=t_max, n_clients=len(client_ids), label_service=label_svc, **kwargs
        )
        
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in selected_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)
        
        return global_trees, selected_ids, round_robin_entries, conv_round, logs
    
    def _get_ranking_criterion(self, **kwargs) -> RankingCriterion:
        return self.ranking_criterion
    
    def _interleave_round_robin(self, client_ranked_entries: Dict[str, List[TreeEntry]],
                                 client_ids: List[str]) -> List[TreeEntry]:
        round_robin_entries = []
        client_indices = {cid: 0 for cid in client_ids}
        max_trees_any_client = max(len(entries) for entries in client_ranked_entries.values()) if client_ranked_entries else 0
        
        for _ in range(max_trees_any_client):
            for client_id in client_ids:
                entries = client_ranked_entries[client_id]
                idx = client_indices[client_id]
                if idx < len(entries):
                    round_robin_entries.append(entries[idx])
                    client_indices[client_id] += 1
        return round_robin_entries
    
    def _progressive_selection_with_convergence(
        self,
        round_robin_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val: np.ndarray,
        t_max: int = None,
        n_clients: int = None,
        label_service: Optional[SimpleLabelService] = None,
        **kwargs
    ) -> Tuple[List[Any], List[TreeEntry], Optional[int], List[Dict]]:
        """Select trees progressively. EPISODE is fixed at n_clients."""
        EPISODE = n_clients if n_clients is not None else 5
        models_built = 0
        stop_counter = 0
        
        selected_entries: List[TreeEntry] = []
        episode_accuracies = []
        round_logs = []
        convergence_round = None

        T_MAX = t_max if t_max is not None else self.T_MAX
        prediction_cache: Dict[int, np.ndarray] = {}

        episode_idx = 0
        while models_built < min(len(round_robin_entries), T_MAX):
            episode_idx += 1
            episode_entries = round_robin_entries[models_built:models_built + EPISODE]
            if not episode_entries:
                break

            for entry in episode_entries:
                selected_entries.append(entry)

            models_built = len(selected_entries)
            predictions_raw = self._predict_ensemble(selected_entries, X_val, cache=prediction_cache)
            predictions = label_service.transform(predictions_raw) if label_service else predictions_raw
            
            metrics_svc = self.metrics_svc or kwargs.get('metrics_service')
            acc = float(metrics_svc.accuracy_score(y_val, predictions)) if metrics_svc else float(np.mean(predictions == y_val))
            f1 = float(metrics_svc.f1_score(y_val, predictions, average='macro')) if metrics_svc else 0.0
            
            episode_accuracies.append(acc)
            round_logs.append({
                'episode': episode_idx,
                'n_trees': models_built,
                'accuracy': acc,
                'macro_f1': f1
            })

            if len(episode_accuracies) >= 2:
                improvement = episode_accuracies[-1] - episode_accuracies[-2]
                convergence_threshold = kwargs.get('convergence_threshold', self.CONVERGENCE)
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


class S5PerClientAccuracyStrategy(PerClientProgressiveStrategy):
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.ACCURACY
    @property
    def strategy_id(self) -> str:
        return "S5"


class S6PerClientF1Strategy(PerClientProgressiveStrategy):
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.MACRO_F1
    @property
    def strategy_id(self) -> str:
        return "S6"


class S7PerClientF1PCDStrategy(PerClientProgressiveStrategy):
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.F1_PCD
    @property
    def strategy_id(self) -> str:
        return "S7"
