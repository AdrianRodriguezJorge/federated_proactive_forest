"""Base class for global strategies (S2-S4) with Progressive Forest aggregation.

These strategies rank all trees globally and incorporate them progressively
using CPF algorithm with early stopping based on validation data.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from src.domain.services.label_service import SimpleLabelService
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry
from ...prediction.voting import calculate_mode

from ...model.cpf_implementation.estimator import ProactiveForestClassifier
from src.domain.metrics.metrics_service import IMetricsService, IDiversityService


class GlobalProgressiveStrategy(ABC):
    """Base class for global ranking strategies with Progressive Forest.

    Subclasses define the ranking criterion (accuracy, macro-F1, F1+PCD).
    All trees are ranked globally and incorporated progressively in episodes.
    Aggregation stops by:
    1. Convergence (CPF early stopping - 2 consecutive episodes with delta < 0.002)
    2. T_max trees reached (límite máximo de árboles en el modelo global)
    3. All trees from all clients have been added

    Parameters:
    - CONVERGENCE: 0.002 (umbral de convergencia)
    - EPISODE_SIZE: 5 (fijo, según requerimientos)
    - T_MAX: 100 (límite máximo de árboles)
    """

    CONVERGENCE = 0.002
    EPISODE_SIZE = 5
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
        return self.__class__.__name__.replace('Strategy', '').replace('Global', 'S')
    
    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: int = None,
        t_max: int = None,
        **kwargs
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry], Optional[int], List[Dict]]:
        """Aggregate trees using global ranking with Progressive Forest.

        Returns:
            - global_trees: list of progressively selected trees
            - selected_ids: mapping of client -> local indices selected
            - selected_entries: TreeEntry list in selection order
            - convergence_round: episode index where convergence was met
            - round_logs: evolution of metrics per episode
        """
        diversity_svc = self.diversity_svc or kwargs.get('diversity_service')
        entries = TreeRanker.build_entries(client_trees, client_metadata, X_val=X_val, diversity_service=diversity_svc)
        
        criterion = self._get_ranking_criterion(**kwargs)
        ranker = TreeRanker(criterion=criterion, 
                           f1_weight=kwargs.get('f1_weight', 0.5),
                           pcd_weight=kwargs.get('pcd_weight', 0.5),
                           diversity_service=diversity_svc)
        ranked_entries = ranker.rank(entries)
        
        if X_val is None or y_val is None:
            global_trees = [e.tree for e in ranked_entries]
            selected_ids = {cid: [] for cid in client_trees.keys()}
            for entry in ranked_entries:
                selected_ids[entry.client_id].append(entry.tree_local_id)
            return global_trees, selected_ids, ranked_entries, None, []

        # Normalize y_val to numeric indices if class_names available
        class_names = kwargs.get('class_names', [])
        label_svc = SimpleLabelService(class_names) if class_names else None
        y_val_norm = label_svc.transform(y_val) if label_svc else np.asarray(y_val)
        
        # Ensure y_val_norm is not object dtype even if it contains strings
        if y_val_norm.dtype == object:
            try:
                y_val_norm = np.array(y_val_norm.tolist())
            except:
                pass
        
        global_trees, selected_entries, conv_round, logs = self._progressive_selection_with_convergence(
            ranked_entries, X_val, y_val_norm, t_max=t_max, label_service=label_svc, **kwargs
        )
        
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in selected_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)
        
        return global_trees, selected_ids, ranked_entries, conv_round, logs
    
    def _get_ranking_criterion(self, **kwargs) -> RankingCriterion:
        return self.ranking_criterion
    
    def _progressive_selection_with_convergence(
        self,
        ranked_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val: np.ndarray,
        t_max: Optional[int] = None,
        label_service: Optional[SimpleLabelService] = None,
        **kwargs
    ) -> Tuple[List[Any], List[TreeEntry], Optional[int], List[Dict]]:
        """Select trees progressively with early stopping. Episode size is fixed at 5."""
        EPISODE = self.EPISODE_SIZE
        models_built = 0
        stop_counter = 0
        
        selected_entries: List[TreeEntry] = []
        episode_accuracies = []
        round_logs = []
        convergence_round = None

        T_MAX = t_max if t_max is not None else self.T_MAX

        episode_idx = 0
        while models_built < min(len(ranked_entries), T_MAX):
            episode_idx += 1
            episode_entries = ranked_entries[models_built:models_built + EPISODE]
            if not episode_entries:
                break

            for entry in episode_entries:
                selected_entries.append(entry)

            models_built = len(selected_entries)
            predictions_raw = self._predict_ensemble(selected_entries, X_val)
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
    
    def _predict_ensemble(self, selected_entries: List[TreeEntry], X: np.ndarray) -> np.ndarray:
        if not selected_entries:
            return np.zeros(X.shape[0], dtype=int)
        
        n_trees = len(selected_entries)
        # Use object dtype to handle both string and numeric predictions before mode calculation
        all_predictions = np.empty((X.shape[0], n_trees), dtype=object)
        for j, entry in enumerate(selected_entries):
            all_predictions[:, j] = entry.tree.predict(X)
        
        return calculate_mode(all_predictions, axis=1)


class S2GlobalAccuracyStrategy(GlobalProgressiveStrategy):
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.ACCURACY
    @property
    def strategy_id(self) -> str:
        return "S2"


class S3GlobalF1Strategy(GlobalProgressiveStrategy):
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.MACRO_F1
    @property
    def strategy_id(self) -> str:
        return "S3"


class S4GlobalF1PCDStrategy(GlobalProgressiveStrategy):
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.F1_PCD
    @property
    def strategy_id(self) -> str:
        return "S4"
