"""Base class for global strategies (S2-S4) with Progressive Forest aggregation.

These strategies rank all trees globally and incorporate them progressively
using CPF algorithm with early stopping based on validation data.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
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
    - INITIAL_EPISODE: 5 (tamaño inicial del episodio, k en la tesis)
    - T_MAX: 100 (límite máximo de árboles, mismo que n_estimators en clientes)
    """

    CONVERGENCE = 0.002
    INITIAL_EPISODE = 5
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
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry]]:
        """Aggregate trees using global ranking with Progressive Forest.

        Args:
            client_trees: Dict mapping client_id to list of trees
            client_metadata: Dict mapping client_id to metadata
            X_val: Validation features for convergence checking
            y_val: Validation labels for convergence checking
            max_trees: NOT used for stopping in global strategies (S2-S4)
            t_max: Maximum number of trees in global model (default: 100)
            **kwargs: Strategy-specific parameters (e.g., f1_weight, pcd_weight)

        Returns:
            - global_trees: list of progressively selected trees
            - selected_ids: mapping of client -> local indices selected
            - selected_entries: TreeEntry list in selection order
        """
        # Build entries for all trees
        diversity_svc = self.diversity_svc or kwargs.get('diversity_service')
        entries = TreeRanker.build_entries(client_trees, client_metadata, X_val=X_val, diversity_service=diversity_svc)
        
        # Get ranking criterion (may use kwargs for F1+PCD)
        criterion = self._get_ranking_criterion(**kwargs)
        ranker = TreeRanker(criterion=criterion, 
                           f1_weight=kwargs.get('f1_weight', 0.5),
                           pcd_weight=kwargs.get('pcd_weight', 0.5),
                           diversity_service=diversity_svc)
        ranked_entries = ranker.rank(entries)
        
        # If no validation data, return all ranked trees (no PF early stopping)
        if X_val is None or y_val is None:
            global_trees = [e.tree for e in ranked_entries]
            selected_ids = {cid: [] for cid in client_trees.keys()}
            for entry in ranked_entries:
                selected_ids[entry.client_id].append(entry.tree_local_id)
            return global_trees, selected_ids, ranked_entries
        
        # Apply Progressive Forest with early stopping using validation data
        global_trees, selected_entries = self._progressive_selection_with_convergence(
            ranked_entries, X_val, y_val, t_max=t_max
        )
        
        # Build selected_ids dictionary
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in selected_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)
        
        return global_trees, selected_ids, selected_entries
    
    def _get_ranking_criterion(self, **kwargs) -> RankingCriterion:
        """Get the ranking criterion, allowing subclasses to override."""
        return self.ranking_criterion
    
    def _progressive_selection_with_convergence(
        self,
        ranked_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val: np.ndarray,
        t_max: int = None
    ) -> Tuple[List[Any], List[TreeEntry]]:
        """Select trees progressively using CPF algorithm with early stopping.

        Aggregation stops by:
        1. Convergence (2 consecutive episodes with accuracy delta < CONVERGENCE)
        2. T_MAX trees reached (límite máximo de árboles)
        3. All trees from all clients have been added

        Args:
            ranked_entries: Trees ranked by criterion
            X_val: Validation features
            y_val: Validation labels
            t_max: Maximum number of trees (default: class T_MAX = 100)

        Returns:
            Tuple of (selected_trees, selected_entries)
        """
        EPISODE = self.INITIAL_EPISODE
        models_built = 0
        stop_counter = 0
        previous_episode_accuracy = None
        episode_accuracy_dif = 0.002

        selected_entries: List[TreeEntry] = []
        episode_accuracies = []

        # Use provided t_max or default to class constant
        T_MAX = t_max if t_max is not None else self.T_MAX

        # Progressive selection: stop by convergence, T_MAX, or all trees added
        while models_built < min(len(ranked_entries), T_MAX):
            # Build an episode of trees
            episode_entries = ranked_entries[models_built:models_built + EPISODE]
            if not episode_entries:
                break

            # Add trees from this episode
            for entry in episode_entries:
                selected_entries.append(entry)

            models_built = len(selected_entries)

            # Evaluate ensemble accuracy on validation set
            predictions = self._predict_ensemble(selected_entries, X_val)
            
            # Use injected metrics service or fallback to kwargs
            metrics_svc = self.metrics_svc or kwargs.get('metrics_service')
            if metrics_svc:
                acc = metrics_svc.accuracy_score(y_val, predictions)
            else:
                # Minimal fallback if not injected
                acc = np.mean(predictions == y_val)
                
            episode_accuracies.append(acc)

            # Check convergence after first episode
            if len(episode_accuracies) >= 2:
                # Calculate marginal improvement
                current_acc = episode_accuracies[-1]
                previous_acc = episode_accuracies[-2]
                improvement = current_acc - previous_acc

                # Convergence: small change
                if improvement < self.CONVERGENCE:
                    stop_counter += 1
                    EPISODE = max(1, EPISODE - 1)
                    if stop_counter >= 2:
                        break
                else:
                    stop_counter = 0
                    EPISODE += 1

        return [e.tree for e in selected_entries], selected_entries
    
    def _predict_ensemble(self, selected_entries: List[TreeEntry], X: np.ndarray) -> np.ndarray:
        """Make predictions using simple majority voting from selected trees.
        
        Args:
            selected_entries: List of selected tree entries
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predicted class labels
        """
        if not selected_entries:
            return np.zeros(X.shape[0], dtype=int)
        
        n_samples = X.shape[0]
        n_trees = len(selected_entries)
        
        # Collect predictions from all trees using vectorized predict(X) method
        all_predictions = np.zeros((X.shape[0], n_trees), dtype=int)
        for j, entry in enumerate(selected_entries):
            all_predictions[:, j] = entry.tree.predict(X)
        
        # Majority voting per sample
        return calculate_mode(all_predictions, axis=1)


class S2GlobalAccuracyStrategy(GlobalProgressiveStrategy):
    """Strategy S2: Global Accuracy with Progressive Forest
    
    Ranks all trees globally by accuracy and incorporates them progressively
    using CPF algorithm with early stopping.
    """
    
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.ACCURACY
    
    @property
    def strategy_id(self) -> str:
        return "S2"


class S3GlobalF1Strategy(GlobalProgressiveStrategy):
    """Strategy S3: Global Macro-F1 with Progressive Forest
    
    Ranks all trees globally by macro-F1 and incorporates them progressively
    using CPF algorithm with early stopping.
    """
    
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.MACRO_F1
    
    @property
    def strategy_id(self) -> str:
        return "S3"


class S4GlobalF1PCDStrategy(GlobalProgressiveStrategy):
    """Strategy S4: Global F1 + PCD with Progressive Forest
    
    Ranks all trees globally by weighted combination of macro-F1 and PCD,
    and incorporates them progressively using CPF algorithm with early stopping.
    """
    
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.F1_PCD
    
    @property
    def strategy_id(self) -> str:
        return "S4"
