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



class PerClientProgressiveStrategy(ABC):
    """Base class for per-client ranking strategies with Progressive Forest.

    Subclasses define the ranking criterion (accuracy, macro-F1, F1+PCD).
    Each client's trees are ranked independently, then incorporated via round-robin.
    
    Aggregation stops by:
    1. Convergence (CPF early stopping - 2 consecutive episodes with improvement < CONVERGENCE)
    2. T_MAX trees reached (límite máximo de árboles en el modelo global)
    3. All trees from all clients have been added
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
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry]]:
        """Aggregate trees using per-client ranking with round-robin and Progressive Forest.

        Args:
            client_trees: Dict mapping client_id to list of trees
            client_metadata: Dict mapping client_id to metadata
            X_val: Validation features for convergence checking
            y_val: Validation labels for convergence checking
            max_trees: If provided, distributed evenly among clients
            max_trees_per_client: Maximum trees per client before round-robin
            t_max: Maximum number of trees in global model (default: 100)
            **kwargs: Strategy-specific parameters (e.g., f1_weight, pcd_weight)

        Returns:
            - global_trees: list of progressively selected trees (round-robin order)
            - selected_ids: mapping of client -> local indices selected
            - selected_entries: TreeEntry list in selection order
        """
        client_ids = list(client_trees.keys())
        if not client_ids:
            return [], {}, []
        
        # Determine max trees per client
        if max_trees_per_client is None and max_trees is not None:
            # Distribute max_trees evenly among clients
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
            # Use TreeRanker's unified logic to build entries with individual tree metrics if available
            entries = TreeRanker.build_entries(
                {client_id: trees}, 
                {client_id: meta},
                X_val=X_val,
                diversity_service=diversity_svc
            )

            
            # Rank within client
            ranked = ranker.rank(entries)
            
            # Limit trees per client if specified
            if max_trees_per_client is not None:
                ranked = ranked[:max_trees_per_client]
            
            client_ranked_entries[client_id] = ranked
        
        # Step 2: Interleave trees from all clients using round-robin
        round_robin_entries = self._interleave_round_robin(client_ranked_entries, client_ids)
        
        # If no validation data, return all round-robin ordered trees (no PF early stopping)
        if X_val is None or y_val is None:
            global_trees = [e.tree for e in round_robin_entries]
            selected_ids = {cid: [] for cid in client_trees.keys()}
            for entry in round_robin_entries:
                selected_ids[entry.client_id].append(entry.tree_local_id)
            return global_trees, selected_ids, round_robin_entries
        
        # Step 3: Apply Progressive Forest with early stopping using validation data
        global_trees, selected_entries = self._progressive_selection_with_convergence(
            round_robin_entries, X_val, y_val, t_max=t_max, n_clients=len(client_ids)
        )
        
        # Build selected_ids dictionary
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in selected_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)
        
        return global_trees, selected_ids, selected_entries
    
    def _get_ranking_criterion(self, **kwargs) -> RankingCriterion:
        """Get the ranking criterion, allowing subclasses to override."""
        return self.ranking_criterion
    
    def _interleave_round_robin(self, client_ranked_entries: Dict[str, List[TreeEntry]],
                                 client_ids: List[str]) -> List[TreeEntry]:
        """Interleave trees from all clients using round-robin scheduling.
        
        When one client runs out of trees, continue with remaining clients
        until all trees from all clients are exhausted.

        Args:
            client_ranked_entries: Dict mapping client_id to ranked tree entries
            client_ids: List of client IDs in order

        Returns:
            List of TreeEntry in round-robin order
        """
        round_robin_entries = []
        
        # Create indices to track position in each client's ranked list
        client_indices = {cid: 0 for cid in client_ids}
        max_trees_any_client = max(
            len(entries) for entries in client_ranked_entries.values()
        ) if client_ranked_entries else 0
        
        # Round-robin: iterate until all clients are exhausted
        for _ in range(max_trees_any_client):
            for client_id in client_ids:
                entries = client_ranked_entries[client_id]
                idx = client_indices[client_id]
                
                # If this client still has trees, add the next one
                if idx < len(entries):
                    round_robin_entries.append(entries[idx])
                    client_indices[client_id] += 1
                # If client is exhausted, skip and continue with next client

        return round_robin_entries
    
    def _progressive_selection_with_convergence(
        self,
        round_robin_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val: np.ndarray,
        t_max: int = None,
        n_clients: int = None
    ) -> Tuple[List[Any], List[TreeEntry]]:
        """Select trees progressively using CPF algorithm with early stopping.

        Aggregation stops by:
        1. Convergence (2 consecutive episodes with accuracy delta < CONVERGENCE)
        2. T_MAX trees reached (límite máximo de árboles)
        3. All trees from all clients have been added
        
        IMPORTANT: EPISODE = n_clients (W en la tesis), ya que en cada iteración
        se agrega el mejor árbol restante de CADA cliente.

        Args:
            round_robin_entries: Trees in round-robin order (already interleaved)
            X_val: Validation features
            y_val: Validation labels
            t_max: Maximum number of trees (default: class T_MAX = 100)
            n_clients: Number of clients (used to set EPISODE = W)

        Returns:
            Tuple of (selected_trees, selected_entries)
        """
        # EPISODE = n_clients (W en la tesis)
        EPISODE = n_clients if n_clients is not None else 5
        models_built = 0
        stop_counter = 0
        previous_episode_accuracy = None
        episode_accuracy_dif = 0.002

        selected_entries: List[TreeEntry] = []
        episode_accuracies = []

        # Use provided t_max or default to class constant
        T_MAX = t_max if t_max is not None else self.T_MAX
        
        # Prediction cache: cache predictions for each unique tree
        # key: tree object (or id), value: np.ndarray of predictions
        prediction_cache: Dict[int, np.ndarray] = {}

        # Progressive selection: stop by convergence, T_MAX, or all trees added
        while models_built < min(len(round_robin_entries), T_MAX):
            # Build an episode of trees (EPISODE = n_clients)
            episode_entries = round_robin_entries[models_built:models_built + EPISODE]
            if not episode_entries:
                break

            # Add trees from this episode
            for entry in episode_entries:
                selected_entries.append(entry)

            models_built = len(selected_entries)

            # Evaluate ensemble accuracy on validation set
            predictions = self._predict_ensemble(selected_entries, X_val, cache=prediction_cache)
            
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
                # Calculate marginal improvement (current - previous)
                current_acc = episode_accuracies[-1]
                previous_acc = episode_accuracies[-2]
                improvement = current_acc - previous_acc

                # Convergence: improvement is smaller than threshold (CONVERGENCE)
                if improvement < self.CONVERGENCE:
                    stop_counter += 1
                    # Slightly reduce EPISODE size to refine the selection (standard CPF)
                    EPISODE = max(1, EPISODE - 1)
                    if self.verbose if hasattr(self, 'verbose') else False:
                        print(f"  [Agregación] Convergencia detectada (mejora={improvement:.5f}, stop_counter={stop_counter})")
                    if stop_counter >= 2:
                        break
                else:
                    stop_counter = 0
                    if self.verbose if hasattr(self, 'verbose') else False:
                        print(f"  [Agregación] Mejora detectada ({improvement:.5f}), próximo episodio={EPISODE}")

        return [e.tree for e in selected_entries], selected_entries
    
    def _predict_ensemble(self, selected_entries: List[TreeEntry], X: np.ndarray, 
                         cache: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        """Make predictions using simple majority voting from selected trees.
        
        Args:
            selected_entries: List of selected tree entries
            X: Feature matrix (n_samples, n_features)
            cache: Optional cache of predictions per tree
            
        Returns:
            Array of predicted class labels
        """
        if not selected_entries:
            return np.zeros(X.shape[0], dtype=int)
        
        n_samples = X.shape[0]
        n_trees = len(selected_entries)
        
        # Collect predictions from all trees using batch inference and optional caching
        all_predictions = np.zeros((n_samples, n_trees), dtype=int)
        for j, entry in enumerate(selected_entries):
            tree_id = id(entry.tree)
            if cache is not None and tree_id in cache:
                all_predictions[:, j] = cache[tree_id]
            else:
                # Use the vectorized predict(X) method
                preds = entry.tree.predict(X)
                all_predictions[:, j] = preds
                if cache is not None:
                    cache[tree_id] = preds
        
        # Majority voting per sample using pure Python/Numpy implementation
        return calculate_mode(all_predictions, axis=1)


class S5PerClientAccuracyStrategy(PerClientProgressiveStrategy):
    """Strategy S5: Per-Client Accuracy with Progressive Forest
    
    Each client's trees are ranked by accuracy independently, then incorporated
    via round-robin with CPF early stopping.
    """
    
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.ACCURACY
    
    @property
    def strategy_id(self) -> str:
        return "S5"


class S6PerClientF1Strategy(PerClientProgressiveStrategy):
    """Strategy S6: Per-Client Macro-F1 with Progressive Forest
    
    Each client's trees are ranked by macro-F1 independently, then incorporated
    via round-robin with CPF early stopping.
    """
    
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.MACRO_F1
    
    @property
    def strategy_id(self) -> str:
        return "S6"


class S7PerClientF1PCDStrategy(PerClientProgressiveStrategy):
    """Strategy S7: Per-Client F1 + PCD with Progressive Forest
    
    Each client's trees are ranked by weighted combination of macro-F1 and PCD,
    then incorporated via round-robin with CPF early stopping.
    """
    
    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.F1_PCD
    
    @property
    def strategy_id(self) -> str:
        return "S7"
