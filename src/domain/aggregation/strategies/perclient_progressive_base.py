"""Base class for per-client strategies (S5-S7) with Progressive Forest aggregation.

These strategies rank trees within each client independently and incorporate them
using round-robin scheduling with Progressive Forest early stopping.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score
from scipy import stats

from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry
from ...model.cpf_implementation.estimator import ProactiveForestClassifier


class PerClientProgressiveStrategy(ABC):
    """Base class for per-client ranking strategies with Progressive Forest.
    
    Subclasses define the ranking criterion (accuracy, macro-F1, F1+PCD).
    Each client's trees are ranked independently, then incorporated via round-robin.
    Aggregation stops only by:
    1. Convergence (CPF early stopping - 2 consecutive episodes with delta < 0.002)
    2. All trees from all clients have been added
    
    The max_trees_per_client parameter limits trees per client BEFORE aggregation.
    During aggregation, early stopping applies to the combined forest.
    """
    
    CONVERGENCE = 0.002
    INITIAL_EPISODE = 5
    
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
        ranker = TreeRanker(criterion=criterion,
                           f1_weight=kwargs.get('f1_weight', 0.5),
                           pcd_weight=kwargs.get('pcd_weight', 0.5))
        
        for client_id, trees in client_trees.items():
            meta = client_metadata[client_id]
            entries = []
            for local_idx, tree in enumerate(trees):
                entries.append(TreeEntry(
                    tree=tree,
                    client_id=client_id,
                    tree_local_id=local_idx,
                    accuracy=meta.accuracy,
                    macro_f1=meta.macro_f1,
                    pcd=meta.pcd,
                ))
            
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
            round_robin_entries, X_val, y_val
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
        
        Args:
            client_ranked_entries: Dict mapping client_id to ranked tree entries
            client_ids: List of client IDs in order
            
        Returns:
            List of TreeEntry in round-robin order
        """
        round_robin_entries = []
        max_len = max(len(entries) for entries in client_ranked_entries.values()) if client_ranked_entries else 0
        
        for i in range(max_len):
            for client_id in client_ids:
                entries = client_ranked_entries[client_id]
                if i < len(entries):
                    round_robin_entries.append(entries[i])
        
        return round_robin_entries
    
    def _progressive_selection_with_convergence(
        self,
        round_robin_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[List[Any], List[TreeEntry]]:
        """Select trees progressively using CPF algorithm with early stopping.
        
        Aggregation stops only by:
        1. Convergence (2 consecutive episodes with accuracy delta < CONVERGENCE)
        2. All trees from all clients have been added
        
        Args:
            round_robin_entries: Trees in round-robin order
            X_val: Validation features
            y_val: Validation labels
            
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
        
        # Progressive selection: stop only by convergence or all trees added
        while models_built < len(round_robin_entries):
            # Build an episode of trees
            episode_entries = round_robin_entries[models_built:models_built + EPISODE]
            if not episode_entries:
                break
            
            # Add trees from this episode
            for entry in episode_entries:
                selected_entries.append(entry)
            
            models_built = len(selected_entries)
            
            # Evaluate ensemble accuracy on validation set
            predictions = self._predict_ensemble(selected_entries, X_val)
            acc = accuracy_score(y_val, predictions)
            episode_accuracies.append(acc)
            
            # Check convergence after first episode
            if len(episode_accuracies) >= 2:
                # Episode accuracy = range (max - min) of accuracies so far
                episode_accuracy = max(episode_accuracies) - min(episode_accuracies)
                
                if previous_episode_accuracy is not None:
                    episode_accuracy_dif = episode_accuracy - previous_episode_accuracy
                
                # Convergence: small change or small range
                if episode_accuracy_dif < self.CONVERGENCE or episode_accuracy < self.CONVERGENCE:
                    stop_counter += 1
                    EPISODE = max(1, EPISODE - 1)
                    if stop_counter == 2:
                        break
                else:
                    stop_counter = 0
                    EPISODE += 1
                
                previous_episode_accuracy = episode_accuracy
        
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
        
        # Collect predictions from all trees (each tree predicts one sample at a time)
        all_predictions = np.zeros((n_samples, n_trees), dtype=int)
        for j, entry in enumerate(selected_entries):
            for i in range(n_samples):
                all_predictions[i, j] = entry.tree.predict(X[i])
        
        # Majority voting per sample
        from scipy import stats
        mode_result = stats.mode(all_predictions, axis=1, keepdims=False)
        return mode_result.mode


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
