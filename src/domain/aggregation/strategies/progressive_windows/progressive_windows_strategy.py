"""Progressive Windows Strategy.

Implementation of the Progressive Windows aggregation strategy
for federated Proactive Forest.

Key features:
- Window-based training (W trees per client per round)
- Dynamic score calculation: Score(T) = α * F1(T) + (1-α) * Diversity(T|G)
- Sequential Round Robin aggregation with immediate global forest update
- Progressive Forest global stopping criterion
- PCD (Partition-Coverage Distance) for diversity measurement
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field

from ...base_strategy import IAggregationStrategy
from ...tree_ranker import TreeEntry
from ....model.cpf_implementation.estimator import ProactiveForestClassifier
from ....model.cpf_implementation.tree import DecisionTree


@dataclass
class WindowReport:
    """Report for a window of trees from a client."""
    client_id: str
    window_trees: List[Any]
    window_f1_scores: List[float]
    window_pcd_scores: List[float]
    accuracy: float
    macro_f1: float


@dataclass
class ProgressiveWindowsResult:
    """Result of Progressive Windows selection process."""
    global_trees: List[Any] = field(default_factory=list)
    selected_ids: Dict[str, List[int]] = field(default_factory=dict)
    all_tree_entries: List[TreeEntry] = field(default_factory=list)
    rounds_completed: int = 0
    convergence_round: Optional[int] = None
    final_accuracy: float = 0.0
    final_macro_f1: float = 0.0


class ProgressiveWindowsStrategy(IAggregationStrategy):
    """
    Progressive Windows aggregation strategy.

    This strategy implements the 6-phase federated learning approach:

    Phase 1: Local training with Proactive Forest (windows of W trees)
    Phase 2: Window communication to server
    Phase 3: Round Robin aggregation with dynamic scoring
    Phase 4: Progressive Forest global stopping criterion
    Phase 5: Client update with global trees
    Phase 6: Hybrid inference (local + global voting)

    The key innovation is the dynamic score calculation that balances
    individual tree performance (F1) with diversity contribution to the
    global forest (PCD).

    Score(T) = α * F1(T) + (1-α) * Diversity(T|G)

    Where:
    - F1(T): Macro-F1 score of tree T
    - Diversity(T|G): PCD-based diversity of T with respect to global forest G
    - α: Weight parameter (default 0.5)

    Round Robin Process:
    1. Generate random permutation of clients each round
    2. For each client in permutation order:
       - Calculate dynamic scores for all trees in window
       - Select tree with highest score
       - Immediately add to global forest
       - Update diversity calculations for next client
    3. Continue until stopping criterion met

    Stopping Criteria (Progressive Forest Global):
    - Convergence: accuracy improvement < threshold for 2 consecutive rounds
    - Max rounds: maximum number of rounds reached
    """

    # Default hyperparameters
    DEFAULT_WINDOW_SIZE = 5  # W: trees per window
    DEFAULT_MAX_ROUNDS = 20  # Maximum rounds (R_MAX)
    DEFAULT_CONVERGENCE_THRESHOLD = 0.002  # Convergence threshold
    DEFAULT_ALPHA = 0.5  # Balance between F1 and Diversity
    DEFAULT_LAMBDA = 0.5  # Hybrid prediction weight (local vs global)

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        alpha: float = DEFAULT_ALPHA,
        lambda_hybrid: float = DEFAULT_LAMBDA,
        verbose: bool = False
    ):
        """
        Initialize Progressive Windows strategy.

        Args:
            window_size: Number of trees per window (W). Default: 5
            max_rounds: Maximum number of rounds (R_MAX). Default: 20
            convergence_threshold: Convergence threshold for early stopping. Default: 0.002
            alpha: Weight for F1 vs Diversity in score calculation. Default: 0.5
            lambda_hybrid: Weight for local vs global in hybrid prediction. Default: 0.5
            verbose: Enable verbose logging. Default: False
        """
        self.window_size = window_size
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.alpha = alpha
        self.lambda_hybrid = lambda_hybrid
        self.verbose = verbose

        # Track global forest state
        self._global_trees: List[Any] = []
        self._global_tree_sources: List[str] = []  # Track which client each tree came from

        # Track convergence info (for FLResults)
        self.convergence_round: Optional[int] = None

    @property
    def strategy_id(self) -> str:
        """Return strategy identifier."""
        return "PW"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: Optional[int] = None,
        max_trees_per_client: Optional[int] = None,
        t_max: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[Any], Dict[str, List[int]], List[Any]]:
        """
        Aggregate trees using Round Robin with Dynamic Scoring.

        This implements Phase 3 (Round Robin aggregation) and Phase 4
        (Progressive Forest global stopping).
        """
        # Override defaults with kwargs
        self.alpha = kwargs.get('alpha', self.alpha)
        self.window_size = kwargs.get('window_size', self.window_size)
        self.max_rounds = kwargs.get('max_rounds', self.max_rounds)

        # If t_max provided, calculate max_rounds from it
        if t_max is not None:
            n_clients = len(client_trees)
            self.max_rounds = max(1, t_max // n_clients)

        client_ids = list(client_trees.keys())
        n_clients = len(client_ids)
        if not client_ids:
            return [], {}, []

        # Initialize result tracking
        result = ProgressiveWindowsResult()
        result.selected_ids = {cid: [] for cid in client_ids}

        # Reset global forest state
        self._global_trees = []
        self._global_tree_sources = []
        self.convergence_round = None

        # Track all tree entries for ranking visualization
        all_tree_entries: List[TreeEntry] = []

        # Progressive Forest global stopping criterion
        round_accuracies: List[float] = []
        stop_counter = 0
        previous_accuracy = None

        # ──────────────────────────────────────────────────────────────────────
        # FASE 3: AGREGACIÓN GLOBAL CON PROGRESSIVE WINDOWS
        # ──────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("\n" + "=" * 100)
            print("🔄 FASE 3: AGREGACIÓN GLOBAL CON PROGRESSIVE WINDOWS")
            print("=" * 100)
            print(f"\n📋 Parámetros de configuración:")
            print(f"   • Número de clientes (k): {n_clients}")
            print(f"   • Tamaño de ventana (W): {self.window_size} árboles")
            print(f"   • Máximo de rondas (R_MAX): {self.max_rounds}")
            print(f"   • Alpha (α) para Score: {self.alpha}")
            print(f"   • Fórmula: Score(T) = α·F1(T) + (1-α)·Diversidad(T|G)")
            print("=" * 100 + "\n")

        # Main Round Robin loop
        for round_num in range(self.max_rounds):
            if self.verbose:
                print("\n" + "=" * 100)
                print(f"🔁 RONDA {round_num + 1}/{self.max_rounds}")
                print("=" * 100)
                print(f"📊 Estado actual del bosque global: {len(self._global_trees)} árboles")

                # Current accuracy before this round
                if len(self._global_trees) > 0 and X_val is not None and y_val is not None:
                    current_acc = self._evaluate_forest_accuracy(self._global_trees, X_val, y_val)
                    print(f"📈 Accuracy actual del bosque global: {current_acc:.6f}")
                else:
                    print(f"📈 Accuracy actual del bosque global: N/A (bosque vacío)")

            # Generate random permutation for this round (mitigate position bias)
            rng = np.random.RandomState(seed=round_num)
            round_permutation = rng.permutation(client_ids).tolist()

            if self.verbose:
                print(f"\n🎲 Permutación aleatoria de clientes: {' → '.join(round_permutation)}")
                print(f"   (orden para mitigar sesgo de posición)")

            # Process each client in permutation order
            trees_added_this_round = 0
            for client_idx, client_id in enumerate(round_permutation):
                # Get client's trees for this round
                client_all_trees = client_trees[client_id]

                # Calculate which trees belong to this round window
                window_start = round_num * self.window_size
                window_end = min(window_start + self.window_size, len(client_all_trees))

                if window_start >= len(client_all_trees):
                    if self.verbose:
                        print(f"\n  ⚠️  {client_id}: Sin árboles disponibles para esta ventana")
                    continue

                window_trees = client_all_trees[window_start:window_end]

                # Get metadata for this client
                client_meta = client_metadata.get(client_id, {})
                if hasattr(client_meta, 'to_dict'):
                    client_meta_dict = client_meta.to_dict()
                elif isinstance(client_meta, dict):
                    client_meta_dict = client_meta
                else:
                    client_meta_dict = {}

                if self.verbose:
                    print(f"\n{'─' * 100}")
                    print(f"👤 TURNO {client_idx + 1}: {client_id}")
                    print(f"{'─' * 100}")
                    print(f"   Ventana: árboles [{window_start}:{window_end}] de {len(client_all_trees)} totales")
                    print(f"   Árboles en ventana: {len(window_trees)}")
                    print(f"   Métricas del cliente: Accuracy={client_meta_dict.get('accuracy', 0):.4f}, "
                          f"F1={client_meta_dict.get('macro_f1', 0):.4f}, PCD={client_meta_dict.get('pcd', 0):.4f}")
                    print(f"   Tamaño bosque global ANTES: {len(self._global_trees)} árboles")

                # Calculate dynamic scores for each tree in window
                tree_scores: List[Tuple[int, Any, float, float, float]] = []

                for local_idx, tree in enumerate(window_trees):
                    global_idx = window_start + local_idx

                    # Get tree's F1 score
                    if 'window_f1_scores' in client_meta_dict and local_idx < len(client_meta_dict['window_f1_scores']):
                        tree_f1 = client_meta_dict['window_f1_scores'][local_idx]
                    else:
                        tree_f1 = client_meta_dict.get('macro_f1', 0.5)

                    # Calculate diversity with respect to current global forest
                    diversity = self._calculate_diversity(tree, self._global_trees)

                    # Dynamic score: Score(T) = α * F1(T) + (1 - α) * Diversity(T|G)
                    # First round: G is empty, diversity not informative → α = 1
                    effective_alpha = self.alpha if len(self._global_trees) > 0 else 1.0
                    score = effective_alpha * tree_f1 + (1 - effective_alpha) * diversity

                    tree_scores.append((global_idx, tree, score, tree_f1, diversity))

                    # Add to all_tree_entries
                    entry = TreeEntry(
                        tree=tree,
                        client_id=client_id,
                        tree_local_id=global_idx,
                        accuracy=client_meta_dict.get('accuracy', 0.0),
                        macro_f1=tree_f1,
                        pcd=diversity,
                    )
                    all_tree_entries.append(entry)

                if not tree_scores:
                    continue

                # PASO 1: Evaluación de score por árbol
                if self.verbose:
                    print(f"\n   📊 PASO 1: Evaluación de scores para {client_id}")
                    print(f"   {'─' * 96}")
                    print(f"   {'Árbol':<10} | {'F1(T)':<12} | {'Diversidad(T|G)':<18} | {'Score(T)':<12} | {'Ranking':<8}")
                    print(f"   {'─' * 96}")

                    ranked_trees = sorted(tree_scores, key=lambda x: x[2], reverse=True)
                    for rank, (idx, tree, score, f1, div) in enumerate(ranked_trees, 1):
                        marker = "⭐" if rank == 1 else "  "
                        print(f"   {marker} T{idx:<7} | {f1:<12.6f} | {div:<18.6f} | {score:<12.6f} | #{rank:<7}")
                    print(f"   {'─' * 96}")
                    if len(self._global_trees) == 0:
                        print(f"   ℹ️  Primera ronda: G está vacío → α=1 (solo F1 determina el score)")

                # PASO 2: Selección del mejor árbol del cliente en turno
                best_local_idx, best_tree, best_score, best_f1, best_diversity = max(tree_scores, key=lambda x: x[2])

                if self.verbose:
                    print(f"\n   ✅ PASO 2: Mejor árbol seleccionado")
                    print(f"   {'─' * 96}")
                    print(f"   • Árbol: T{best_local_idx} (índice local en ventana)")
                    print(f"   • Score: {best_score:.6f} = {self.alpha:.2f}×{best_f1:.6f} + {1-self.alpha:.2f}×{best_diversity:.6f}")
                    print(f"   • F1(T): {best_f1:.6f}")
                    print(f"   • Diversidad(T|G): {best_diversity:.6f}")

                # PASO 3: Incorporación inmediata y actualización del bosque global
                self._global_trees.append(best_tree)
                self._global_tree_sources.append(client_id)
                result.global_trees.append(best_tree)
                result.selected_ids[client_id].append(best_local_idx)
                trees_added_this_round += 1

                if self.verbose:
                    print(f"\n   🌳 PASO 3: Árbol incorporado al bosque global")
                    print(f"   {'─' * 96}")
                    print(f"   • Bosque global DESPUÉS: {len(self._global_trees)} árboles")
                    print(f"   • Cliente contribuyente: {client_id}")
                    print(f"   • Este árbol afectará los scores de los siguientes clientes")

            # End of round
            result.rounds_completed = round_num + 1

            # ──────────────────────────────────────────────────────────────────────
            # FASE 4: CRITERIO DE PARADA (PROGRESSIVE GLOBAL)
            # ──────────────────────────────────────────────────────────────────────
            if X_val is not None and y_val is not None and len(self._global_trees) > 0:
                current_accuracy = self._evaluate_forest_accuracy(self._global_trees, X_val, y_val)
                round_accuracies.append(current_accuracy)

                if self.verbose:
                    print(f"\n{'=' * 100}")
                    print(f"📊 FASE 4: CRITERIO DE PARADA (PROGRESSIVE GLOBAL)")
                    print(f"{'=' * 100}")
                    print(f"   Accuracy después de ronda {round_num + 1}: {current_accuracy:.6f}")

                # Check convergence
                if previous_accuracy is not None:
                    accuracy_improvement = abs(current_accuracy - previous_accuracy)

                    if self.verbose:
                        print(f"   Accuracy anterior: {previous_accuracy:.6f}")
                        print(f"   Mejora: |{current_accuracy:.6f} - {previous_accuracy:.6f}| = {accuracy_improvement:.6f}")
                        print(f"   Umbral de convergencia: {self.convergence_threshold}")

                    if accuracy_improvement <= self.convergence_threshold:
                        stop_counter += 1
                        if self.verbose:
                            print(f"   ⚠️  Mejora ≤ umbral → Contador de parada: {stop_counter}/2")

                        if stop_counter >= 2:
                            if self.verbose:
                                print(f"\n{'=' * 100}")
                                print(f"✅ CONVERGENCIA ALCANZADA")
                                print(f"{'=' * 100}")
                                print(f"   • Rondas completadas: {round_num + 1}")
                                print(f"   • Árboles en bosque global: {len(self._global_trees)}")
                                print(f"   • Accuracy final: {current_accuracy:.6f}")
                                print(f"{'=' * 100}\n")
                            result.convergence_round = round_num + 1
                            self.convergence_round = round_num + 1
                            break
                    else:
                        stop_counter = 0
                        if self.verbose:
                            print(f"   ✅ Mejora > umbral → Reiniciar contador")

                previous_accuracy = current_accuracy
                result.final_accuracy = current_accuracy

        # Set convergence_round if not already set
        if self.convergence_round is None:
            self.convergence_round = result.rounds_completed

        # Calculate final macro F1
        if X_val is not None and y_val is not None:
            result.final_macro_f1 = self._evaluate_forest_f1(self._global_trees, X_val, y_val)

        # Final summary
        if self.verbose:
            print(f"\n{'=' * 100}")
            print(f"📊 RESUMEN FINAL DE AGREGACIÓN")
            print(f"{'=' * 100}")
            print(f"   • Rondas completadas: {result.rounds_completed}")
            print(f"   • Árboles en bosque global: {len(result.global_trees)}")
            print(f"   • Accuracy final: {result.final_accuracy:.6f}")
            print(f"   • Macro-F1 final: {result.final_macro_f1:.6f}")
            print(f"   • Convergencia: Ronda {self.convergence_round}")
            print(f"\n   • Árboles seleccionados por cliente:")
            for cid in client_ids:
                n_selected = len(result.selected_ids[cid])
                n_total = len(client_trees[cid])
                pct = n_selected / n_total * 100 if n_total > 0 else 0
                print(f"      {cid}: {n_selected}/{n_total} ({pct:.1f}%)")
            print(f"{'=' * 100}\n")

        return result.global_trees, result.selected_ids, all_tree_entries

    def _calculate_diversity(self, tree: Any, global_trees: List[Any]) -> float:
        """Calculate diversity of a tree with respect to the global forest using PCD."""
        if not global_trees:
            return 1.0

        pcd_sum = 0.0
        for global_tree in global_trees:
            pcd = self._calculate_pcd_between_trees(tree, global_tree)
            pcd_sum += pcd

        return pcd_sum / len(global_trees)

    def _calculate_pcd_between_trees(self, tree1: Any, tree2: Any) -> float:
        """Calculate Partition-Coverage Distance (PCD) between two trees."""
        try:
            struct1 = tree1.get_tree_structure() if hasattr(tree1, 'get_tree_structure') else tree1
            struct2 = tree2.get_tree_structure() if hasattr(tree2, 'get_tree_structure') else tree2

            depth1 = self._get_tree_depth(struct1)
            depth2 = self._get_tree_depth(struct2)
            nodes1 = self._count_nodes(struct1)
            nodes2 = self._count_nodes(struct2)

            depth_diff = abs(depth1 - depth2) / max(depth1, depth2, 1)
            size_diff = abs(nodes1 - nodes2) / max(nodes1, nodes2, 1)

            return min(1.0, (depth_diff + size_diff) / 2)
        except Exception:
            return 0.5

    def _get_tree_depth(self, tree_structure: Any) -> int:
        """Get the depth of a tree from its structure."""
        if isinstance(tree_structure, dict):
            if 'depth' in tree_structure:
                return tree_structure['depth']
            if 'children' in tree_structure:
                children = tree_structure['children']
                if not children:
                    return 1
                return 1 + max(self._get_tree_depth(c) for c in children)
        return 1

    def _count_nodes(self, tree_structure: Any) -> int:
        """Count the number of nodes in a tree."""
        if isinstance(tree_structure, dict):
            count = 1
            if 'children' in tree_structure:
                for child in tree_structure['children']:
                    count += self._count_nodes(child)
            return count
        return 1

    def _evaluate_forest_accuracy(self, trees: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy of forest predictions using majority voting."""
        from scipy import stats

        n_samples = X.shape[0]
        n_trees = len(trees)

        if n_trees == 0 or n_samples == 0:
            return 0.0

        all_predictions = np.zeros((n_samples, n_trees), dtype=int)

        for j, tree in enumerate(trees):
            for i in range(n_samples):
                try:
                    all_predictions[i, j] = tree.predict(X[i])
                except Exception:
                    all_predictions[i, j] = 0

        mode_result = stats.mode(all_predictions, axis=1, keepdims=False)
        predictions = mode_result.mode

        return float(np.mean(predictions == y))

    def _evaluate_forest_f1(self, trees: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate macro F1 score of forest predictions."""
        from sklearn.metrics import f1_score
        from scipy import stats

        n_samples = X.shape[0]
        n_trees = len(trees)

        if n_trees == 0 or n_samples == 0:
            return 0.0

        all_predictions = np.zeros((n_samples, n_trees), dtype=int)

        for j, tree in enumerate(trees):
            for i in range(n_samples):
                try:
                    all_predictions[i, j] = tree.predict(X[i])
                except Exception:
                    all_predictions[i, j] = 0

        mode_result = stats.mode(all_predictions, axis=1, keepdims=False)
        predictions = mode_result.mode

        try:
            return float(f1_score(y, predictions, average='macro', zero_division=0))
        except Exception:
            return 0.0

    def get_global_trees(self) -> List[Any]:
        """Return current global forest trees."""
        return self._global_trees

    def get_global_tree_sources(self) -> List[str]:
        """Return list of client IDs that contributed each global tree."""
        return self._global_tree_sources
