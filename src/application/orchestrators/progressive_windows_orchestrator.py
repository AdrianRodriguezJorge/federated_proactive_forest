"""
Progressive Windows Orchestrator - Implementación Incremental Verdadera.

Este módulo implementa el flujo Progressive Windows exactamente como se describe en la propuesta teórica:
- Entrenamiento incremental por ventanas (W árboles por ronda)
- Comunicación de ventana al servidor después de cada ronda
- Agregación Round Robin con score dinámico
- Criterio de parada Progressive Forest global

NO afecta las estrategias S1-S7 que usan FLEXOrchestrator.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, f1_score

from src.domain.model.proactive_forest import ProactiveForest
from src.domain.aggregation.strategies.progressive_windows import ProgressiveWindowsStrategy
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metadata.client_metadata import ClientMetadata


@dataclass
class ProgressiveWindowsResults:
    """Resultados del entrenamiento Progressive Windows incremental."""
    strategy_id: str = "PW"
    global_accuracy: float = 0.0
    global_macro_f1: float = 0.0
    n_trees_global: int = 0
    client_ids: List[str] = field(default_factory=list)
    client_accuracies: Dict[str, float] = field(default_factory=dict)
    client_f1_scores: Dict[str, float] = field(default_factory=dict)
    client_metadata: Dict[str, ClientMetadata] = field(default_factory=dict)
    selected_ids: Dict[str, List[int]] = field(default_factory=dict)
    all_tree_entries: List[Any] = field(default_factory=list)
    global_report: Any = None
    global_predictions: np.ndarray = None
    num_rounds: int = 0
    convergence_round: Optional[int] = None
    client_hybrid_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    y_test: np.ndarray = None
    class_names: List[str] = field(default_factory=list)
    client_hybrid_forest_sizes: Dict[str, int] = field(default_factory=dict)
    round_logs: List[Dict[str, Any]] = field(default_factory=list)  # Logs detallados por ronda


class ProgressiveWindowsOrchestrator:
    """
    Orquestador para Progressive Windows Incremental Verdadero.

    Flujo:
    1. Por cada ronda:
       a. Cada cliente entrena UNA VENTANA de W árboles
       b. Clientes envían ventana al servidor
       c. Servidor hace Round Robin con score dinámico
       d. Servidor verifica convergencia (Progressive Forest)
       e. Si converge → PARA, si no → siguiente ronda

    2. Tras convergencia:
       a. Servidor distribuye árboles globales a clientes
       b. Clientes hacen merge (excluyendo propios seleccionados)
       c. Inferencia híbrida (λ·local + (1-λ)·global)
    """

    def __init__(
        self,
        config: Union[dict, Any],
        dataset_split: DatasetSplit,
        verbose: bool = True
    ):
        """
        Initialize orchestrator.
        """
        if isinstance(config, dict):
            self.config_dict = config
            self.dataset_cfg = config.get('dataset', {})
            self.fed_cfg = config.get('federation', {})
            self.model_cfg = config.get('model', {})
            self.agg_cfg = config.get('aggregation', {})
            self.pred_cfg = config.get('prediction', {})
        else:
            self.config_dict = config.dict()
            self.dataset_cfg = config.dataset.dict()
            self.fed_cfg = config.federation.dict()
            self.model_cfg = config.model.dict()
            self.agg_cfg = config.aggregation.dict()
            self.pred_cfg = config.prediction.dict()

        self.dataset_split = dataset_split
        self.verbose = verbose

        # Services
        from src.domain.services.label_service import SimpleLabelService
        self.label_svc = SimpleLabelService(dataset_split.class_names)

        # Parámetros Progressive Windows
        self.n_clients = self.fed_cfg.get('n_clients', 5)
        self.window_size = self.agg_cfg.get('window_size', 5)
        self.max_rounds = self.agg_cfg.get('max_rounds', 20)
        self.f1_weight = self.agg_cfg.get('f1_weight', 0.5)
        self.convergence_threshold = self.agg_cfg.get('convergence_threshold', 0.002)
        self.local_weight = self.pred_cfg.get('local_weight', 0.5)

        # Estado del entrenamiento
        self.client_forests: Dict[str, ProactiveForest] = {}
        self.client_partitions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.global_trees: List[Any] = []
        self.global_tree_sources: List[str] = []  # Qué cliente dio cada árbol
        self.round_logs: List[Dict[str, Any]] = []

        # Estrategia Progressive Windows
        self.strategy = ProgressiveWindowsStrategy(
            window_size=self.window_size,
            max_rounds=self.max_rounds,
            f1_weight=self.f1_weight,
            convergence_threshold=self.convergence_threshold,
            verbose=self.verbose
        )

    def run_federated_round(self) -> ProgressiveWindowsResults:
        """
        Ejecuta entrenamiento federado Progressive Windows incremental verdadero.

        Returns:
            ProgressiveWindowsResults con todos los resultados
        """
        if self.verbose:
            print("\n" + "=" * 100)
            print("🚀 INICIANDO PROGRESSIVE WINDOWS INCREMENTAL VERDADERO")
            print("=" * 100)
            print(f"\n📋 Configuración:")
            print(f"   • Clientes: {self.n_clients}")
            print(f"   • Ventana (W): {self.window_size} árboles")
            print(f"   • Máx rondas: {self.max_rounds}")
            print(f"   • F1 Weight (α): {self.f1_weight}")
            print(f"   • PCD Weight (β): {1.0 - self.f1_weight:.2f}")
            print(f"   • Lambda: {self.local_weight}")
            print("=" * 100 + "\n")

        # Inicializar resultados
        results = ProgressiveWindowsResults()
        results.client_ids = [f"client_{i}" for i in range(self.n_clients)]
        results.y_test = self.label_svc.transform(self.dataset_split.y_test)
        results.class_names = self.dataset_split.class_names

        # Variables para criterio de parada
        round_accuracies: List[float] = []
        stop_counter = 0
        previous_accuracy = None

        for round_num in range(self.max_rounds):
            if self.verbose:
                print("\n" + "=" * 100)
                print(f"🔁 RONDA {round_num + 1}/{self.max_rounds}")
                print("=" * 100)
                print(f"📊 Bosque global actual: {len(self.global_trees)} árboles")

                X_val_server = self.dataset_split.X_val if self.dataset_split.X_val is not None else self.dataset_split.X_test
                y_val_server = self.dataset_split.y_val if self.dataset_split.y_val is not None else self.dataset_split.y_test
                y_val_eval = self.label_svc.transform(y_val_server)

                if len(self.global_trees) > 0:
                    current_acc = self._evaluate_forest_accuracy(
                        self.global_trees, X_val_server, y_val_eval
                    )
                    print(f"📈 Accuracy actual (val): {current_acc:.6f}")
                else:
                    print(f"📈 Accuracy actual: N/A (bosque vacío)")

            round_log = {
                'round': round_num + 1,
                'trees_before': len(self.global_trees),
                'client_windows': {},
                'selected_trees': {},
                'round_accuracy': None
            }

            # FASE 1: Cada cliente entrena UNA VENTANA de W árboles
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🌱 FASE 1: ENTRENAMIENTO DE VENTANAS LOCALES")
                print(f"{'─' * 100}")

            client_windows: Dict[str, List[Any]] = {}
            client_window_metadata: Dict[str, Dict[str, Any]] = {}

            for client_id in results.client_ids:
                window_trees, window_metrics = self._train_window(
                    client_id=client_id,
                    window_index=round_num,
                    X_train=self.client_partitions[client_id][0],
                    y_train=self.client_partitions[client_id][1]
                )

                client_windows[client_id] = window_trees
                client_window_metadata[client_id] = {
                    'accuracy': window_metrics['accuracy'],
                    'macro_f1': window_metrics['macro_f1'],
                    'pcd': window_metrics['pcd'],
                    'window_f1_scores': window_metrics.get('tree_f1_scores', []),
                    'n_trees': len(window_trees)
                }
                round_log['client_windows'][client_id] = len(window_trees)

                if self.verbose:
                    print(f"   {client_id}: {len(window_trees)} árboles entrenados "
                          f"(Accuracy={window_metrics['accuracy']:.4f})")

            # FASE 2-3: Servidor recibe ventanas y hace Round Robin
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🔄 FASE 2-3: AGREGACIÓN ROUND ROBIN DINÁMICA")
                print(f"{'─' * 100}")

            selected_trees, selected_ids_round, all_entries = self._round_robin_aggregation(
                client_windows=client_windows,
                client_metadata=client_window_metadata,
                round_num=round_num
            )

            for client_id in results.client_ids:
                if client_id in selected_ids_round and selected_ids_round[client_id]:
                    tree_idx = selected_ids_round[client_id][0] if selected_ids_round[client_id] else None
                    if tree_idx is not None and tree_idx < len(client_windows[client_id]):
                        selected_tree = client_windows[client_id][tree_idx]
                        self.global_trees.append(selected_tree)
                        self.global_tree_sources.append(client_id)

                        if client_id not in results.selected_ids:
                            results.selected_ids[client_id] = []
                        results.selected_ids[client_id].append(tree_idx)

            round_log['selected_trees'] = {k: len(v) for k, v in selected_ids_round.items()}
            round_log['trees_after'] = len(self.global_trees)

            if self.verbose:
                print(f"\n   🌳 Árboles en bosque global: {len(self.global_trees)}")
                for cid in results.client_ids:
                    n_selected = len(results.selected_ids.get(cid, []))
                    print(f"      {cid}: {n_selected} árboles seleccionados (total)")

            # FASE 4: Criterio de Parada Progressive Forest
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🛑 FASE 4: CRITERIO DE PARADA (PROGRESSIVE GLOBAL)")
                print(f"{'─' * 100}")

            X_val_server = self.dataset_split.X_val if self.dataset_split.X_val is not None else self.dataset_split.X_test
            y_val_server = self.dataset_split.y_val if self.dataset_split.y_val is not None else self.dataset_split.y_test
            y_val_eval = self.label_svc.transform(y_val_server)

            current_accuracy = self._evaluate_forest_accuracy(
                self.global_trees, X_val_server, y_val_eval
            )
            round_accuracies.append(current_accuracy)
            round_log['round_accuracy'] = current_accuracy
            self.round_logs.append(round_log)

            if previous_accuracy is not None:
                improvement = abs(current_accuracy - previous_accuracy)
                if improvement <= self.convergence_threshold:
                    stop_counter += 1
                    if stop_counter >= 2:
                        results.convergence_round = round_num + 1
                        results.num_rounds = round_num + 1
                        break
                else:
                    stop_counter = 0

            previous_accuracy = current_accuracy
            results.num_rounds = round_num + 1

        if results.convergence_round is None:
            results.convergence_round = results.num_rounds

        # FASE 5-6: Actualización de clientes e Inferencia Híbrida
        results.global_accuracy = self._evaluate_forest_accuracy(
            self.global_trees, self.dataset_split.X_test, results.y_test
        )
        results.global_macro_f1 = self._evaluate_forest_f1(
            self.global_trees, self.dataset_split.X_test, results.y_test
        )
        results.n_trees_global = len(self.global_trees)
        results.global_predictions = self._predict_forest(self.global_trees, self.dataset_split.X_test)

        for client_id in results.client_ids:
            local_trees = self.client_forests.get(client_id, ProactiveForest(
                n_estimators=self.window_size * results.num_rounds,
                alpha=self.config_dict.get('model', {}).get('alpha', 0.1),
                class_names=self.dataset_split.class_names
            )).get_trees()

            external_global_trees = [
                tree for tree, source in zip(self.global_trees, self.global_tree_sources)
                if source != client_id
            ]

            hybrid_preds = self._hybrid_predict(
                X=self.dataset_split.X_test,
                local_trees=local_trees,
                global_trees=external_global_trees,
                local_weight=self.local_weight
            )

            results.client_hybrid_predictions[client_id] = hybrid_preds
            results.client_hybrid_forest_sizes[client_id] = len(local_trees) + len(external_global_trees)
            results.client_accuracies[client_id] = accuracy_score(results.y_test, hybrid_preds)
            results.client_f1_scores[client_id] = f1_score(results.y_test, hybrid_preds, average='macro', zero_division=0)

        self.client_forests.clear()
        import gc
        gc.collect()
        return results

    def _train_window(self, client_id: str, window_index: int, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[List[Any], Dict[str, Any]]:
        n_estimators = self.window_size
        alpha_pf = self.config_dict.get('model', {}).get('alpha', 0.1)
        seed = self.config_dict.get('seed', 42)

        if len(X_train) < 10:
            X_tr_local, X_val_local, y_tr_local, y_val_local = X_train, X_train, y_train, y_train
        else:
            from sklearn.model_selection import train_test_split
            X_tr_local, X_val_local, y_tr_local, y_val_local = train_test_split(
                X_train, y_train, test_size=0.2, random_state=seed
            )

        pf = ProactiveForest(n_estimators=n_estimators, alpha=alpha_pf, verbose=False, class_names=self.dataset_split.class_names)
        pf.fit(X_tr_local, y_tr_local)
        self.client_forests[client_id] = pf
        window_trees = pf.get_trees()

        y_pred_raw = pf.predict(X_val_local)
        y_pred = self.label_svc.transform(y_pred_raw)
        y_val_eval = self.label_svc.transform(y_val_local)

        metrics = {
            'accuracy': accuracy_score(y_val_eval, y_pred),
            'macro_f1': f1_score(y_val_eval, y_pred, average='macro', zero_division=0),
            'pcd': self._calculate_window_pcd(window_trees),
            'tree_f1_scores': [f1_score(y_val_eval, self._predict_tree(t, X_val_local), average='macro', zero_division=0) for t in window_trees]
        }
        return window_trees, metrics

    def _round_robin_aggregation(self, client_windows, client_metadata, round_num) -> Tuple[List[Any], Dict[str, List[int]], List[Any]]:
        rng = np.random.RandomState(seed=round_num)
        client_ids = list(client_windows.keys())
        round_permutation = rng.permutation(client_ids).tolist()

        selected_trees = []
        selected_ids = {cid: [] for cid in client_ids}
        all_entries = []

        for client_id in round_permutation:
            window_trees = client_windows[client_id]
            meta = client_metadata[client_id]
            tree_scores = []
            tree_f1_scores_list = meta.get('tree_f1_scores', [])

            for local_idx, tree in enumerate(window_trees):
                tree_f1 = tree_f1_scores_list[local_idx] if local_idx < len(tree_f1_scores_list) else meta.get('macro_f1', 0.5)
                diversity = self._calculate_diversity(tree, self.global_trees)
                effective_f1_weight = self.f1_weight if len(self.global_trees) > 0 else 1.0
                score = effective_f1_weight * tree_f1 + (1.0 - effective_f1_weight) * diversity
                tree_scores.append((local_idx, tree, score, tree_f1, diversity))

            tree_scores_sorted = sorted(tree_scores, key=lambda x: x[2], reverse=True)
            best_idx, best_tree, best_score, best_f1, best_div = tree_scores_sorted[0]
            selected_trees.append(best_tree)
            selected_ids[client_id].append(best_idx)

            from src.domain.aggregation.tree_ranker import TreeEntry
            all_entries.append(TreeEntry(tree=best_tree, client_id=client_id, tree_local_id=best_idx, accuracy=meta['accuracy'], macro_f1=best_f1, pcd=best_div))

        return selected_trees, selected_ids, all_entries

    def _calculate_diversity(self, tree, global_trees) -> float:
        if not global_trees: return 1.0
        return sum(self._calculate_pcd_between_trees(tree, gt) for gt in global_trees) / len(global_trees)

    def _calculate_pcd_between_trees(self, tree1, tree2) -> float:
        try:
            s1, s2 = (t.get_tree_structure() if hasattr(t, 'get_tree_structure') else t for t in (tree1, tree2))
            d1, d2 = self._get_tree_depth(s1), self._get_tree_depth(s2)
            n1, n2 = self._count_nodes(s1), self._count_nodes(s2)
            return min(1.0, (abs(d1 - d2) / max(d1, d2, 1) + abs(n1 - n2) / max(n1, n2, 1)) / 2)
        except Exception: return 0.5

    def _get_tree_depth(self, s) -> int:
        if isinstance(s, dict) and 'children' in s and s['children']:
            return 1 + max(self._get_tree_depth(c) for c in s['children'])
        return s.get('depth', 1) if isinstance(s, dict) else 1

    def _count_nodes(self, s) -> int:
        if isinstance(s, dict) and 'children' in s:
            return 1 + sum(self._count_nodes(c) for c in s['children'])
        return 1

    def _calculate_window_pcd(self, trees) -> float:
        if len(trees) < 2: return 0.5
        pairs = [(t1, t2) for i, t1 in enumerate(trees) for j, t2 in enumerate(trees) if i < j]
        return sum(self._calculate_pcd_between_trees(*p) for p in pairs) / len(pairs)

    def _evaluate_forest_accuracy(self, trees, X, y) -> float:
        return float(accuracy_score(y, self._predict_forest(trees, X))) if trees else 0.0

    def _evaluate_forest_f1(self, trees, X, y) -> float:
        return float(f1_score(y, self._predict_forest(trees, X), average='macro', zero_division=0)) if trees else 0.0

    def _predict_forest(self, trees, X) -> np.ndarray:
        from scipy import stats
        if not trees: return np.zeros(len(X), dtype=int)
        all_preds = np.column_stack([self._predict_tree(t, X) for t in trees])
        return stats.mode(all_preds, axis=1, keepdims=False).mode

    def _predict_tree(self, tree, X) -> np.ndarray:
        return self.label_svc.transform([tree.predict(x) for x in X])

    def _hybrid_predict(self, X, local_trees, global_trees, local_weight=0.5) -> np.ndarray:
        n_classes = len(self.dataset_split.class_names)
        lp = self._get_class_probabilities(local_trees, X, n_classes)
        gp = self._get_class_probabilities(global_trees, X, n_classes)
        return np.argmax(local_weight * lp + (1 - local_weight) * gp, axis=1)

    def _get_class_probabilities(self, trees, X, n_classes) -> np.ndarray:
        if not trees: return np.ones((len(X), n_classes)) / n_classes
        all_preds = np.column_stack([self._predict_tree(t, X) for t in trees])
        probs = np.zeros((len(X), n_classes))
        for i in range(len(X)):
            counts = np.bincount(all_preds[i, :], minlength=n_classes)
            probs[i, :] = counts[:n_classes]
        return probs / len(trees)
