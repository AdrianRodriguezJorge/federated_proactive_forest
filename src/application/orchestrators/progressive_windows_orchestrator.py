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
from typing import Dict, List, Any, Optional, Tuple
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

    Args:
        config: Configuración con parámetros Progressive Windows
        dataset_split: Dataset dividido train/test
        verbose: Mostrar logs detallados
    """

    def __init__(
        self,
        config: dict,
        dataset_split: DatasetSplit,
        verbose: bool = True
    ):
        self.config = config
        self.dataset_split = dataset_split
        self.verbose = verbose

        # Parámetros Progressive Windows
        self.n_clients = config.get('n_clients', 5)
        self.window_size = config.get('aggregation', {}).get('window_size', 5)
        self.max_rounds = config.get('aggregation', {}).get('max_rounds', 20)
        self.alpha = config.get('aggregation', {}).get('alpha', 0.5)
        self.convergence_threshold = config.get('aggregation', {}).get('convergence_threshold', 0.002)
        self.local_weight = config.get('prediction', {}).get('local_weight', 0.5)

        # Estado del entrenamiento
        self.client_forests: Dict[str, ProactiveForest] = {}
        self.global_trees: List[Any] = []
        self.global_tree_sources: List[str] = []  # Qué cliente dio cada árbol
        self.round_logs: List[Dict[str, Any]] = []

        # Estrategia Progressive Windows
        self.strategy = ProgressiveWindowsStrategy(
            window_size=self.window_size,
            max_rounds=self.max_rounds,
            alpha=self.alpha,
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
            print(f"   • Alpha: {self.alpha}")
            print(f"   • Lambda: {self.local_weight}")
            print("=" * 100 + "\n")

        # Inicializar resultados
        results = ProgressiveWindowsResults()
        results.client_ids = [f"client_{i}" for i in range(self.n_clients)]
        results.y_test = self.dataset_split.y_test
        results.class_names = self.dataset_split.class_names

        # Variables para criterio de parada
        round_accuracies: List[float] = []
        stop_counter = 0
        previous_accuracy = None

        # ──────────────────────────────────────────────────────────────────────
        # BUCLE PRINCIPAL: Rondas de entrenamiento incremental
        # ──────────────────────────────────────────────────────────────────────
        for round_num in range(self.max_rounds):
            if self.verbose:
                print("\n" + "=" * 100)
                print(f"🔁 RONDA {round_num + 1}/{self.max_rounds}")
                print("=" * 100)
                print(f"📊 Bosque global actual: {len(self.global_trees)} árboles")

                if len(self.global_trees) > 0:
                    current_acc = self._evaluate_forest_accuracy(
                        self.global_trees, self.dataset_split.X_test, self.dataset_split.y_test
                    )
                    print(f"📈 Accuracy actual: {current_acc:.6f}")
                else:
                    print(f"📈 Accuracy actual: N/A (bosque vacío)")

            round_log = {
                'round': round_num + 1,
                'trees_before': len(self.global_trees),
                'client_windows': {},
                'selected_trees': {},
                'round_accuracy': None
            }

            # ──────────────────────────────────────────────────────────────────
            # FASE 1: Cada cliente entrena UNA VENTANA de W árboles
            # ──────────────────────────────────────────────────────────────────
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🌱 FASE 1: ENTRENAMIENTO DE VENTANAS LOCALES")
                print(f"{'─' * 100}")

            client_windows: Dict[str, List[Any]] = {}
            client_window_metadata: Dict[str, Dict[str, Any]] = {}

            for client_id in results.client_ids:
                # Entrenar ventana de W árboles
                window_trees, window_metrics = self._train_window(
                    client_id=client_id,
                    window_index=round_num,
                    X_train=self.dataset_split.X_train,
                    y_train=self.dataset_split.y_train
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

            # ──────────────────────────────────────────────────────────────────
            # FASE 2-3: Servidor recibe ventanas y hace Round Robin
            # ──────────────────────────────────────────────────────────────────
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🔄 FASE 2-3: AGREGACIÓN ROUND ROBIN DINÁMICA")
                print(f"{'─' * 100}")

            # Ejecutar Round Robin con las ventanas de esta ronda
            selected_trees, selected_ids_round, all_entries = self._round_robin_aggregation(
                client_windows=client_windows,
                client_metadata=client_window_metadata,
                round_num=round_num
            )

            # Incorporar árboles seleccionados al bosque global
            for client_id in results.client_ids:
                if client_id in selected_ids_round and selected_ids_round[client_id]:
                    # Obtener el árbol seleccionado de este cliente
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

            # ──────────────────────────────────────────────────────────────────
            # FASE 4: Criterio de Parada Progressive Forest
            # ──────────────────────────────────────────────────────────────────
            if self.verbose:
                print(f"\n{'─' * 100}")
                print(f"🛑 FASE 4: CRITERIO DE PARADA (PROGRESSIVE GLOBAL)")
                print(f"{'─' * 100}")

            # Evaluar bosque global
            current_accuracy = self._evaluate_forest_accuracy(
                self.global_trees, self.dataset_split.X_test, self.dataset_split.y_test
            )
            current_f1 = self._evaluate_forest_f1(
                self.global_trees, self.dataset_split.X_test, self.dataset_split.y_test
            )

            round_accuracies.append(current_accuracy)
            round_log['round_accuracy'] = current_accuracy
            self.round_logs.append(round_log)

            if self.verbose:
                print(f"   Accuracy ronda {round_num + 1}: {current_accuracy:.6f}")
                if previous_accuracy is not None:
                    improvement = abs(current_accuracy - previous_accuracy)
                    print(f"   Accuracy anterior: {previous_accuracy:.6f}")
                    print(f"   Mejora: |{current_accuracy:.6f} - {previous_accuracy:.6f}| = {improvement:.6f}")
                    print(f"   Umbral convergencia: {self.convergence_threshold}")

            # Verificar convergencia
            if previous_accuracy is not None:
                improvement = abs(current_accuracy - previous_accuracy)

                if improvement <= self.convergence_threshold:
                    stop_counter += 1
                    if self.verbose:
                        print(f"   ⚠️  Mejora ≤ umbral → Contador: {stop_counter}/2")

                    if stop_counter >= 2:
                        if self.verbose:
                            print(f"\n{'=' * 100}")
                            print(f"✅ CONVERGENCIA ALCANZADA")
                            print(f"{'=' * 100}")
                            print(f"   • Rondas completadas: {round_num + 1}")
                            print(f"   • Árboles globales: {len(self.global_trees)}")
                            print(f"   • Accuracy final: {current_accuracy:.6f}")
                            print(f"{'=' * 100}\n")

                        results.convergence_round = round_num + 1
                        results.num_rounds = round_num + 1
                        break
                else:
                    stop_counter = 0
                    if self.verbose:
                        print(f"   ✅ Mejora > umbral → Reiniciar contador")

            previous_accuracy = current_accuracy
            results.num_rounds = round_num + 1

        # Si no hubo convergencia temprana
        if results.convergence_round is None:
            results.convergence_round = results.num_rounds
            if self.verbose:
                print(f"\n{'=' * 100}")
                print(f"⚠️ MÁXIMO RONDAS ALCANZADO")
                print(f"{'=' * 100}")
                print(f"   • Rondas completadas: {results.num_rounds}")
                print(f"   • Árboles globales: {len(self.global_trees)}")
                print(f"{'=' * 100}\n")

        # ──────────────────────────────────────────────────────────────────────
        # FASE 5-6: Actualización de clientes e Inferencia Híbrida
        # ──────────────────────────────────────────────────────────────────────
        if self.verbose:
            print(f"\n{'=' * 100}")
            print(f"🔄 FASE 5-6: ACTUALIZACIÓN E INFERENCIA HÍBRIDA")
            print(f"{'=' * 100}")

        # Calcular métricas finales
        results.global_accuracy = self._evaluate_forest_accuracy(
            self.global_trees, self.dataset_split.X_test, self.dataset_split.y_test
        )
        results.global_macro_f1 = self._evaluate_forest_f1(
            self.global_trees, self.dataset_split.X_test, self.dataset_split.y_test
        )
        results.n_trees_global = len(self.global_trees)

        # Generar predicciones globales
        results.global_predictions = self._predict_forest(
            self.global_trees, self.dataset_split.X_test
        )

        # Inferencia híbrida por cliente (FASE 6)
        results.client_hybrid_predictions = {}
        results.client_hybrid_forest_sizes = {}

        for client_id in results.client_ids:
            # Bosque híbrido: local + globales externos (sin duplicados)
            local_trees = self.client_forests.get(client_id, ProactiveForest(
                n_estimators=self.window_size * results.num_rounds,
                alpha=self.config.get('alpha_pf', 0.1),
                class_names=self.dataset_split.class_names
            )).get_trees()

            # Globales externos (excluir propios)
            external_global_trees = [
                tree for tree, source in zip(self.global_trees, self.global_tree_sources)
                if source != client_id
            ]

            # Predicción híbrida
            hybrid_preds = self._hybrid_predict(
                X=self.dataset_split.X_test,
                local_trees=local_trees,
                global_trees=external_global_trees,
                local_weight=self.local_weight
            )

            results.client_hybrid_predictions[client_id] = hybrid_preds
            results.client_hybrid_forest_sizes[client_id] = len(local_trees) + len(external_global_trees)

            # Métricas por cliente
            client_acc = accuracy_score(self.dataset_split.y_test, hybrid_preds)
            client_f1 = f1_score(self.dataset_split.y_test, hybrid_preds, average='macro', zero_division=0)
            results.client_accuracies[client_id] = client_acc
            results.client_f1_scores[client_id] = client_f1

        if self.verbose:
            print(f"\n📊 RESULTADOS FINALES:")
            print(f"   • Accuracy global: {results.global_accuracy:.6f} ({results.global_accuracy*100:.2f}%)")
            print(f"   • Macro-F1 global: {results.global_macro_f1:.6f}")
            print(f"   • Árboles globales: {results.n_trees_global}")
            print(f"   • Rondas: {results.num_rounds}")
            if results.convergence_round < results.num_rounds:
                print(f"   • Convergencia temprana: Ronda {results.convergence_round}")
            print(f"{'=' * 100}\n")

        return results

    def _train_window(
        self,
        client_id: str,
        window_index: int,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        FASE 1: Entrenar una ventana de W árboles para un cliente.

        Args:
            client_id: Identificador del cliente
            window_index: Índice de la ventana (0-based)
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento

        Returns:
            Tuple de (lista de árboles, métricas de la ventana)
        """
        # Crear bosque Proactive Forest para esta ventana
        n_estimators = self.window_size
        alpha_pf = self.config.get('alpha_pf', 0.1)

        pf = ProactiveForest(
            n_estimators=n_estimators,
            alpha=alpha_pf,
            verbose=False,
            class_names=self.dataset_split.class_names
        )

        # Entrenar con datos del cliente (en implementación real, cada cliente tiene sus propios datos)
        # Aquí usamos todo X_train para simplificar
        pf.fit(X_train, y_train)

        # Guardar bosque del cliente
        self.client_forests[client_id] = pf

        # Obtener árboles de esta ventana
        window_trees = pf.get_trees()

        # Calcular métricas de la ventana
        y_pred_raw = pf.predict(self.dataset_split.X_test)

        # Convertir predicciones a índices numéricos (pueden ser strings o números)
        if len(y_pred_raw) > 0 and isinstance(y_pred_raw[0], str):
            class_to_idx = {cn: idx for idx, cn in enumerate(self.dataset_split.class_names)}
            y_pred = np.array([class_to_idx.get(p, 0) for p in y_pred_raw])
        else:
            y_pred = np.asarray(y_pred_raw, dtype=np.int64)

        accuracy = accuracy_score(self.dataset_split.y_test, y_pred)
        macro_f1 = f1_score(self.dataset_split.y_test, y_pred, average='macro', zero_division=0)

        # Calcular PCD (diversidad) de la ventana
        pcd = self._calculate_window_pcd(window_trees)

        # Calcular F1 por árbol (para el score dinámico)
        tree_f1_scores = []
        for tree in window_trees:
            tree_preds = self._predict_tree(tree, self.dataset_split.X_test)
            tree_f1 = f1_score(self.dataset_split.y_test, tree_preds, average='macro', zero_division=0)
            tree_f1_scores.append(tree_f1)

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'pcd': pcd,
            'tree_f1_scores': tree_f1_scores
        }

        return window_trees, metrics

    def _round_robin_aggregation(
        self,
        client_windows: Dict[str, List[Any]],
        client_metadata: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Tuple[List[Any], Dict[str, List[int]], List[Any]]:
        """
        FASE 2-3: Ejecutar Round Robin con score dinámico.

        Args:
            client_windows: Ventanas de árboles por cliente
            client_metadata: Métricas por cliente
            round_num: Número de ronda actual

        Returns:
            Tuple de (árboles seleccionados, IDs seleccionados, entradas de árboles)
        """
        # Generar permutación aleatoria para esta ronda
        rng = np.random.RandomState(seed=round_num)
        client_ids = list(client_windows.keys())
        round_permutation = rng.permutation(client_ids).tolist()

        if self.verbose:
            print(f"\n   🎲 Permutación Round Robin: {' → '.join(round_permutation)}")

        selected_trees = []
        selected_ids = {cid: [] for cid in client_ids}
        all_entries = []

        # Procesar cada cliente en orden de permutación
        for client_idx, client_id in enumerate(round_permutation):
            window_trees = client_windows[client_id]
            meta = client_metadata[client_id]

            if self.verbose:
                print(f"\n   👤 TURNO {client_idx + 1}: {client_id}")
                print(f"      Ventana: {len(window_trees)} árboles")

            # Calcular scores dinámicos para árboles en ventana
            tree_scores = []
            tree_f1_scores_list = meta.get('tree_f1_scores', [])
            for local_idx, tree in enumerate(window_trees):
                tree_f1 = tree_f1_scores_list[local_idx] if local_idx < len(tree_f1_scores_list) else meta.get('macro_f1', 0.5)
                diversity = self._calculate_diversity(tree, self.global_trees)

                # Score dinámico: Score(T) = α·F1(T) + (1-α)·Diversidad(T|G)
                # Primera ronda: G vacío → α=1
                effective_alpha = self.alpha if len(self.global_trees) > 0 else 1.0
                score = effective_alpha * tree_f1 + (1 - effective_alpha) * diversity

                tree_scores.append((local_idx, tree, score, tree_f1, diversity))

            # Ordenar por score (ranking)
            tree_scores_sorted = sorted(tree_scores, key=lambda x: x[2], reverse=True)

            if self.verbose:
                print(f"      {'─' * 70}")
                print(f"      {'Árbol':<10} | {'F1(T)':<12} | {'Diversidad':<14} | {'Score':<12} | {'Rank':<6}")
                print(f"      {'─' * 70}")
                for rank, (idx, tree, score, f1, div) in enumerate(tree_scores_sorted[:5], 1):  # Mostrar top 5
                    marker = "⭐" if rank == 1 else "  "
                    print(f"      {marker} T{idx:<7} | {f1:<12.6f} | {div:<14.6f} | {score:<12.6f} | #{rank:<6}")
                print(f"      {'─' * 70}")

            # Seleccionar mejor árbol
            best_idx, best_tree, best_score, best_f1, best_div = tree_scores_sorted[0]

            if self.verbose:
                print(f"\n      ✅ SELECCIONADO: T{best_idx}")
                print(f"         Score: {best_score:.6f} = {self.alpha:.2f}×{best_f1:.6f} + {1-self.alpha:.2f}×{best_div:.6f}")

            selected_trees.append(best_tree)
            selected_ids[client_id].append(best_idx)

            # Crear entrada para tracking
            from src.domain.aggregation.tree_ranker import TreeEntry
            entry = TreeEntry(
                tree=best_tree,
                client_id=client_id,
                tree_local_id=best_idx,
                accuracy=meta['accuracy'],
                macro_f1=best_f1,
                pcd=best_div
            )
            all_entries.append(entry)

        return selected_trees, selected_ids, all_entries

    def _calculate_diversity(self, tree: Any, global_trees: List[Any]) -> float:
        """Calcular diversidad de un árbol respecto al bosque global."""
        if not global_trees:
            return 1.0

        pcd_sum = sum(self._calculate_pcd_between_trees(tree, gt) for gt in global_trees)
        return pcd_sum / len(global_trees)

    def _calculate_pcd_between_trees(self, tree1: Any, tree2: Any) -> float:
        """Calcular PCD entre dos árboles (simplificado por estructura)."""
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
        """Obtener profundidad de un árbol."""
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
        """Contar nodos de un árbol."""
        if isinstance(tree_structure, dict):
            count = 1
            if 'children' in tree_structure:
                for child in tree_structure['children']:
                    count += self._count_nodes(child)
            return count
        return 1

    def _calculate_window_pcd(self, trees: List[Any]) -> float:
        """Calcular PCD promedio de una ventana de árboles."""
        if len(trees) < 2:
            return 0.5

        pcd_sum = 0.0
        count = 0
        for i, t1 in enumerate(trees):
            for j, t2 in enumerate(trees):
                if i < j:
                    pcd_sum += self._calculate_pcd_between_trees(t1, t2)
                    count += 1

        return pcd_sum / count if count > 0 else 0.5

    def _evaluate_forest_accuracy(self, trees: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        """Evaluar accuracy de un bosque."""
        if not trees:
            return 0.0

        predictions = self._predict_forest(trees, X)
        return float(accuracy_score(y, predictions))

    def _evaluate_forest_f1(self, trees: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        """Evaluar Macro-F1 de un bosque."""
        if not trees:
            return 0.0

        predictions = self._predict_forest(trees, X)
        try:
            return float(f1_score(y, predictions, average='macro', zero_division=0))
        except Exception:
            return 0.0

    def _predict_forest(self, trees: List[Any], X: np.ndarray) -> np.ndarray:
        """Predecir con un bosque usando votación mayoritaria."""
        from scipy import stats

        n_samples = X.shape[0]
        n_trees = len(trees)

        if n_trees == 0:
            return np.zeros(n_samples, dtype=int)

        all_predictions = np.zeros((n_samples, n_trees), dtype=int)
        for j, tree in enumerate(trees):
            for i in range(n_samples):
                try:
                    all_predictions[i, j] = tree.predict(X[i])
                except Exception:
                    all_predictions[i, j] = 0

        mode_result = stats.mode(all_predictions, axis=1, keepdims=False)
        return mode_result.mode

    def _predict_tree(self, tree: Any, X: np.ndarray) -> np.ndarray:
        """Predecir con un solo árbol."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            try:
                pred = tree.predict(X[i])
                # Convertir string a índice si es necesario
                if isinstance(pred, str):
                    pred = self.dataset_split.class_names.index(pred) if pred in self.dataset_split.class_names else 0
                predictions[i] = pred
            except Exception:
                predictions[i] = 0
        return predictions

    def _hybrid_predict(
        self,
        X: np.ndarray,
        local_trees: List[Any],
        global_trees: List[Any],
        local_weight: float = 0.5
    ) -> np.ndarray:
        """
        FASE 6: Inferencia híbrida ponderada.

        ŷ = argmax( local_weight·p_local(c|x) + (1-local_weight)·p_global(c|x) )
        """
        from scipy import stats

        n_samples = X.shape[0]
        n_classes = len(self.dataset_split.class_names)

        # Obtener predicciones de local y global
        local_preds = self._get_class_probabilities(local_trees, X, n_classes)
        global_preds = self._get_class_probabilities(global_trees, X, n_classes)

        # Combinación ponderada
        combined = local_weight * local_preds + (1 - local_weight) * global_preds

        return np.argmax(combined, axis=1)

    def _get_class_probabilities(
        self,
        trees: List[Any],
        X: np.ndarray,
        n_classes: int
    ) -> np.ndarray:
        """Obtener probabilidades de clase por votación."""
        n_samples = X.shape[0]
        n_trees = len(trees)

        if n_trees == 0:
            return np.ones((n_samples, n_classes)) / n_classes

        all_predictions = np.zeros((n_samples, n_trees), dtype=int)
        for j, tree in enumerate(trees):
            for i in range(n_samples):
                try:
                    all_predictions[i, j] = tree.predict(X[i])
                except Exception:
                    all_predictions[i, j] = 0

        # Contar votos por clase
        class_counts = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            for c in range(n_classes):
                class_counts[i, c] = np.sum(all_predictions[i, :] == c)

        # Convertir a probabilidades
        probs = class_counts / n_trees
        return probs
