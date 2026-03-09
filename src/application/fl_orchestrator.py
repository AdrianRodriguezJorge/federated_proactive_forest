"""
Orquestador de la ronda federada única.

Flujo completo:
  INIT → TRAIN (con CPF) → COLLECT metadatos → AGGREGATE → UPDATE clientes → EVALUATE
"""
from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from proactive_forest.estimator import ProactiveForestClassifier
from proactive_forest.newalg import ComparativeProgressiveForest

from src.domain.aggregation.strategies import build_strategy
from src.domain.aggregation.tree_ranker import TreeEntry, TreeRanker
from src.domain.metadata.client_metadata import ClientMetadata
from src.domain.metrics.forest_evaluator import ForestEvaluator, ForestReport
from src.domain.prediction.hybrid_predictor import HybridPredictor
from src.domain.update.client_updater import ClientUpdater
from src.domain.dataset.base_adapter import DatasetSplit


# ── Resultado de la ronda ──────────────────────────────────────────────────────
@dataclass
class FLResults:
    strategy_id: str
    global_accuracy: float
    global_macro_f1: float
    n_trees_global: int
    client_ids: List[str]
    global_report: ForestReport
    client_reports: Dict[str, ForestReport]
    # Para la UI de ranking
    all_tree_entries: List[TreeEntry]
    selected_ids: Dict[str, List[int]]
    client_metadata: Dict[str, ClientMetadata]


# ── Orquestador ────────────────────────────────────────────────────────────────
class FLOrchestrator:
    """
    Ejecuta una ronda federada completa con ProactiveForestClassifier + CPF.
    """

    def __init__(self, dataset_split: DatasetSplit, config: dict,
                 step_callback: Optional[Callable] = None):
        self.ds = dataset_split
        self.cfg = config
        self.cb = step_callback or (lambda *a, **kw: None)

    @classmethod
    def from_config(cls, config: dict, step_callback=None) -> "FLOrchestrator":
        """
        Construye el orquestador desde un dict de config cargado del YAML o la UI.
        Espera config['_dataset_split'] ya cargado por la UI / CLI.
        """
        return cls(dataset_split=config['_dataset_split'],
                   config=config, step_callback=step_callback)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_forest(self) -> ProactiveForestClassifier:
        m = self.cfg.get('model', {})
        return ProactiveForestClassifier(
            n_estimators=m.get('n_estimators', 100),
            alpha=m.get('alpha', 0.1),
            bootstrap=m.get('bootstrap', True),
            split_criterion=m.get('split_criterion', 'entropy'),
            feature_selection=m.get('feature_selection', 'prob'),
        )

    def _partition(self) -> Dict[str, tuple]:
        """Particiona X_train en n_clients fragmentos (IID o Dirichlet non-IID)."""
        n = self.cfg.get('federation', {}).get('n_clients', 5)
        dist = self.cfg.get('federation', {}).get('distribution', 'iid')
        X, y = self.ds.X_train, self.ds.y_train
        indices = np.arange(len(y))

        if dist == 'iid':
            splits = np.array_split(indices, n)
        elif dist == 'noniid_dirichlet':
            alpha_d = self.cfg.get('federation', {}).get('dirichlet_alpha', 0.5)
            classes = np.unique(y)
            splits = [[] for _ in range(n)]
            for c in classes:
                idx_c = indices[y == c]
                np.random.shuffle(idx_c)
                proportions = np.random.dirichlet(np.repeat(alpha_d, n))
                proportions = (proportions * len(idx_c)).astype(int)
                proportions[-1] = len(idx_c) - proportions[:-1].sum()
                ptr = 0
                for i, p in enumerate(proportions):
                    splits[i].extend(idx_c[ptr:ptr + p].tolist())
                    ptr += p
        else:
            splits = np.array_split(indices, n)

        clients = {}
        for i, idx in enumerate(splits):
            idx = np.array(idx)
            if len(idx) == 0:
                continue
            cid = f"client_{i}"
            clients[cid] = (X[idx], y[idx])
        return clients

    # ── Ronda principal ───────────────────────────────────────────────────────
    def run(self) -> FLResults:
        val_split = self.cfg.get('metadata', {}).get('validation_split', 0.2)
        use_cpf   = self.cfg.get('model', {}).get('use_progressive_stopping', True)
        verbose   = self.cfg.get('verbose', False)

        self.cb("1/6 Particionando datos", 5)
        client_partitions = self._partition()
        client_ids = list(client_partitions.keys())

        # ── PASO 2: TRAIN local ───────────────────────────────────────────────
        self.cb("2/6 Entrenando bosques locales", 15)
        client_forests: Dict[str, ProactiveForestClassifier] = {}
        client_metadata: Dict[str, ClientMetadata] = {}

        for cid, (Xc, yc) in client_partitions.items():
            # Split val local para calcular metadatos
            if val_split > 0 and len(Xc) > 10:
                Xtr, Xval, ytr, yval = train_test_split(Xc, yc, test_size=val_split,
                                                        random_state=42)
            else:
                Xtr, Xval, ytr, yval = Xc, Xc, yc, yc

            pf = self._build_forest()
            if use_cpf:
                cpf = ComparativeProgressiveForest(pf, verbose=verbose)
                cpf.fit(Xtr, ytr, Xval, yval)
                pf = cpf.return_forest()
            else:
                pf.fit(Xtr, ytr)

            # Calcular metadatos sobre val split
            y_pred_val = pf.predict(Xval)
            acc = float(accuracy_score(yval, y_pred_val))
            f1  = float(f1_score(yval, y_pred_val, average='macro', zero_division=0))
            try:
                pcd = float(pf.diversity_measure(Xval, yval, diversity='pcd'))
            except Exception:
                pcd = 0.0

            client_forests[cid] = pf
            client_metadata[cid] = ClientMetadata(
                client_id=cid,
                n_trees=len(pf.get_trees()),
                accuracy=acc,
                macro_f1=f1,
                pcd=pcd,
            )

        # ── PASO 3: COLLECT árboles ───────────────────────────────────────────
        self.cb("3/6 Recolectando árboles de clientes", 40)
        client_trees = {cid: pf.get_trees() for cid, pf in client_forests.items()}

        # Construir all_tree_entries para UI (antes de agregar)
        all_entries = TreeRanker.build_entries(client_trees, client_metadata)

        # ── PASO 4: AGGREGATE ─────────────────────────────────────────────────
        self.cb("4/6 Agregando bosque global", 55)
        agg_cfg = self.cfg.get('aggregation', {})
        strategy = build_strategy(agg_cfg)
        global_trees, selected_ids = strategy.aggregate(client_trees, client_metadata)

        # Anotar selected_local_tree_ids en los metadatos
        for cid, ids in selected_ids.items():
            client_metadata[cid].selected_local_tree_ids = ids

        # Actualizar scores en all_entries para la UI de ranking
        for entry in all_entries:
            entry_ids = selected_ids.get(entry.client_id, [])
            # score ya calculado en tree_ranker; si S1 no tiene score lo ponemos a 0
            if not hasattr(entry, 'score') or entry.score == 0.0:
                entry.score = entry.accuracy  # fallback para S1

        # ── PASO 5: UPDATE clientes ───────────────────────────────────────────
        self.cb("5/6 Actualizando bosques locales (No-Repeat Merge)", 70)
        pred_cfg = self.cfg.get('prediction', {})
        local_w  = pred_cfg.get('local_weight', 0.4)
        global_w = pred_cfg.get('global_weight', 0.6)
        n_classes = len(self.ds.class_names)

        for cid, pf in client_forests.items():
            merged = ClientUpdater.merge(
                local_trees=pf.get_trees(),
                global_trees=global_trees,
                selected_local_ids=selected_ids.get(cid, []),
            )
            pf.set_trees(merged)

        # ── PASO 6: EVALUATE ──────────────────────────────────────────────────
        self.cb("6/6 Evaluando modelos", 85)
        Xt, yt = self.ds.X_test, self.ds.y_test
        class_names = self.ds.class_names

        # Bosque global (usando un forest proxy)
        global_forest_proxy = copy.copy(list(client_forests.values())[0])
        global_forest_proxy.set_trees(global_trees)
        global_report = ForestEvaluator.evaluate(global_forest_proxy, Xt, yt, class_names)

        # Por cliente (bosque extendido local+global)
        client_reports = {}
        for cid, pf in client_forests.items():
            client_reports[cid] = ForestEvaluator.evaluate(pf, Xt, yt, class_names)

        self.cb("Completado", 100)

        return FLResults(
            strategy_id=strategy.strategy_id,
            global_accuracy=global_report.accuracy,
            global_macro_f1=global_report.macro_f1,
            n_trees_global=len(global_trees),
            client_ids=client_ids,
            global_report=global_report,
            client_reports=client_reports,
            all_tree_entries=all_entries,
            selected_ids=selected_ids,
            client_metadata=client_metadata,
        )
