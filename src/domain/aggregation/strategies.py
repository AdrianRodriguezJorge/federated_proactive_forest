"""
Las 7 estrategias de agregación.

S1: Simple Pool — todos los árboles sin ordenar
S2–S4: Lista global ordenada + Progressive stopping
S5–S7: Una lista por cliente (round-robin) + Progressive stopping

Criterios: Accuracy (S2/S5) | Macro-F1 (S3/S6) | α·F1+β·PCD (S4/S7)
"""
from typing import Any, Dict, List, Tuple
from src.domain.aggregation.base_strategy import IAggregationStrategy
from src.domain.aggregation.tree_ranker import TreeRanker, TreeEntry, RankingCriterion
from src.domain.aggregation.cpf_stopper import ProgressiveStopper


# ── S1: Simple Pool ────────────────────────────────────────────────────────────
class S1SimplePool(IAggregationStrategy):
    """Todos los árboles sin ordenar. Baseline de comparación."""

    @property
    def strategy_id(self) -> str:
        return "s1_simple_pool"

    def aggregate(self, client_trees, client_metadata):
        all_trees, selected_ids = [], {}
        for cid, trees in client_trees.items():
            all_trees.extend(trees)
            selected_ids[cid] = list(range(len(trees)))
        return all_trees, selected_ids


# ── S2–S4: Global sorted ──────────────────────────────────────────────────────
class _GlobalSortedStrategy(IAggregationStrategy):
    """Base para S2, S3, S4: lista global ordenada por criterio + Progressive stopping."""

    def __init__(self, criterion: RankingCriterion, f1_weight: float = 0.5,
                 pcd_weight: float = 0.5, convergence: float = 0.002,
                 episode_size: int = 5):
        self._ranker = TreeRanker(criterion, f1_weight, pcd_weight)
        self._stopper_cfg = dict(convergence=convergence, episode_size=episode_size)

    def aggregate(self, client_trees, client_metadata):
        entries = TreeRanker.build_entries(client_trees, client_metadata)
        ranked = self._ranker.rank(entries)
        stopper = ProgressiveStopper(**self._stopper_cfg)
        stopper.reset()

        selected_trees = []
        selected_ids: Dict[str, List[int]] = {cid: [] for cid in client_trees}
        # Accuracy proxy: usamos el score del árbol como proxy de accuracy per-tree
        acc_sequence = []

        for entry in ranked:
            selected_trees.append(entry.tree)
            selected_ids[entry.client_id].append(entry.tree_local_id)
            acc_sequence.append(entry.accuracy)
            if stopper.should_stop(acc_sequence):
                break

        return selected_trees, selected_ids


class S2GlobalAccuracy(_GlobalSortedStrategy):
    def __init__(self, **kw):
        super().__init__(RankingCriterion.ACCURACY, **kw)

    @property
    def strategy_id(self) -> str:
        return "s2_global_accuracy"


class S3GlobalF1(_GlobalSortedStrategy):
    def __init__(self, **kw):
        super().__init__(RankingCriterion.MACRO_F1, **kw)

    @property
    def strategy_id(self) -> str:
        return "s3_global_f1"


class S4GlobalF1PCD(_GlobalSortedStrategy):
    def __init__(self, f1_weight: float = 0.5, pcd_weight: float = 0.5, **kw):
        super().__init__(RankingCriterion.F1_PCD, f1_weight=f1_weight,
                         pcd_weight=pcd_weight, **kw)

    @property
    def strategy_id(self) -> str:
        return "s4_global_f1_pcd"


# ── S5–S7: Per-client sorted ──────────────────────────────────────────────────
class _PerClientStrategy(IAggregationStrategy):
    """Base para S5, S6, S7: una cola por cliente, round-robin + Progressive stopping."""

    def __init__(self, criterion: RankingCriterion, f1_weight: float = 0.5,
                 pcd_weight: float = 0.5, convergence: float = 0.002,
                 episode_size: int = 5):
        self._ranker = TreeRanker(criterion, f1_weight, pcd_weight)
        self._stopper_cfg = dict(convergence=convergence, episode_size=episode_size)

    def aggregate(self, client_trees, client_metadata):
        per_client_queues = {}
        for cid, trees in client_trees.items():
            entries = TreeRanker.build_entries({cid: trees}, {cid: client_metadata[cid]})
            per_client_queues[cid] = self._ranker.rank(entries)

        stopper = ProgressiveStopper(**self._stopper_cfg)
        stopper.reset()
        selected_trees = []
        selected_ids: Dict[str, List[int]] = {cid: [] for cid in client_trees}
        pointers = {cid: 0 for cid in client_trees}
        client_ids = list(client_trees.keys())
        acc_sequence = []

        while True:
            added_this_round = 0
            for cid in client_ids:
                ptr = pointers[cid]
                queue = per_client_queues[cid]
                if ptr < len(queue):
                    entry = queue[ptr]
                    selected_trees.append(entry.tree)
                    selected_ids[cid].append(entry.tree_local_id)
                    pointers[cid] += 1
                    acc_sequence.append(entry.accuracy)
                    added_this_round += 1
                    if stopper.should_stop(acc_sequence):
                        return selected_trees, selected_ids
            if added_this_round == 0:
                break

        return selected_trees, selected_ids


class S5PerClientAccuracy(_PerClientStrategy):
    def __init__(self, **kw):
        super().__init__(RankingCriterion.ACCURACY, **kw)

    @property
    def strategy_id(self) -> str:
        return "s5_perclient_accuracy"


class S6PerClientF1(_PerClientStrategy):
    def __init__(self, **kw):
        super().__init__(RankingCriterion.MACRO_F1, **kw)

    @property
    def strategy_id(self) -> str:
        return "s6_perclient_f1"


class S7PerClientF1PCD(_PerClientStrategy):
    def __init__(self, f1_weight: float = 0.5, pcd_weight: float = 0.5, **kw):
        super().__init__(RankingCriterion.F1_PCD, f1_weight=f1_weight,
                         pcd_weight=pcd_weight, **kw)

    @property
    def strategy_id(self) -> str:
        return "s7_perclient_f1_pcd"


# ── Factory ────────────────────────────────────────────────────────────────────
def build_strategy(cfg: dict) -> IAggregationStrategy:
    """Instancia la estrategia correcta desde el dict de config."""
    s = cfg.get('strategy', 's1_simple_pool')
    f1_w = cfg.get('f1_weight', 0.5)
    pcd_w = cfg.get('pcd_weight', 0.5)
    conv = cfg.get('convergence', 0.002)
    ep = cfg.get('episode_size', 5)
    kw = dict(convergence=conv, episode_size=ep)
    mapping = {
        's1_simple_pool':        S1SimplePool,
        's2_global_accuracy':    lambda: S2GlobalAccuracy(**kw),
        's3_global_f1':          lambda: S3GlobalF1(**kw),
        's4_global_f1_pcd':      lambda: S4GlobalF1PCD(f1_weight=f1_w, pcd_weight=pcd_w, **kw),
        's5_perclient_accuracy': lambda: S5PerClientAccuracy(**kw),
        's6_perclient_f1':       lambda: S6PerClientF1(**kw),
        's7_perclient_f1_pcd':   lambda: S7PerClientF1PCD(f1_weight=f1_w, pcd_weight=pcd_w, **kw),
    }
    if s not in mapping:
        raise ValueError(f"Estrategia desconocida: {s}. Opciones: {list(mapping.keys())}")
    factory = mapping[s]
    return factory() if callable(factory) and not isinstance(factory, type) else factory()
