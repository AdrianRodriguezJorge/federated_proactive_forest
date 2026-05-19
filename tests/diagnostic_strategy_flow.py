"""Focused diagnostic: verify benchmark-equivalent flow for S2-S7."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.domain.services.label_service import SimpleLabelService
from src.domain.aggregation.tree_ranker import TreeRanker, TreeEntry, RankingCriterion
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector
from src.domain.aggregation.strategies.global_progressive_base import S2GlobalAccuracyStrategy, S3GlobalF1Strategy, S4GlobalF1PCDStrategy
from src.domain.aggregation.strategies.perclient_progressive_base import S5PerClientAccuracyStrategy
from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService
from src.infrastructure.metrics.diversity_service import PredictionBasedDiversityService
from src.domain.model.proactive_forest import ProactiveForest


def test_strategy_on_wine():
    """Use Wine dataset with proper splits to test S2-S7 behavior."""
    wine = load_wine()
    X, y_int = wine.data, wine.target
    y = y_int.astype(str)
    class_names = [str(c) for c in sorted(np.unique(y))]
    print(f"Dataset: Wine, classes={class_names}, n_samples={len(X)}")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Simulate 3 federated clients
    n = len(X_tr) // 3
    client_trees = {}
    client_meta = {}
    for i in range(3):
        start, end = i*n, (i+1)*n if i < 2 else len(X_tr)
        m = ProactiveForest(n_estimators=30, alpha=0.1, class_names=class_names, random_state=i*10)
        m.fit(X_tr[start:end], y_tr[start:end])
        cid = f"client_{i}"
        trees = m.get_trees()
        client_trees[cid] = trees
        client_meta[cid] = {}
        print(f"  Client {i}: {end-start} samples, {len(trees)} trees")

    metrics_svc = SklearnMetricsService()
    diversity_svc = PredictionBasedDiversityService()
    label_svc = SimpleLabelService(class_names)

    # Test each strategy
    strategies = {
        "S1 (pool)": None,
        "S2 (global acc)": S2GlobalAccuracyStrategy(metrics_svc, diversity_svc),
        "S3 (global f1)": S3GlobalF1Strategy(metrics_svc, diversity_svc),
        "S4 (global f1+pcd)": S4GlobalF1PCDStrategy(metrics_svc, diversity_svc),
        "S5 (perclient acc)": S5PerClientAccuracyStrategy(metrics_svc, diversity_svc),
    }

    label_svc_test = SimpleLabelService(class_names)
    y_test_num = label_svc_test.transform(y_test)

    for name, strategy in strategies.items():
        if strategy is None:
            # S1: pool all
            all_trees = []
            for trees in client_trees.values():
                all_trees.extend(trees)
            # Predict via majority vote
            from src.domain.prediction.voting import calculate_mode
            n_samples = X_test.shape[0]
            pred_matrix = np.empty((n_samples, len(all_trees)), dtype=object)
            for j, tree in enumerate(all_trees):
                pred_matrix[:, j] = tree.predict(X_test)
            preds_raw = calculate_mode(pred_matrix, axis=1)
            preds = label_svc.transform(preds_raw)
            n_selected = len(all_trees)
            conv = None
        else:
            result = strategy.aggregate(
                client_trees=client_trees,
                client_metadata=client_meta,
                X_val=X_val, y_val=y_val,
                t_max=100,
                class_names=class_names,
                metrics_service=metrics_svc,
                diversity_service=diversity_svc,
            )
            global_trees, selected_ids, ranked, conv, logs = result
            n_selected = len(global_trees)

            # Predict with selected global trees
            pred_matrix = np.empty((X_test.shape[0], n_selected), dtype=object)
            for j, tree in enumerate(global_trees):
                pred_matrix[:, j] = tree.predict(X_test)
            preds_raw = calculate_mode(pred_matrix, axis=1)
            preds = label_svc.transform(preds_raw)

        acc = accuracy_score(y_test_num, preds)
        f1 = f1_score(y_test_num, preds, average="macro")
        print(f"  {name}: trees={n_selected}, conv={conv}, acc={acc:.4f}, f1={f1:.4f}")

    # KEY ANALYSIS: How does the hybrid prediction in ResultConsolidator change results?
    print("\n--- Hybrid Prediction Analysis ---")
    from src.domain.prediction.hybrid_predictor import HybridPredictor

    # Use S2 result as example
    s2 = S2GlobalAccuracyStrategy(metrics_svc, diversity_svc)
    global_trees_s2, _, _, _, _ = s2.aggregate(
        client_trees=client_trees, client_metadata=client_meta,
        X_val=X_val, y_val=y_val, t_max=100,
        class_names=class_names
    )

    for cid in client_trees:
        local = client_trees[cid]
        external = [t for t in global_trees_s2 if not any(t is lt for lt in local)]

        predictor = HybridPredictor(
            local_weight=0.4, global_weight=0.6,
            n_classes=len(class_names), class_names=class_names,
            label_service=label_svc, use_weighted=True
        )
        hybrid_preds = predictor.predict(X_test, local, external)
        hybrid_acc = accuracy_score(y_test_num, hybrid_preds)

        # Compare with pure global
        pred_m = np.empty((X_test.shape[0], len(global_trees_s2)), dtype=object)
        for j, t in enumerate(global_trees_s2):
            pred_m[:, j] = t.predict(X_test)
        pure_global_preds = label_svc.transform(calculate_mode(pred_m, axis=1))
        pure_global_acc = accuracy_score(y_test_num, pure_global_preds)

        print(f"  {cid}: hybrid_acc={hybrid_acc:.4f}, pure_global_acc={pure_global_acc:.4f}, "
              f"local={len(local)} trees, external_global={len(external)} trees")


if __name__ == "__main__":
    test_strategy_on_wine()
