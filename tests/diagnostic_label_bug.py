"""Focused diagnostic: Progressive Selector label mismatch bug.

The main diagnostic test showed acc=0.0, f1=0.0 in ProgressiveSelector.
This investigates why.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from src.domain.services.label_service import SimpleLabelService
from src.domain.aggregation.tree_ranker import TreeRanker, TreeEntry, RankingCriterion
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector
from src.domain.prediction.voting import calculate_mode
from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService
from src.domain.model.proactive_forest import ProactiveForest


def main():
    iris = load_iris()
    X, y = iris.data, iris.target.astype(str)

    # Train to get trees
    m = ProactiveForest(n_estimators=20, alpha=0.1, class_names=["0","1","2"], random_state=42)
    m.fit(X[:100], y[:100])
    trees = m.get_trees()

    print(f"Number of trees: {len(trees)}")

    # What does a single tree predict?
    pred_raw = trees[0].predict(X[100:])
    print(f"Tree[0] raw predictions (first 10): {pred_raw[:10]}")
    print(f"Tree[0] prediction dtype: {pred_raw.dtype}")
    print(f"y_val raw (first 10): {y[100:110]}")

    # Label service transform
    label_svc = SimpleLabelService(["0","1","2"])
    y_val_norm = label_svc.transform(y[100:])
    print(f"y_val_norm (first 10): {y_val_norm[:10]}")
    print(f"y_val_norm dtype: {y_val_norm.dtype}")

    # Transform tree predictions
    pred_transformed = label_svc.transform(pred_raw)
    print(f"Tree[0] transformed preds (first 10): {pred_transformed[:10]}")

    # Direct accuracy
    direct_acc = accuracy_score(y_val_norm, pred_transformed)
    print(f"Direct accuracy tree[0]: {direct_acc:.4f}")

    # Now simulate what ProgressiveSelector._predict_ensemble does
    entries = []
    for i, tree in enumerate(trees):
        preds = tree.predict(X[100:])
        acc = accuracy_score(y[100:], preds.astype(str))
        entries.append(TreeEntry(tree=tree, client_id="c0", tree_local_id=i,
                                 accuracy=acc, macro_f1=acc, pcd=0.0))

    # Build prediction matrix like _predict_ensemble
    n_samples = X[100:].shape[0]
    n_trees = len(entries)
    all_predictions = np.empty((n_samples, n_trees), dtype=object)
    for j, entry in enumerate(entries):
        all_predictions[:, j] = entry.tree.predict(X[100:])

    print(f"\nPrediction matrix dtype: {all_predictions.dtype}")
    print(f"Prediction matrix[0, :5]: {all_predictions[0, :5]}")

    # Calculate mode
    mode_result = calculate_mode(all_predictions, axis=1)
    print(f"Mode result (first 10): {mode_result[:10]}")
    print(f"Mode result dtype: {type(mode_result[0])}")

    # NOW: what happens when we transform the mode result?
    mode_transformed = label_svc.transform(mode_result)
    print(f"Mode transformed (first 10): {mode_transformed[:10]}")

    # The key comparison inside ProgressiveSelector
    acc_after = accuracy_score(y_val_norm, mode_transformed)
    print(f"\nAccuracy y_val_norm vs mode_transformed: {acc_after:.4f}")

    # Check what metrics_svc does
    metrics_svc = SklearnMetricsService()
    acc_svc = metrics_svc.accuracy_score(y_val_norm, mode_transformed)
    print(f"SklearnMetricsService accuracy: {acc_svc:.4f}")

    # MISMATCH CHECK: are the types compatible?
    print(f"\ny_val_norm type: {type(y_val_norm[0])}, dtype: {y_val_norm.dtype}")
    print(f"mode_transformed type: {type(mode_transformed[0])}, dtype: {mode_transformed.dtype}")
    print(f"y_val_norm[0] == mode_transformed[0]? {y_val_norm[0] == mode_transformed[0]}")

    # Try without label_service transform on predictions
    acc_raw_mode = accuracy_score(y[100:].astype(str), np.array(mode_result).astype(str))
    print(f"\nRaw mode vs raw labels accuracy: {acc_raw_mode:.4f}")

    # Now run actual ProgressiveSelector
    print("\n--- Running actual ProgressiveSelector ---")
    ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
    ranked = ranker.rank(entries)

    selector = ProgressiveSelector(metrics_service=metrics_svc)
    sel_trees, sel_entries, conv, logs = selector.select(
        candidate_entries=ranked,
        X_val=X[100:], y_val_norm=y_val_norm,
        episode_size=5, t_max=100,
        convergence_threshold=0.002, label_service=label_svc
    )
    for log in logs:
        print(f"  Episode {log['episode']}: trees={log['n_trees']}, "
              f"acc={log['accuracy']:.4f}, f1={log['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
