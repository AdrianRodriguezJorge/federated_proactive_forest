"""Rigorously test convergence variations on progressive selection without modifying src/."""
import sys, os
from typing import List
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.domain.services.label_service import SimpleLabelService
from src.domain.aggregation.tree_ranker import TreeRanker, TreeEntry, RankingCriterion
from src.domain.prediction.voting import calculate_mode
from src.domain.model.proactive_forest import ProactiveForest


class CustomProgressiveSelector:
    """A local copy of ProgressiveSelector to test modifications easily."""
    def __init__(self, convergence_threshold: float, episode_size: int, min_episodes: int = 1):
        self.convergence_threshold = convergence_threshold
        self.episode_size = episode_size
        self.min_episodes = min_episodes

    def select(
        self,
        candidate_entries: List[TreeEntry],
        X_val: np.ndarray,
        y_val_norm: np.ndarray,
        t_max: int,
        label_service: SimpleLabelService,
    ):
        stop_counter = 0
        selected_entries: List[TreeEntry] = []
        episode_accuracies = []
        episode_idx = 0

        remaining_candidates = candidate_entries.copy()
        prediction_cache = {}

        while len(selected_entries) < t_max and remaining_candidates:
            episode_idx += 1

            # Pick next episode
            current_episode = remaining_candidates[:self.episode_size]
            remaining_candidates = remaining_candidates[self.episode_size:]

            if not current_episode:
                break

            for entry in current_episode:
                selected_entries.append(entry)
                tree_id = id(entry.tree)
                if tree_id not in prediction_cache:
                    prediction_cache[tree_id] = entry.tree.predict(X_val)

            # Evaluate current ensemble
            raw_predictions = self._predict_ensemble(selected_entries, X_val, cache=prediction_cache)
            predictions = label_service.transform(raw_predictions)
            acc = float(np.mean(predictions == y_val_norm))
            episode_accuracies.append(acc)

            # Check convergence
            if len(episode_accuracies) >= 2 and episode_idx >= self.min_episodes:
                improvement = episode_accuracies[-1] - episode_accuracies[-2]
                if improvement < self.convergence_threshold:
                    stop_counter += 1
                    if stop_counter >= 2:
                        break
                else:
                    stop_counter = 0

        selected_trees = [e.tree for e in selected_entries]
        return selected_trees, episode_idx, episode_accuracies

    def _predict_ensemble(self, selected_entries, X, cache):
        n_samples = X.shape[0]
        n_trees = len(selected_entries)
        all_predictions = np.empty((n_samples, n_trees), dtype=object)
        for j, entry in enumerate(selected_entries):
            tree_id = id(entry.tree)
            if tree_id in cache:
                all_predictions[:, j] = cache[tree_id]
            else:
                preds = entry.tree.predict(X)
                all_predictions[:, j] = preds
                cache[tree_id] = preds
        return calculate_mode(all_predictions, axis=1)


def evaluate_ensemble(trees, X_test, y_test_num, label_svc):
    if not trees:
        return 0.0, 0.0
    n_samples = X_test.shape[0]
    pred_matrix = np.empty((n_samples, len(trees)), dtype=object)
    for j, tree in enumerate(trees):
        pred_matrix[:, j] = tree.predict(X_test)
    preds_raw = calculate_mode(pred_matrix, axis=1)
    preds = label_svc.transform(preds_raw)
    acc = accuracy_score(y_test_num, preds)
    f1 = f1_score(y_test_num, preds, average="macro", zero_division=0)
    return acc, f1


def run_experiment_on_dataset(dataset_name, X, y, class_names):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # Train client models
    n = len(X_tr) // 3
    client_trees = {}
    client_meta = {}
    for i in range(3):
        start, end = i*n, (i+1)*n if i < 2 else len(X_tr)
        m = ProactiveForest(n_estimators=40, alpha=0.1, class_names=class_names, random_state=i*10)
        m.fit(X_tr[start:end], y_tr[start:end])
        cid = f"client_{i}"
        client_trees[cid] = m.get_trees()
        client_meta[cid] = {}

    label_svc = SimpleLabelService(class_names)
    y_val_num = label_svc.transform(y_val)
    y_test_num = label_svc.transform(y_test)

    # Build and rank entries using S2 (global accuracy)
    all_entries = TreeRanker.build_entries(client_trees, client_meta, X_val, y_val)
    ranker = TreeRanker(criterion=RankingCriterion.ACCURACY)
    ranked = ranker.rank(all_entries)

    # Define Configurations
    configs = {
        "Baseline": {"threshold": 0.002, "episode_size": 5, "min_episodes": 1},
        "Var A (Threshold=0.0)": {"threshold": 0.0, "episode_size": 5, "min_episodes": 1},
        "Var B (Episode=15)": {"threshold": 0.002, "episode_size": 15, "min_episodes": 1},
        "Var C (Min Episode=4)": {"threshold": 0.002, "episode_size": 5, "min_episodes": 4},
    }

    print(f"\n==================================================")
    print(f"DATASET: {dataset_name} ({len(ranked)} candidate trees total)")
    print(f"==================================================")

    for name, cfg in configs.items():
        selector = CustomProgressiveSelector(
            convergence_threshold=cfg["threshold"],
            episode_size=cfg["episode_size"],
            min_episodes=cfg["min_episodes"]
        )
        selected_trees, episodes, acc_history = selector.select(
            candidate_entries=ranked,
            X_val=X_val,
            y_val_norm=y_val_num,
            t_max=100,
            label_service=label_svc
        )

        test_acc, test_f1 = evaluate_ensemble(selected_trees, X_test, y_test_num, label_svc)
        print(f"  {name:22} -> Episodes: {episodes:2} | Selected Trees: {len(selected_trees):3} | "
              f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    # Test on Wine
    wine = load_wine()
    run_experiment_on_dataset("Wine", wine.data, wine.target.astype(str), [str(c) for c in np.unique(wine.target)])

    # Test on Digits (larger, more realistic)
    digits = load_digits()
    run_experiment_on_dataset("Digits", digits.data, digits.target.astype(str), [str(c) for c in np.unique(digits.target)])
