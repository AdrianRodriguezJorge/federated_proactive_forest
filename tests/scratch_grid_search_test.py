"""Grid search test for convergence parameters on a subset of datasets and strategies.

Varies:
  - min_episodes: [4, 5]
  - global_episode_size: [5, 10, 15]
Strategies:
  - S4 (Global F1 + PCD)
  - S6 (Per-Client F1)
  - PW (Progressive Windows)
Datasets:
  - Car, Sonar, Vowel, Spambase
"""
import sys, os
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector

# Global override variables for grid search
TEST_EPISODE_SIZE = 5
TEST_MIN_EPISODES = 4


# --- MONKEY PATCHING SELECTOR TO OVERRIDE EPISODE SIZE AND PATIENCE ---
def patched_select(
    self,
    candidate_entries,
    X_val: np.ndarray,
    y_val_norm: np.ndarray,
    episode_size: int,
    t_max: int,
    convergence_threshold: float,
    label_service=None,
    ranker=None,
):
    # Override episode_size dynamically with the grid search parameter
    actual_episode_size = TEST_EPISODE_SIZE
    
    stop_counter = 0
    selected_entries = []
    episode_accuracies = []
    round_logs = []
    convergence_round = None
    prediction_cache = {}

    n_val_samples = X_val.shape[0]
    global_hits_per_sample = np.zeros(n_val_samples, dtype=int)
    remaining_candidates = candidate_entries.copy()
    episode_idx = 0

    while len(selected_entries) < t_max and remaining_candidates:
        episode_idx += 1

        # Re-ranking if needed
        if ranker and self.diversity_svc and ranker.criterion.value == "f1_pcd":
            for entry in remaining_candidates:
                tree_id = id(entry.tree)
                if tree_id not in prediction_cache:
                    prediction_cache[tree_id] = entry.tree.predict(X_val)
                preds = prediction_cache[tree_id]
                entry.pcd = self.diversity_svc.calculate_marginal_pcd(
                    candidate_predictions=preds,
                    current_hits_per_sample=global_hits_per_sample,
                    n_existing_trees=len(selected_entries),
                    y_true=y_val_norm,
                )
            remaining_candidates = ranker.rank(remaining_candidates)

        current_episode = remaining_candidates[:actual_episode_size]
        remaining_candidates = remaining_candidates[actual_episode_size:]

        if not current_episode:
            break

        for entry in current_episode:
            selected_entries.append(entry)
            tree_id = id(entry.tree)
            if tree_id not in prediction_cache:
                prediction_cache[tree_id] = entry.tree.predict(X_val)
            preds_raw = prediction_cache[tree_id]
            if label_service:
                preds = label_service.transform(preds_raw)
            else:
                preds = preds_raw
            global_hits_per_sample += (preds == y_val_norm).astype(int)

        raw_predictions = self._predict_ensemble(selected_entries, X_val, cache=prediction_cache)
        if label_service:
            predictions = label_service.transform(raw_predictions)
        else:
            predictions = raw_predictions

        if self.metrics_svc:
            acc = float(self.metrics_svc.accuracy_score(y_val_norm, predictions))
            f1 = float(self.metrics_svc.f1_score(y_val_norm, predictions, average="macro"))
        else:
            acc = float(np.mean(predictions == y_val_norm))
            f1 = 0.0

        episode_accuracies.append(acc)
        round_logs.append({
            "episode": episode_idx,
            "n_trees": len(selected_entries),
            "accuracy": acc,
            "macro_f1": f1
        })

        # Apply convergence check only if minimum episodes limit is met
        if len(episode_accuracies) >= 2 and episode_idx >= TEST_MIN_EPISODES:
            improvement = episode_accuracies[-1] - episode_accuracies[-2]
            if improvement < convergence_threshold:
                stop_counter += 1
                if stop_counter >= 2:
                    convergence_round = episode_idx
                    break
            else:
                stop_counter = 0

    selected_trees = [e.tree for e in selected_entries]
    return selected_trees, selected_entries, convergence_round, round_logs


ProgressiveSelector.select = patched_select


# --- MONKEY PATCHING PW ORCHESTRATOR ---
from src.infrastructure.flex.flex_deploy_model_pf import deploy_server_config_pf
from src.infrastructure.flex.flex_pw_progressive import (
    aggregate_trees_pw, collect_new_trees_pw, deploy_global_forest_pw,
    set_client_windows_pw, train_window_pf_pw
)
from src.infrastructure.flex.flex_evaluate_pf import (
    evaluate_global_pf_model, evaluate_global_pf_model_at_clients
)
from flex.data.dataset import Dataset
from src.application.orchestrators.result_consolidator import ResultConsolidator
from src.application.orchestrators.fl_results import FLResults
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.metrics.forest_evaluator import ForestEvaluator

def patched_pw_run(self, n_bootstrap: int = 0) -> FLResults:
    if not self.federated_data or self.dataset_split is None:
        raise ValueError("Federation not setup.")

    self.flex_pool.servers.map(deploy_server_config_pf, self.flex_pool.clients)
    active_client_ids = list(self.flex_pool.clients.actor_ids)
    server_id = "server"
    X_val_server = self.dataset_split.X_val
    y_val_server = self.dataset_split.y_val
    y_val_encoded = self.label_svc.transform(y_val_server)
    global_correct_counts = np.zeros(X_val_server.shape[0], dtype=int)
    self.flex_pool._models[server_id]["global_trees"] = []
    global_stop_counter = 0
    global_prev_acc = None

    for round_idx in range(self.max_rounds):
        if not active_client_ids:
            break
        self.flex_pool.clients.map(train_window_pf_pw, active_ids=active_client_ids)

        newly_converged = []
        for cid in active_client_ids:
            client_model = self.flex_pool._models.get(str(cid)) or self.flex_pool._models.get(cid)
            if client_model is not None:
                meta = client_model.get("metadata", {})
                has_conv = meta.get("has_converged", False) if isinstance(meta, dict) else getattr(meta, "has_converged", False)
                if has_conv:
                    newly_converged.append(cid)

        self.flex_pool.aggregators.map(collect_new_trees_pw, self.flex_pool.clients)
        self.flex_pool.aggregators.map(aggregate_trees_pw)
        self.flex_pool.aggregators.map(set_client_windows_pw, self.flex_pool.servers)

        client_windows = self.flex_pool._models[server_id].get("client_windows", {})

        for cid in active_client_ids:
            cid_str = str(cid)
            window_data = client_windows.get(cid_str, {})
            w_trees = window_data.get("trees", [])
            w_metrics = window_data.get("metrics", [])
            if not w_trees:
                continue

            best_tree = None
            best_score = -1e9
            for idx, tree in enumerate(w_trees):
                tree_f1 = w_metrics[idx].get("macro_f1", 0.0) if idx < len(w_metrics) else 0.0
                if len(self.global_trees) > 0:
                    cand_preds_raw = tree.predict(X_val_server)
                    cand_preds_norm = self.label_svc.transform(cand_preds_raw) if self.label_svc else cand_preds_raw
                    diversity = self.diversity_svc.calculate_marginal_pcd(
                        candidate_predictions=cand_preds_norm,
                        current_hits_per_sample=global_correct_counts,
                        n_existing_trees=len(self.global_trees),
                        y_true=y_val_encoded,
                    )
                else:
                    diversity = 1.0
                effective_f1_weight = self.f1_weight if len(self.global_trees) > 0 else 1.0
                score = (effective_f1_weight * tree_f1) + ((1.0 - effective_f1_weight) * diversity)
                if score > best_score:
                    best_score = score
                    best_tree = tree

            if best_tree is not None:
                self.global_trees.append(best_tree)
                preds_raw = best_tree.predict(X_val_server)
                preds_int = self.label_svc.transform(preds_raw)
                global_correct_counts += (preds_int == y_val_encoded).astype(int)

        self.flex_pool._models[server_id]["global_trees"] = self.global_trees
        self.flex_pool._models[server_id]["trees"] = self.global_trees
        self.flex_pool.servers.map(deploy_global_forest_pw, self.flex_pool.clients)

        # Global Convergence Check with min_rounds (TEST_MIN_EPISODES)
        round_num = round_idx + 1
        if len(self.global_trees) > 0:
            global_forest_eval = ProactiveForest.from_trees(self.global_trees, class_names=self.label_svc.classes)
            report = ForestEvaluator.evaluate(
                global_forest_eval, X_val_server, y_val_server,
                class_names=self.label_svc.classes, metrics_svc=self.metrics_svc
            )
            acc_diff = report.accuracy - global_prev_acc if global_prev_acc is not None else report.accuracy
            
            # Apply convergence check only if min rounds met
            if global_prev_acc is not None and acc_diff <= self.convergence_threshold and round_num >= TEST_MIN_EPISODES:
                global_stop_counter += 1
                if global_stop_counter >= 2:
                    break
            else:
                global_stop_counter = 0
            global_prev_acc = report.accuracy

        for cid in newly_converged:
            active_client_ids.remove(cid)

    global_forest = ProactiveForest.from_trees(self.global_trees, class_names=self.label_svc.classes)
    self.flex_pool._models[server_id]["model"] = global_forest
    server_val_dataset = Dataset.from_array(X_val_server, y_val_server)
    server_eval = self.flex_pool.servers.map(evaluate_global_pf_model, test_data=server_val_dataset)
    self.flex_pool.clients.map(evaluate_global_pf_model_at_clients)
    consolidator = ResultConsolidator(self.label_svc, self.config)
    return consolidator.consolidate(
        strategy_name="PW", flex_pool=self.flex_pool, dataset_split=self.dataset_split,
        server_eval=server_eval, n_bootstrap=n_bootstrap
    )


ProgressiveTreeOrchestrator.run_federated_round = patched_pw_run


def load_dataset(ds_name: str) -> DatasetSplit:
    preset = DATASET_METADATA[ds_name.lower()]
    df = pd.read_csv(preset["file_path"], sep=preset["sep"])
    cols_to_drop = preset.get("columns_to_drop", [])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    target = preset["target_column"]
    X_raw = df.drop(columns=[target])
    y_raw = df[target].astype(str).values
    
    cat_cols = list(preset.get("categorical_features", []))
    for col in X_raw.columns:
        is_obj = X_raw[col].dtype == "object"
        is_str = is_string_dtype(X_raw[col])
        if (is_obj or is_str) and col not in cat_cols:
            cat_cols.append(col)
            
    le = LabelEncoder()
    y_raw_encoded = le.fit_transform(y_raw)
    
    # 60/20/20 train/val/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw_encoded, test_size=0.20, random_state=42, stratify=y_raw_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    # Preprocessing
    X_tr = X_train.values.copy()
    X_va = X_val.values.copy()
    X_te = X_test.values.copy()
    
    cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    num_cols_idx = [X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols]
    
    if cat_cols_idx:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[:, cat_cols_idx] = enc.fit_transform(X_tr[:, cat_cols_idx].astype(str))
        X_va[:, cat_cols_idx] = enc.transform(X_va[:, cat_cols_idx].astype(str))
        X_te[:, cat_cols_idx] = enc.transform(X_te[:, cat_cols_idx].astype(str))
        
    if num_cols_idx:
        scaler = StandardScaler()
        X_tr[:, num_cols_idx] = scaler.fit_transform(X_tr[:, num_cols_idx])
        X_va[:, num_cols_idx] = scaler.transform(X_va[:, num_cols_idx])
        X_te[:, num_cols_idx] = scaler.transform(X_te[:, num_cols_idx])
        
    return DatasetSplit(
        X_train=X_tr.astype(np.float64),
        X_val=X_va.astype(np.float64),
        X_test=X_te.astype(np.float64),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(X_raw.columns),
        class_names=[str(c) for c in le.classes_],
        dataset_name=ds_name
    )


def evaluate_grid_combination(
    ds_name: str, strategy: str, min_episodes: int, episode_size: int, split: DatasetSplit
) -> Dict[str, Any]:
    global TEST_EPISODE_SIZE, TEST_MIN_EPISODES
    
    TEST_EPISODE_SIZE = episode_size
    TEST_MIN_EPISODES = min_episodes

    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {"n_estimators": 40, "alpha": 0.1, "voting": "soft"},
        "aggregation": {
            "strategy": strategy,
            "max_rounds": 20,
            "global_convergence_threshold": 0.002,
            "convergence_threshold": 0.002,
            "global_episode_size": episode_size,
            "window_size": episode_size if strategy != "pw" else 5
        },
    }

    if strategy == "pw":
        # For PW, window_size dictates trees trained per client round
        config["aggregation"]["window_size"] = episode_size
        orch = ProgressiveTreeOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)
        
    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)
        acc = res.hybrid_accuracy_mean
        f1 = res.hybrid_f1_mean
        n_trees = len(orch.flex_pool._models["server"].get("trees", []))
    except Exception as e:
        print(f"Error running {strategy} (min_ep={min_episodes}, ep_size={episode_size}) on {ds_name}: {e}")
        acc, f1, n_trees = 0.0, 0.0, 0
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()
            
    return {
        "strategy": strategy,
        "dataset": ds_name,
        "min_episodes": min_episodes,
        "episode_size": episode_size,
        "accuracy": acc,
        "f1_macro": f1,
        "n_trees": n_trees
    }


def main():
    datasets = ["Car", "Sonar", "Vowel", "Spambase"]
    strategies = ["s4_global_f1_pcd", "s6_perclient_f1", "pw"]
    
    # Grid parameters
    min_episodes_list = [4, 5]
    episode_sizes = [5, 10, 15]

    print("======================================================================")
    print("CONVERGENCE PARAMETERS GRID SEARCH BENCHMARK")
    print("======================================================================")

    splits = {}
    for d in datasets:
        print(f"Loading {d} dataset...")
        splits[d] = load_dataset(d)

    tasks = []
    for d in datasets:
        for strat in strategies:
            for minep in min_episodes_list:
                for epsize in episode_sizes:
                    tasks.append((d, strat, minep, epsize))

    print(f"\nRunning {len(tasks)} grid tasks in parallel (n_jobs=2)...")
    results = Parallel(n_jobs=2, verbose=10)(
        delayed(evaluate_grid_combination)(task[0], task[1], task[2], task[3], splits[task[0]])
        for task in tasks
    )

    df_grid = pd.DataFrame(results)
    df_grid.to_csv("results/grid_search_convergence.csv", index=False)

    print("\n======================================================================")
    print("GRID SEARCH RESULTS SUMMARY (Averaged across 4 datasets)")
    print("======================================================================")

    for strat in strategies:
        print(f"\nStrategy: {strat.upper()}")
        print("-" * 70)
        df_strat = df_grid[df_grid["strategy"] == strat]
        for minep in min_episodes_list:
            for epsize in episode_sizes:
                df_comb = df_strat[(df_strat["min_episodes"] == minep) & (df_strat["episode_size"] == epsize)]
                avg_acc = df_comb["accuracy"].mean()
                avg_f1 = df_comb["f1_macro"].mean()
                avg_trees = df_comb["n_trees"].mean()
                print(f"  Patience (Min Ep/Rounds): {minep} | Episode/Window Size: {epsize:2} -> "
                      f"Acc: {avg_acc:.4f} | F1: {avg_f1:.4f} | Trees: {avg_trees:.1f}")


if __name__ == "__main__":
    main()
