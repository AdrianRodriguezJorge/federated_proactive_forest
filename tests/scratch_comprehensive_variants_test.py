"""Comprehensive convergence variants benchmark across all datasets and progressive strategies.

This script evaluates:
  - Strategies: S2, S3, S4, S5, S6, S7, PW
  - Datasets: Iris, Car, Sonar, Vowel, Spambase, Letter, Optdigits, Nursery
  - Configurations:
    1. Baseline (threshold=0.002, episode=5, min_episodes=1)
    2. Var A (threshold=0.0, episode=5, min_episodes=1)
    3. Var B (threshold=0.002, episode=10, min_episodes=1)
    4. Var C (threshold=0.002, episode=15, min_episodes=1)
    5. Var D (threshold=0.002, episode=5, min_episodes=4)

All source code under src/ remains unmodified. Patching is done dynamically.
"""
import sys, os
import warnings
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from joblib import Parallel, delayed
import pytest

# These are long-running scratch/benchmark scripts that monkey-patch core
# behaviour. Skip during normal unit test runs to avoid side effects.
pytest.skip("Skipping scratch benchmark tests during unit test runs", allow_module_level=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.application.orchestrators.fed_data_distributor import FedDataDistributor
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector
from src.domain.prediction.voting import calculate_mode

# Global flags for monkey patching
TEST_MIN_EPISODES = 1
TEST_MIN_ROUNDS = 1


# --- MONKEY PATCHING PROGRESSIVE SELECTOR ---
def patched_select(
    self,
    candidate_entries,
    X_val: np.ndarray,
    y_val_norm: np.ndarray,
    episode_size: int,
    t_max: int,
    convergence_threshold: float,
    min_episodes: int = 1,
    label_service=None,
    ranker=None,
):
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

        current_episode = remaining_candidates[:episode_size]
        remaining_candidates = remaining_candidates[episode_size:]

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

        # Apply convergence threshold & min_episodes restriction
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

        # Global Convergence Check with min_rounds
        round_num = round_idx + 1
        if len(self.global_trees) > 0:
            global_forest_eval = ProactiveForest.from_trees(self.global_trees, class_names=self.label_svc.classes)
            report = ForestEvaluator.evaluate(
                global_forest_eval, X_val_server, y_val_server,
                class_names=self.label_svc.classes, metrics_svc=self.metrics_svc
            )
            acc_diff = report.accuracy - global_prev_acc if global_prev_acc is not None else report.accuracy
            
            # Apply convergence check only if min rounds reached
            if global_prev_acc is not None and acc_diff <= self.convergence_threshold and round_num >= TEST_MIN_ROUNDS:
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


def evaluate_config(ds_name: str, strategy: str, config_name: str, split: DatasetSplit) -> Dict[str, Any]:
    global TEST_MIN_EPISODES, TEST_MIN_ROUNDS
    
    # Configure variables for the test
    if config_name == "Baseline":
        threshold = 0.002
        episode_size = 5
        TEST_MIN_EPISODES = 1
        TEST_MIN_ROUNDS = 1
    elif config_name == "Var A (Thresh=0.0)":
        threshold = 0.0
        episode_size = 5
        TEST_MIN_EPISODES = 1
        TEST_MIN_ROUNDS = 1
    elif config_name == "Var B (Episode=10)":
        threshold = 0.002
        episode_size = 10
        TEST_MIN_EPISODES = 1
        TEST_MIN_ROUNDS = 1
    elif config_name == "Var C (Episode=15)":
        threshold = 0.002
        episode_size = 15
        TEST_MIN_EPISODES = 1
        TEST_MIN_ROUNDS = 1
    elif config_name == "Var D (Patience=4)":
        threshold = 0.002
        episode_size = 5
        TEST_MIN_EPISODES = 4
        TEST_MIN_ROUNDS = 4
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # PW special rule: B and C don't apply to PW since its episode size is fixed to n_clients.
    if strategy == "pw" and config_name in ["Var B (Episode=10)", "Var C (Episode=15)"]:
        return {
            "strategy": strategy, "dataset": ds_name, "config": config_name,
            "accuracy": None, "f1_macro": None, "n_trees": None
        }

    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {"n_estimators": 40, "alpha": 0.1, "voting": "soft"},
        "aggregation": {
            "strategy": strategy,
            "max_rounds": 20,
            "global_convergence_threshold": threshold,
            "convergence_threshold": threshold,
            "global_episode_size": episode_size,
            "window_size": episode_size if strategy != "pw" else 5
        },
    }

    if strategy == "pw":
        orch = ProgressiveTreeOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)
        
    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)
        acc = res.hybrid_accuracy_mean
        f1 = res.hybrid_f1_mean
        # Get ensemble size
        n_trees = len(orch.flex_pool._models["server"].get("trees", []))
    except Exception as e:
        print(f"Error running {strategy} with {config_name} on {ds_name}: {e}")
        acc, f1, n_trees = 0.0, 0.0, 0
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()
            
    return {
        "strategy": strategy, "dataset": ds_name, "config": config_name,
        "accuracy": acc, "f1_macro": f1, "n_trees": n_trees
    }


def main():
    datasets = ["Iris", "Car", "Sonar", "Vowel", "Spambase", "Letter", "Optdigits", "Nursery"]
    strategies = ["s2_global_accuracy", "s3_global_f1", "s4_global_f1_pcd", "s5_perclient_accuracy", "s6_perclient_f1", "s7_perclient_f1_pcd", "pw"]
    configs = ["Baseline", "Var A (Thresh=0.0)", "Var B (Episode=10)", "Var C (Episode=15)", "Var D (Patience=4)"]

    print("======================================================================")
    print("RUNNING COMPREHENSIVE CONVERGENCE VARIANTS BENCHMARK")
    print("======================================================================")

    # Pre-load all datasets to memory
    splits = {}
    for d in datasets:
        print(f"Loading {d} dataset...")
        splits[d] = load_dataset(d)

    print("\nStarting evaluation of all combinations in parallel...")
    
    # We can parallelize the outer loops to speed up execution
    tasks = []
    for d in datasets:
        for strat in strategies:
            for cfg in configs:
                tasks.append((d, strat, cfg))

    # Run in parallel using joblib
    results = Parallel(n_jobs=2, verbose=10)(
        delayed(evaluate_config)(task[0], task[1], task[2], splits[task[0]])
        for task in tasks
    )

    # Organise results
    records = []
    for r in results:
        # Ignore skipped configurations for PW
        if r["accuracy"] is None:
            continue
        records.append(r)

    df_res = pd.DataFrame(records)
    df_res.to_csv("results/convergence_variants_comprehensive.csv", index=False)

    print("\n======================================================================")
    print("COMPREHENSIVE RESULTS SUMMARY (Averaged across all 8 datasets)")
    print("======================================================================")

    for strat in strategies:
        print(f"\nStrategy: {strat.upper()}")
        df_strat = df_res[df_res["strategy"] == strat]
        for cfg in configs:
            df_cfg = df_strat[df_strat["config"] == cfg]
            if df_cfg.empty:
                continue
            avg_acc = df_cfg["accuracy"].mean()
            avg_f1 = df_cfg["f1_macro"].mean()
            avg_trees = df_cfg["n_trees"].mean()
            print(f"  {cfg:22} -> Avg Acc: {avg_acc:.4f} | Avg F1: {avg_f1:.4f} | Avg Selected Trees: {avg_trees:.1f}")

    print("\nGlobal comparison across ALL strategies combined:")
    for cfg in configs:
        df_cfg = df_res[df_res["config"] == cfg]
        avg_acc = df_cfg["accuracy"].mean()
        avg_f1 = df_cfg["f1_macro"].mean()
        avg_trees = df_cfg["n_trees"].mean()
        print(f"  {cfg:22} -> Avg Acc: {avg_acc:.4f} | Avg F1: {avg_f1:.4f} | Avg Selected Trees: {avg_trees:.1f}")


if __name__ == "__main__":
    main()
