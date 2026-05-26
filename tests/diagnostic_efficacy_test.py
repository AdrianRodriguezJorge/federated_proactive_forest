"""Diagnostic Efficacy Tests for Federated Proactive Forest.

Tests designed to identify bugs, bad practices, and methodological issues
that reduce strategy efficacy.
"""
import sys, os, warnings
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.services.label_service import SimpleLabelService
from src.domain.aggregation.tree_ranker import TreeRanker, TreeEntry, RankingCriterion
from src.domain.aggregation.services.tree_metric_extractor import TreeMetricExtractor
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector
from src.domain.prediction.voting import calculate_mode
from src.domain.update.roulette_updater import RouletteUpdater
from src.domain.model.hybrid_forest import HybridForest
from src.infrastructure.metrics.diversity_service import PredictionBasedDiversityService
from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService


def make_split(X, y, class_names):
    """Helper to create a DatasetSplit from arrays."""
    from src.domain.dataset.base_adapter import DatasetSplit
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    return DatasetSplit(
        X_train=X_tr, X_val=X_val, X_test=X_test,
        y_train=y_tr, y_val=y_val, y_test=y_test,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        class_names=class_names,
    )


# ============================================================================
# TEST 1: TreeMetricExtractor uses CLIENT-LEVEL PCD, not per-tree PCD
# ============================================================================
def test_tree_metric_extractor_pcd_is_client_level():
    """CRITICAL: TreeMetricExtractor assigns the SAME PCD to ALL trees from
    the same client because it computes PCD over the entire client's prediction
    matrix. This means ranking by PCD within a client is USELESS - all trees
    from the same client get identical PCD scores."""
    print("\n" + "="*70)
    print("TEST 1: TreeMetricExtractor PCD granularity")
    print("="*70)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(str)
    from src.domain.model.proactive_forest import ProactiveForest

    model = ProactiveForest(n_estimators=20, alpha_pf=0.1, class_names=["0","1","2"])
    model.fit(X[:100], y[:100])
    trees = model.get_trees()

    extractor = TreeMetricExtractor()
    entries = extractor.extract_metrics(
        client_trees={"c1": trees},
        client_metadata={"c1": {}},
        X_val=X[100:], y_val=y[100:]
    )

    pcds = [e["pcd"] for e in entries]
    unique_pcds = len(set(pcds))

    print(f"  Trees from client c1: {len(entries)}")
    print(f"  Unique PCD values: {unique_pcds}")
    print(f"  PCD values (first 5): {pcds[:5]}")

    if unique_pcds == 1:
        print("  [ISSUE CONFIRMED] All trees from same client get IDENTICAL PCD.")
        print("  -> S4 and S7 (F1+PCD) ranking is partially broken for intra-client ranking.")
        print("  -> PCD component adds NO discriminative power within a client.")
        return False
    else:
        print("  [OK] PCD varies per tree.")
        return True


# ============================================================================
# TEST 2: S8 Roulette variants produce IDENTICAL results
# ============================================================================
def test_s8_variants_identical():
    """Check if all S8 variants produce identical results (as seen in benchmark)."""
    print("\n" + "="*70)
    print("TEST 2: S8 Roulette variant differentiation")
    print("="*70)

    from src.domain.aggregation.strategies.s8_roulette_strategy import (
        S8WeightedAverageStrategy, S8SimpleMeanStrategy, S8MedianStrategy,
        S8ConsensusStrategy, S8ProactivePCDStrategy
    )

    np.random.seed(42)
    n_features = 10
    # Simulate 3 clients with DIFFERENT roulettes
    v1 = np.random.dirichlet(np.ones(n_features))
    v2 = np.random.dirichlet(np.ones(n_features) * 0.5)
    v3 = np.random.dirichlet(np.ones(n_features) * 2.0)

    client_vectors = {"c1": v1, "c2": v2, "c3": v3}
    sizes = {"c1": 100, "c2": 50, "c3": 200}
    f1s = {"c1": 0.9, "c2": 0.6, "c3": 0.75}
    pcds = {"c1": 0.3, "c2": 0.5, "c3": 0.2}

    results = {}
    results["WEIGHTED"] = S8WeightedAverageStrategy().aggregate_vectors(client_vectors, sizes)
    results["MEAN"] = S8SimpleMeanStrategy().aggregate_vectors(client_vectors)
    results["MEDIAN"] = S8MedianStrategy().aggregate_vectors(client_vectors)
    results["CONSENSUS"] = S8ConsensusStrategy().aggregate_vectors(client_vectors, client_f1_scores=f1s)
    results["PCD"] = S8ProactivePCDStrategy().aggregate_vectors(client_vectors, client_pcd_scores=pcds)

    print("  Checking if variants produce different global roulettes...")
    all_identical = True
    for name1, r1 in results.items():
        for name2, r2 in results.items():
            if name1 < name2:
                diff = np.max(np.abs(r1 - r2))
                if diff > 1e-10:
                    all_identical = False
                    print(f"  {name1} vs {name2}: max_diff={diff:.6f} [DIFFERENT]")
                else:
                    print(f"  {name1} vs {name2}: max_diff={diff:.10f} [IDENTICAL]")

    if all_identical:
        print("  [ISSUE] All variants identical - aggregation math works but")
        print("  real-world clients may produce near-identical roulettes.")
    else:
        print("  [OK] Variants produce different outputs with synthetic data.")
        print("  -> Real issue: clients may converge to same roulette due to local_roulette_weight=0")

    # Test: with local_roulette_weight=0, client FULLY adopts global roulette each round
    updater = RouletteUpdater(local_roulette_weight=0.0)
    local = np.random.dirichlet(np.ones(n_features))
    glob = np.random.dirichlet(np.ones(n_features))
    fused = updater.fuse(local, glob)
    diff = np.max(np.abs(fused - glob))
    print(f"\n  local_roulette_weight=0 fusion test: local is COMPLETELY replaced by global")
    print(f"  max|fused - global| = {diff:.15f}")
    if diff < 1e-10:
        print("  [CRITICAL ISSUE] local_roulette_weight=0 means ALL clients adopt the EXACT same")
        print("  global roulette after round 1. All variants collapse to MEAN.")
        print("  -> This is WHY all S8 variants produce identical benchmark results!")
        return False
    return True


# ============================================================================
# TEST 3: HybridForest weight balance
# ============================================================================
def test_hybrid_forest_weights():
    """Test if 0.4/0.6 local/global weights hurt performance."""
    print("\n" + "="*70)
    print("TEST 3: HybridForest weight sensitivity")
    print("="*70)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    from src.domain.model.proactive_forest import ProactiveForest
    m1 = ProactiveForest(n_estimators=30, alpha_pf=0.1, class_names=["0","1","2"])
    m1.fit(X_tr[:50], y_tr[:50].astype(str))
    local_trees = m1.get_trees()

    m2 = ProactiveForest(n_estimators=30, alpha_pf=0.1, class_names=["0","1","2"])
    m2.fit(X_tr[50:], y_tr[50:].astype(str))
    global_trees = m2.get_trees()

    label_svc = SimpleLabelService(["0","1","2"])
    configs = [(0.4, 0.6), (0.5, 0.5), (0.3, 0.7), (0.6, 0.4), (0.2, 0.8)]
    best_f1 = -1
    best_cfg = None

    for lw, gw in configs:
        hf = HybridForest(local_trees, global_trees, lw, gw, 3, ["0","1","2"], label_svc)
        preds = hf.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"  Weights local={lw}, global={gw}: F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = (lw, gw)

    print(f"  Best config: local={best_cfg[0]}, global={best_cfg[1]} (F1={best_f1:.4f})")
    if best_cfg != (0.4, 0.6):
        print("  [FINDING] Default 0.4/0.6 is NOT optimal for this test case.")
    return True


# ============================================================================
# TEST 4: Progressive Selector convergence sensitivity
# ============================================================================
def test_progressive_convergence_threshold():
    """Test if convergence threshold 0.002 is too aggressive."""
    print("\n" + "="*70)
    print("TEST 4: Progressive Selector convergence threshold sensitivity")
    print("="*70)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(str)
    from src.domain.model.proactive_forest import ProactiveForest

    # Train models to get trees
    all_trees = []
    for seed in range(3):
        m = ProactiveForest(n_estimators=40, alpha_pf=0.1, class_names=["0","1","2"], random_state=seed*10)
        m.fit(X[:100], y[:100])
        all_trees.extend(m.get_trees())

    # Build entries
    entries = []
    for i, tree in enumerate(all_trees):
        preds = tree.predict(X[100:])
        acc = accuracy_score(y[100:], preds.astype(str))
        f1 = f1_score(y[100:], preds.astype(str), average="macro", zero_division=0)
        entries.append(TreeEntry(tree=tree, client_id="c0", tree_local_id=i,
                                 accuracy=acc, macro_f1=f1, pcd=0.0))

    ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
    ranked = ranker.rank(entries)

    metrics_svc = SklearnMetricsService()
    label_svc = SimpleLabelService(["0","1","2"])
    y_val_norm = label_svc.transform(y[100:])

    thresholds = [0.002, 0.005, 0.001, 0.0005, 0.0]
    for thresh in thresholds:
        selector = ProgressiveSelector(metrics_service=metrics_svc)
        trees, selected, conv_round, logs = selector.select(
            candidate_entries=[TreeEntry(tree=e.tree, client_id=e.client_id,
                              tree_local_id=e.tree_local_id, accuracy=e.accuracy,
                              macro_f1=e.macro_f1, pcd=e.pcd) for e in ranked],
            X_val=X[100:], y_val_norm=y_val_norm,
            episode_size=5, max_trees=100,
            convergence_threshold=thresh, label_service=label_svc
        )
        final_acc = logs[-1]["accuracy"] if logs else 0
        final_f1 = logs[-1]["macro_f1"] if logs else 0
        print(f"  threshold={thresh:.4f}: selected={len(trees)} trees, "
              f"conv_round={conv_round}, acc={final_acc:.4f}, f1={final_f1:.4f}")

    print("  [ANALYSIS] If 0.002 selects far fewer trees than 0.0, it may be")
    print("  too aggressive and stopping before optimal ensemble size.")
    return True


# ============================================================================
# TEST 5: Label encoding round-trip integrity
# ============================================================================
def test_label_encoding_roundtrip():
    """Verify label encoding doesn't corrupt predictions."""
    print("\n" + "="*70)
    print("TEST 5: Label encoding round-trip integrity")
    print("="*70)

    test_cases = [
        (["setosa", "versicolor", "virginica"], ["versicolor", "setosa", "virginica"]),
        (["0", "1", "2"], [0, 1, 2]),
        (["a", "b", "c", "d"], ["b", "a", "d", "c"]),
    ]

    all_pass = True
    for classes, inputs in test_cases:
        svc = SimpleLabelService(classes)
        encoded = svc.transform(inputs)
        decoded = svc.inverse_transform(encoded)
        match = all(str(d) == str(i) for d, i in zip(decoded, [str(x) for x in inputs]))

        if not match:
            print(f"  [FAIL] classes={classes}, input={inputs}")
            print(f"         encoded={encoded}, decoded={decoded}")
            all_pass = False
        else:
            print(f"  [OK] classes={classes}: roundtrip correct")

    # Test integer passthrough (critical for benchmark)
    svc = SimpleLabelService(["0", "1", "2"])
    int_input = np.array([0, 1, 2, 1, 0])
    encoded = svc.transform(int_input)
    if not np.array_equal(int_input, encoded):
        print(f"  [ISSUE] Integer passthrough modifies values: {int_input} -> {encoded}")
        all_pass = False
    else:
        print(f"  [OK] Integer passthrough preserved")

    return all_pass


# ============================================================================
# TEST 6: Voting mode correctness
# ============================================================================
def test_voting_mode():
    """Verify majority voting handles edge cases."""
    print("\n" + "="*70)
    print("TEST 6: Voting mode correctness")
    print("="*70)

    # Integer path
    preds_int = np.array([[0,0,1],[1,1,0],[2,2,2],[0,1,2]])
    result_int = calculate_mode(preds_int, axis=1)
    expected_int = np.array([0, 1, 2, 0])  # tie-break: lowest index
    print(f"  Integer mode: {result_int} (expected: {expected_int})")

    # String path
    preds_str = np.array([["a","a","b"],["b","b","a"],["c","c","c"]])
    result_str = calculate_mode(preds_str, axis=1)
    expected_str = np.array(["a", "b", "c"])
    print(f"  String mode: {result_str} (expected: {expected_str})")

    ok = np.array_equal(result_int, expected_int) and np.array_equal(result_str, expected_str)
    if ok:
        print("  [OK] Voting mode correct")
    else:
        print("  [ISSUE] Voting mode has errors")
    return ok


# ============================================================================
# TEST 7: S2-S4 vs S5-S7 structural difference analysis
# ============================================================================
def test_global_vs_perclient_tree_selection():
    """Test whether global vs per-client ranking produces meaningfully
    different ensembles."""
    print("\n" + "="*70)
    print("TEST 7: Global vs Per-client tree selection structural analysis")
    print("="*70)

    iris = load_iris()
    X, y = iris.data, iris.target.astype(str)
    from src.domain.model.proactive_forest import ProactiveForest

    client_trees = {}
    client_meta = {}
    for i in range(3):
        start, end = i*33, (i+1)*33+1
        m = ProactiveForest(n_estimators=30, alpha_pf=0.1, class_names=["0","1","2"], random_state=i*10)
        m.fit(X[start:end], y[start:end])
        cid = f"c{i}"
        client_trees[cid] = m.get_trees()
        client_meta[cid] = {}

    # Global ranking (S3-like)
    all_entries = TreeRanker.build_entries(client_trees, client_meta, X[100:], y[100:])
    ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
    global_ranked = ranker.rank(all_entries)

    # Count client representation in top-20
    top20 = global_ranked[:20]
    client_counts = {}
    for e in top20:
        client_counts[e.client_id] = client_counts.get(e.client_id, 0) + 1

    print(f"  Global ranking top-20 client distribution: {client_counts}")
    total_clients = len(client_trees)
    max_rep = max(client_counts.values()) if client_counts else 0
    min_rep = min(client_counts.values()) if len(client_counts) == total_clients else 0

    if max_rep > 15:
        print("  [FINDING] Global ranking heavily biased toward one client!")
        print("  -> Per-client (S5-S7) may give better generalization via diversity.")
    elif min_rep == 0:
        print("  [FINDING] Some clients get 0 trees in global top-20!")
        print("  -> Their data distribution is completely ignored.")
    else:
        print("  [OK] Reasonable distribution across clients.")
    return True


# ============================================================================
# TEST 8: StandardScaler impact on tree-based models
# ============================================================================
def test_scaling_impact():
    """Test if StandardScaler in benchmark helps or hurts tree models."""
    print("\n" + "="*70)
    print("TEST 8: StandardScaler impact on tree-based models")
    print("="*70)

    from src.domain.model.proactive_forest import ProactiveForest
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y = iris.data, iris.target.astype(str)
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Without scaling
    m1 = ProactiveForest(n_estimators=50, alpha_pf=0.1, class_names=["0","1","2"], random_state=42)
    m1.fit(X_tr, y_tr)
    preds1 = m1.predict(X_test)
    f1_no_scale = f1_score(y_test, preds1, average="macro")

    # With scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_test_s = scaler.transform(X_test)
    m2 = ProactiveForest(n_estimators=50, alpha_pf=0.1, class_names=["0","1","2"], random_state=42)
    m2.fit(X_tr_s, y_tr)
    preds2 = m2.predict(X_test_s)
    f1_with_scale = f1_score(y_test, preds2, average="macro")

    print(f"  Without StandardScaler: F1={f1_no_scale:.4f}")
    print(f"  With StandardScaler:    F1={f1_with_scale:.4f}")
    diff = f1_with_scale - f1_no_scale
    print(f"  Difference: {diff:+.4f}")

    if abs(diff) < 0.01:
        print("  [OK] Scaling has minimal impact (expected for tree-based models).")
    elif diff < -0.01:
        print("  [FINDING] Scaling HURTS performance! Trees are scale-invariant.")
        print("  -> benchmark uses StandardScaler unnecessarily.")
    return True


# ============================================================================
# TEST 9: Validation data leakage in ProactiveForest.fit()
# ============================================================================
def test_validation_leak_small_data():
    """When data < 10 samples, fit() uses training data as validation."""
    print("\n" + "="*70)
    print("TEST 9: Validation data leakage with small datasets")
    print("="*70)

    from src.domain.model.proactive_forest import ProactiveForest

    X_small = np.random.randn(8, 4)
    y_small = np.array(["a","b","a","b","a","b","a","b"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = ProactiveForest(n_estimators=10, alpha_pf=0.1, class_names=["a","b"])
        try:
            m.fit(X_small, y_small)
            has_warning = any("Data leakage" in str(warning.message) for warning in w)
            if has_warning:
                print("  [OK] Warning raised about data leakage for small datasets.")
            else:
                print("  [ISSUE] No warning about data leakage.")
        except Exception as e:
            print(f"  [ERROR] Fit failed: {e}")
    return True


# ============================================================================
# TEST 10: Benchmark results analysis - S8 collapse detection
# ============================================================================
def test_benchmark_s8_collapse():
    """Analyze the actual benchmark CSV for S8 variant collapse."""
    print("\n" + "="*70)
    print("TEST 10: Benchmark results - S8 variant collapse analysis")
    print("="*70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "results_final_benchmark.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] No benchmark CSV found.")
        return True

    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    datasets = set(r["dataset"] for r in rows)
    s8_strategies = ["s8_weighted_average", "s8_simple_mean", "s8_median", "s8_consensus", "s8_proactive_pcd"]

    identical_count = 0
    total_datasets = 0
    for ds in sorted(datasets):
        ds_rows = {r["strategy"]: float(r["f1_mean"]) for r in rows if r["dataset"] == ds}
        s8_f1s = {s: ds_rows.get(s, -1) for s in s8_strategies if s in ds_rows}
        if len(s8_f1s) >= 2:
            total_datasets += 1
            vals = list(s8_f1s.values())
            if max(vals) - min(vals) < 0.001:
                identical_count += 1
                print(f"  {ds}: ALL S8 variants IDENTICAL (F1={vals[0]:.4f})")
            else:
                print(f"  {ds}: S8 variants differ (range={max(vals)-min(vals):.4f})")

    if identical_count > 0:
        print(f"\n  [CRITICAL] {identical_count}/{total_datasets} datasets have identical S8 results!")
        print("  Root cause: local_roulette_weight=0 forces full global roulette adoption.")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("FEDERATED PROACTIVE FOREST - DIAGNOSTIC EFFICACY ANALYSIS")
    print("=" * 70)

    results = {}
    tests = [
        ("TreeMetricExtractor PCD granularity", test_tree_metric_extractor_pcd_is_client_level),
        ("S8 variant differentiation", test_s8_variants_identical),
        ("HybridForest weight sensitivity", test_hybrid_forest_weights),
        ("Progressive convergence threshold", test_progressive_convergence_threshold),
        ("Label encoding roundtrip", test_label_encoding_roundtrip),
        ("Voting mode correctness", test_voting_mode),
        ("Global vs PerClient selection", test_global_vs_perclient_tree_selection),
        ("StandardScaler impact", test_scaling_impact),
        ("Validation data leakage", test_validation_leak_small_data),
        ("S8 benchmark collapse", test_benchmark_s8_collapse),
    ]

    for name, func in tests:
        try:
            result = func()
            results[name] = "PASS" if result else "ISSUE"
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[name] = f"ERROR: {e}"

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        status = "OK" if result == "PASS" else "XX"
        print(f"  [{status}] {name}: {result}")
