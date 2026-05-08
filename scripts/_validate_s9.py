"""Quick validation script for S9 domain layer imports and logic."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# 1. Test RouletteUpdater
from src.domain.update.roulette_updater import RouletteUpdater
u = RouletteUpdater(beta=0.3)
local = np.array([0.4, 0.3, 0.2, 0.1])
glob = np.array([0.25, 0.25, 0.25, 0.25])
fused = u.fuse(local, glob)
print("Fused roulette: %s, sum=%.6f" % (fused, fused.sum()))
assert abs(fused.sum() - 1.0) < 1e-10, "Fused roulette does not sum to 1!"
print("[OK] RouletteUpdater")

# 2. Test S9 strategies
from src.domain.aggregation.strategies.s9_roulette_strategy import (
    create_roulette_strategy,
    S9WeightedAverageStrategy,
    S9SimpleMeanStrategy,
    S9MedianStrategy,
    S9ConsensusStrategy,
)

vecs = {
    "c1": np.array([0.4, 0.3, 0.2, 0.1]),
    "c2": np.array([0.1, 0.2, 0.3, 0.4]),
    "c3": np.array([0.25, 0.25, 0.25, 0.25]),
}
sizes = {"c1": 1000, "c2": 500, "c3": 250}
f1s = {"c1": 0.92, "c2": 0.70, "c3": 0.85}

for name in ["S9_MEAN", "S9_WEIGHTED", "S9_MEDIAN", "S9_CONSENSUS"]:
    s = create_roulette_strategy(name)
    r = s.aggregate_vectors(vecs, client_dataset_sizes=sizes, client_f1_scores=f1s)
    assert abs(r.sum() - 1.0) < 1e-10, "%s result does not sum to 1!" % name
    print("  %s: %s (sum=%.6f) [OK]" % (name, r, r.sum()))

# 3. Test ProactiveForestClassifier get/set probabilities
from src.domain.model.cpf_implementation.estimator import ProactiveForestClassifier
clf = ProactiveForestClassifier(n_estimators=10, alpha=0.1)
probs = clf.get_feature_probabilities()
print("  Pre-fit probs: %s" % probs)

clf._n_features = 4
clf._feature_prob = [0.25, 0.25, 0.25, 0.25]
probs = clf.get_feature_probabilities()
assert len(probs) == 4, "Should have 4 probabilities"
print("  Post-set probs: %s [OK]" % probs)

clf.set_feature_probabilities([0.1, 0.2, 0.3, 0.4])
probs = clf.get_feature_probabilities()
assert probs == [0.1, 0.2, 0.3, 0.4], "Set failed!"
print("  After set: %s [OK]" % probs)

# 4. Test ProactiveForest wrapper
from src.domain.model.proactive_forest import ProactiveForest
pf = ProactiveForest(n_estimators=10, alpha=0.1)
pf._classifier._n_features = 4
pf._classifier._feature_prob = [0.25, 0.25, 0.25, 0.25]
vec = pf.get_feature_probabilities()
assert len(vec) == 4
print("  ProactiveForest.get_feature_probabilities: %s [OK]" % vec)

new_probs = np.array([0.1, 0.2, 0.3, 0.4])
pf.set_feature_probabilities(new_probs)
vec2 = pf.get_feature_probabilities()
assert np.allclose(vec2, new_probs)
print("  ProactiveForest.set_feature_probabilities: %s [OK]" % vec2)

print("")
print("=" * 50)
print("[PASS] ALL S9 DOMAIN LAYER TESTS PASSED")
print("=" * 50)
