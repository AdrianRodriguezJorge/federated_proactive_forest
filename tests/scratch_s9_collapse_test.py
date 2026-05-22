"""Demonstration of S9 local_roulette_weight=0 collapse across federated rounds."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.update.roulette_updater import RouletteUpdater
from src.domain.aggregation.strategies.s9_roulette_strategy import (
    S9WeightedAverageStrategy, S9SimpleMeanStrategy, S9MedianStrategy,
    S9ConsensusStrategy, S9ProactivePCDStrategy
)


def run_simulation(local_roulette_weight: float):
    print(f"\n==========================================")
    print(f"SIMULATING S9 FEDERATION WITH local_roulette_weight = {local_roulette_weight}")
    print(f"==========================================")

    # 3 features
    n_features = 3
    # Initial local roulette vectors (randomly initialized for each client)
    np.random.seed(10)
    c1_local = np.array([0.7, 0.2, 0.1])
    c2_local = np.array([0.1, 0.8, 0.1])
    c3_local = np.array([0.2, 0.2, 0.6])

    print(f"Initial Roulettes:")
    print(f"  Client 1: {c1_local}")
    print(f"  Client 2: {c2_local}")
    print(f"  Client 3: {c3_local}")

    updater = RouletteUpdater(local_roulette_weight=local_roulette_weight)

    # Let's run 3 rounds
    client_vectors = {"c1": c1_local.copy(), "c2": c2_local.copy(), "c3": c3_local.copy()}
    sizes = {"c1": 100, "c2": 100, "c3": 100}
    f1s = {"c1": 0.9, "c2": 0.8, "c3": 0.85}
    pcds = {"c1": 0.3, "c2": 0.4, "c3": 0.35}

    strategies = {
        "Simple Mean": S9SimpleMeanStrategy(),
        "Weighted Average": S9WeightedAverageStrategy(),
        "Median": S9MedianStrategy(),
        "Consensus": S9ConsensusStrategy(),
        "Proactive PCD": S9ProactivePCDStrategy(),
    }

    for round_idx in range(1, 4):
        print(f"\n--- ROUND {round_idx} ---")
        # Step 1: Server aggregates the current client vectors using the 5 strategies
        # To show what each strategy outputs:
        global_outputs = {}
        for name, strat in strategies.items():
            if name == "Simple Mean":
                global_outputs[name] = strat.aggregate_vectors(client_vectors)
            elif name == "Weighted Average":
                global_outputs[name] = strat.aggregate_vectors(client_vectors, sizes)
            elif name == "Median":
                global_outputs[name] = strat.aggregate_vectors(client_vectors)
            elif name == "Consensus":
                global_outputs[name] = strat.aggregate_vectors(client_vectors, client_f1_scores=f1s)
            elif name == "Proactive PCD":
                global_outputs[name] = strat.aggregate_vectors(client_vectors, client_pcd_scores=pcds)

        print("Global Roulette aggregated by each strategy:")
        for name, vec in global_outputs.items():
            print(f"  {name:16}: {vec}")

        # Let's pick "Simple Mean" as the chosen global vector deployed to clients
        # (This is what happens when running one variant)
        global_deployed = global_outputs["Simple Mean"]

        # Step 2: Clients receive global roulette and fuse it with their local roulette
        # Then, during local training, they generate some small updates (say, noise)
        # representing their new local model learning.
        for cid in ["c1", "c2", "c3"]:
            # Client fuses local with global
            fused = updater.fuse(client_vectors[cid], global_deployed)
            # Simulated local training update: add a small local bias/noise and normalize
            noise = np.random.dirichlet(np.ones(n_features)) * 0.1
            new_local = fused + noise
            client_vectors[cid] = new_local / np.sum(new_local)

        print("\nNew Client Roulettes (after fusion + local learning):")
        for cid, vec in client_vectors.items():
            print(f"  {cid}: {vec}")


if __name__ == "__main__":
    run_simulation(local_roulette_weight=0.0)
    run_simulation(local_roulette_weight=0.3)
