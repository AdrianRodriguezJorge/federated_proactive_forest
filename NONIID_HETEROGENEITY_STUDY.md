# Non-IID Heterogeneity Study Guide

## Overview

Created 4 example configurations demonstrating the FLEX Non-IID Dirichlet distribution spectrum with different aggregation strategies. These enable systematic study of how data heterogeneity affects Proactive Forest federated learning.

## Alpha Parameter Guide

| Alpha | Heterogeneity | Client Distribution | Real-World Scenario | Use Case |
|-------|---------------|-------------------|-------------------|----------|
| **0.1** | Extreme | Highly skewed toward 1-2 classes | Different device types with specialized sensors | Study robustness to extreme non-IID |
| **0.5** | High | Skewed toward 2-3 classes | Geographic/demographic clustering | Study real-world heterogeneity |
| **1.0** | Moderate | Somewhat balanced, distinct patterns | Mixed device types / partial clustering | Study practical scenarios |
| **10.0** | Low | Nearly uniform (close to IID) | Well-stratified sampling | Study IID vs Non-IID transition |
| **∞** | None (IID) | Perfectly uniform | Use `distribution: iid` instead | Baseline for comparison |

## Configuration Reference

### [exp_iid_s1.yaml](../exp_iid_s1.yaml)
- **Distribution**: IID (uniform across clients)
- **Strategy**: S1 (Simple Pool - all trees)
- **Purpose**: Baseline IID scenario
- **Expected Accuracy**: Highest (all trees, uniform data)
- **Communication Cost**: Highest (all trees transmitted)

### [exp_noniid_low_heterogeneity_s2.yaml](../exp_noniid_low_heterogeneity_s2.yaml)
- **Distribution**: Non-IID Dirichlet (alpha=10.0)
- **Strategy**: S2 (Global Accuracy ranking)
- **Purpose**: Near-IID comparison, simple ranking
- **Expected Accuracy**: Similar to IID but slightly lower
- **When to Use**: Testing if heterogeneity level matters little
- **Comparison Target**: Run this + IID setup to validate alpha=10.0 ≈ IID

### [exp_noniid_moderate_heterogeneity_s5.yaml](../exp_noniid_moderate_heterogeneity_s5.yaml)
- **Distribution**: Non-IID Dirichlet (alpha=1.0)
- **Strategy**: S5 (Per-Client Accuracy)
- **Purpose**: Realistic heterogeneity with client-personalized selection
- **Expected Accuracy**: Moderate (benefits from per-client tuning)
- **When to Use**: Modeling realistic federated scenarios
- **Key Difference**: S5 selects trees best for each client's local data

### [exp_noniid_dirichlet_s7.yaml](../exp_noniid_dirichlet_s7.yaml)
- **Distribution**: Non-IID Dirichlet (alpha=0.5)
- **Strategy**: S7 (Per-Client F1+PCD)
- **Purpose**: High heterogeneity + diversity-aware selection
- **Expected Accuracy**: Good balance of accuracy + diversity
- **When to Use**: Classes are imbalanced and heterogeneous
- **Key Difference**: Balances accuracy (F1) with forest diversity (PCD)

### [exp_noniid_extreme_heterogeneity_s3.yaml](../exp_noniid_extreme_heterogeneity_s3.yaml)
- **Distribution**: Non-IID Dirichlet (alpha=0.1)
- **Strategy**: S3 (Global F1 ranking)
- **Purpose**: Extreme heterogeneity with global F1 focus
- **Expected Accuracy**: Lower accuracy on some clients (due to extreme non-IID)
- **When to Use**: Studying robustness to extreme data divergence
- **Key Insight**: Shows how much heterogeneity can hurt performance

## Recommended Experiment Sequences

### Sequence 1: Understanding Heterogeneity Impact
Run these in order to see how heterogeneity level affects performance:
1. `exp_iid_s1.yaml` - Baseline IID
2. `exp_noniid_low_heterogeneity_s2.yaml` - alpha=10.0 (should be similar)
3. `exp_noniid_moderate_heterogeneity_s5.yaml` - alpha=1.0
4. `exp_noniid_extreme_heterogeneity_s3.yaml` - alpha=0.1

**Expected Pattern**: Accuracy decreases as alpha decreases (more heterogeneity)

### Sequence 2: Strategy Comparison at Fixed Heterogeneity
Compare different strategies at the same heterogeneity level:
1. `exp_noniid_dirichlet_s7.yaml` (S7: F1+diversity)
2. Create new variant with S6 (replace `strategy: S6`)
3. Create new variant with S5 (replace `strategy: S5`)
4. Create new variant with S4 (replace `strategy: S4`)

**Expected Pattern**: Per-client strategies (S5,S6,S7) outperform global strategies (S1,S2,S3) when heterogeneity is high

### Sequence 3: Alpha Fine-Tuning
Run heterogeneity sweeps:
- Create variants: alpha = 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
- Keep all other params fixed (S5 strategy recommended)
- Plot: accuracy vs alpha

## Using These Configs with Streamlit

### Configuration Selection
When Streamlit `page_config.py` is updated with distribution selector:

```
Distribution Type:
  ○ IID (uniform)
  ○ Non-IID Dirichlet (heterogeneous)

If Non-IID Dirichlet:
  Alpha Parameter: [slider 0.1 - 10.0] ← controls heterogeneity
  Pre-configured:
    □ Extreme (0.1)
    □ High (0.5)
    □ Moderate (1.0)
    □ Low (10.0)
```

### Loading Config by Experiment Name
```python
# In Streamlit app
config_path = f"configs/experiments/{experiment_name}.yaml"
config = load_config(config_path)
orchestrator = FLEXOrchestratorV2(config)
results = orchestrator.run_federation()
```

## Interpretation Guide

### Metrics to Track Across Runs

| Metric | IID Expected | Non-IID Expected | What It Shows |
|--------|------------|-----------------|--------------|
| **Accuracy** | High, stable | Lower, varies by alpha | Model quality |
| **Per-Client Accuracy Variance** | Low | High (increases as alpha ↓) | Heterogeneity impact |
| **Communication Cost** | Moderate (depends on strategy) | Similar (strategy-driven) | Efficiency |
| **Convergence Rounds** | Fast (3-5) | Slower (5-10) | Data quality effect |

### Reading Results

**Good Signs:**
- Per-client strategies (S5,S6,S7) outperform global (S1,S2,S3) when alpha < 1.0
- F1+PCD strategies (S4,S7) show better balance on imbalanced datasets
- Low accuracy variance with alpha=10.0 (validates near-IID behavior)

**Warning Signs:**
- Global strategies outperform per-client strategies (may indicate weak heterogeneity signal)
- Accuracy collapse with alpha=0.1 (check that CPF early-stopping is working)
- Communication cost doesn't reflect strategy differences (check implementation)

## Advanced Usage

### Custom Alpha Values
Create new experiment config with custom alpha:
```yaml
federation:
  distribution: "noniid_dirichlet"
  alpha: 0.3  # Your custom value
```

### Multi-Round Convergence Study
Extend any config with:
```yaml
federation:
  n_rounds: 10  # Multiple federation rounds
  log_interval: 1
```

Then plot accuracy convergence curve across rounds.

### Comparing Different Datasets
Use configs as templates with different datasets:
- `dataset: Iris` (3 classes, small)
- `dataset: NSL-KDD` (5 classes, larger)
- `dataset: CUSTOM` (add to datasets infrastructure)

With more classes and larger datasets, non-IID effects are typically more pronounced.

## Debugging

### Issue: Alpha parameter not being applied
**Check**: 
- YAML format correct (`alpha: 0.5` not `alpha=0.5`)
- `distribution: noniid_dirichlet` specified (not `iid`)
- FLEXOrchestratorV2 is being used (not v1)

### Issue: Results look too similar across different alphas
**Check**:
- Dataset is large enough (non-IID effects need sample diversity)
- n_clients >= 5 (too few clients masks heterogeneity)
- Validation split is appropriate (0.2 recommended)

### Issue: Per-client strategies not outperforming global
**Check**:
- Alpha is actually low (< 1.0)
- Dataset has meaningful class imbalance
- Per-client trees are being selected correctly

## Next Steps

1. **Run baseline**: `exp_iid_s1.yaml` to verify setup works
2. **Run heterogeneity study**: All 4 configurations in sequence
3. **Analyze results**: Compare accuracy patterns with alpha values
4. **Extend**: Create custom configs for your specific scenarios
5. **Plot**: Generate convergence curves and heterogeneity impact graphs

## File Structure

```
configs/experiments/
├── exp_iid_s1.yaml
├── exp_noniid_low_heterogeneity_s2.yaml
├── exp_noniid_moderate_heterogeneity_s5.yaml
├── exp_noniid_dirichlet_s7.yaml
└── exp_noniid_extreme_heterogeneity_s3.yaml
```

All configs are compatible with `FLEXOrchestratorV2` and ready to run through Streamlit interface.
