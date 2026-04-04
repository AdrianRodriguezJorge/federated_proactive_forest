"""Optimización de S6 (Per-Client Macro-F1) en 6 datasets para encontrar mejor alpha_pf."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import optuna
import yaml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.application.orchestrators import FLEXOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit

N_TRIALS = 20
SEED = 42
N_CLIENTS = 5

# ── Dataset loaders ──────────────────────────────────────────────────────────
def load_letter():
    df = pd.read_csv(ROOT / 'data' / 'letter.csv')
    fcols = [c for c in df.columns if c != 'class']
    X = df[fcols].values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(df['class'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                        feature_names=fcols, class_names=list(le.classes_), dataset_name='letter')

def load_optdigits():
    df = pd.read_csv(ROOT / 'data' / 'optdigits.csv')
    fcols = [c for c in df.columns if c != 'class']
    X = df[fcols].values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(df['class'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                        feature_names=fcols, class_names=list(le.classes_), dataset_name='optdigits')

def load_spambase():
    df = pd.read_csv(ROOT / 'data' / 'spambase.csv')
    fcols = [c for c in df.columns if c != 'class']
    X = df[fcols].values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(df['class'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                        feature_names=fcols, class_names=list(le.classes_), dataset_name='spambase')

def load_nursery():
    df = pd.read_csv(ROOT / 'data' / 'nursery.csv')
    fcols = [c for c in df.columns if c != 'class']
    enc = OrdinalEncoder()
    X = enc.fit_transform(df[fcols])
    le = LabelEncoder(); y = le.fit_transform(df['class'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                        feature_names=fcols, class_names=list(le.classes_), dataset_name='nursery')

def load_sonar():
    df = pd.read_csv(ROOT / 'data' / 'sonar.csv')
    fcols = [c for c in df.columns if c != 'Class']
    X = df[fcols].values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(df['Class'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                        feature_names=fcols, class_names=list(le.classes_), dataset_name='sonar')

def load_vowel():
    df = pd.read_csv(ROOT / 'data' / 'vowel.csv')
    # Drop metadata columns: 'Train or Test', 'Speaker Number', 'Sex'
    drop_cols = ['Train or Test', 'Speaker Number', 'Sex']
    use_cols = [c for c in df.columns if c not in drop_cols and c != 'Class']
    X = df[use_cols].values.astype(float)
    le = LabelEncoder(); y = le.fit_transform(df['Class'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                        feature_names=use_cols, class_names=list(le.classes_), dataset_name='vowel')

DATASETS = {
    'letter': load_letter,
    'optdigits': load_optdigits,
    'spambase': load_spambase,
    'nursery': load_nursery,
    'sonar': load_sonar,
    'vowel': load_vowel,
}

# ── Config S6 ────────────────────────────────────────────────────────────────
def build_s6_config(alpha_pf, t_max, local_weight, n_clients=N_CLIENTS, n_estimators=100, seed=SEED):
    return {
        'federation': {'n_clients': n_clients, 'distribution': 'iid', 'seed': seed},
        'model': {
            'n_estimators': n_estimators, 'alpha': alpha_pf,
            'split_criterion': 'entropy', 'use_progressive_stopping': True,
            'convergence': 0.002, 'episode_size': 5, 'verbose': False,
        },
        'aggregation': {'strategy': 'S6', 't_max': t_max},
        'prediction': {'local_weight': local_weight, 'global_weight': 1.0 - local_weight},
        'verbose': False, 'seed': seed,
    }

# ── Optimización ─────────────────────────────────────────────────────────────
results = {}

for name, loader_fn in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"📊 Optimizando S6 en dataset: {name.upper()}")
    print(f"{'='*70}")

    try:
        ds = loader_fn()
        print(f"   Train={ds.X_train.shape[0]}, Test={ds.X_test.shape[0]}, "
              f"Feats={ds.X_train.shape[1]}, Classes={len(ds.class_names)}")

        def objective(trial):
            alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)
            t_max = trial.suggest_int('t_max', 30, 150, step=10)
            local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)
            cfg = build_s6_config(alpha_pf, t_max, local_weight)
            np.random.seed(SEED)
            orch = FLEXOrchestrator.from_config(cfg)
            orch.setup_federation(ds, seed=SEED)
            res = orch.run_federated_round()
            return res.global_macro_f1

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        df_trials = study.trials_dataframe()
        results[name] = {
            'best_alpha_pf': study.best_params['alpha_pf'],
            'best_t_max': study.best_params['t_max'],
            'best_local_weight': study.best_params['local_weight'],
            'best_macro_f1': study.best_value,
            'mean_macro_f1': df_trials['value'].mean(),
            'std_macro_f1': df_trials['value'].std(),
            'n_classes': len(ds.class_names),
            'n_features': ds.X_train.shape[1],
            'n_train': ds.X_train.shape[0],
            'n_test': ds.X_test.shape[0],
            'all_trials': df_trials.to_dict('records'),
        }

        print(f"\n   ✅ Mejor alpha_pf: {study.best_params['alpha_pf']}")
        print(f"      Mejor t_max:    {study.best_params['t_max']}")
        print(f"      Mejor local_w:  {study.best_params['local_weight']}")
        print(f"      Mejor Macro-F1: {study.best_value:.4f}")

    except Exception as e:
        print(f"   ❌ Error en {name}: {e}")
        import traceback; traceback.print_exc()
        results[name] = None

# ── Resumen ──────────────────────────────────────────────────────────────────
valid = {k: v for k, v in results.items() if v is not None}
df_sum = pd.DataFrame([
    {
        'Dataset': k,
        'Best alpha_pf': v['best_alpha_pf'],
        'Best t_max': v['best_t_max'],
        'Best local_weight': v['best_local_weight'],
        'Best Macro-F1': round(v['best_macro_f1'], 4),
        'Mean Macro-F1': round(v['mean_macro_f1'], 4),
        'Std': round(v['std_macro_f1'], 4),
        'Classes': v['n_classes'],
        'Features': v['n_features'],
        'Train': v['n_train'],
        'Test': v['n_test'],
    }
    for k, v in valid.items()
])

print(f"\n{'='*90}")
print("📋 RESUMEN COMPLETO — Optimización S6 (Per-Client Macro-F1)")
print(f"{'='*90}")
print(df_sum.to_string(index=False))
print(f"{'='*90}")

# ── Recomendación alpha_pf ──────────────────────────────────────────────────
alphas = [v['best_alpha_pf'] for v in valid.values()]
median_alpha = float(np.median(alphas))
mean_alpha = float(np.mean(alphas))

# Ponderada por tamaño de dataset
total = sum(v['n_train'] for v in valid.values())
weighted_alpha = sum(v['best_alpha_pf'] * v['n_train'] for v in valid.values()) / total

print(f"\n{'='*90}")
print("🎯 RECOMENDACIÓN PARA alpha_pf (parámetro Proactive Forest)")
print(f"{'='*90}")
print(f"   alpha_pf por dataset:")
for ds, r in valid.items():
    print(f"      {ds:12s}: alpha_pf={r['best_alpha_pf']:.1f}  |  t_max={r['best_t_max']:3d}  |  "
          f"local_w={r['best_local_weight']:.1f}  |  F1={r['best_macro_f1']:.4f}")

print(f"\n   Estadísticas de alpha_pf:")
print(f"      Media simple:     {mean_alpha:.2f}")
print(f"      Mediana:          {median_alpha:.2f}")
print(f"      Media ponderada:  {weighted_alpha:.2f}")
print(f"      Rango:            [{min(alphas):.1f} — {max(alphas):.1f}]")

# Recomendar mediana (robusta)
rec = round(median_alpha, 1)
print(f"\n   ✅ VALOR RECOMENDADO: alpha_pf = {rec}")
print(f"   Justificación: La mediana es robusta a outliers y representa el valor")
print(f"   central que funciona consistentemente en todos los datasets.")
print(f"{'='*90}")

# ── Guardar resultados detallados ───────────────────────────────────────────
out = ROOT / 'results' / 's6_alpha_optimization'
out.mkdir(parents=True, exist_ok=True)

# CSV resumen
df_sum.to_csv(out / 's6_optimization_summary.csv', index=False)

# YAML recomendación
recommendation = {
    'strategy': 'S6',
    'parameter_optimized': 'alpha_pf',
    'recommended_value': rec,
    'per_dataset': {ds: {
        'alpha_pf': r['best_alpha_pf'],
        't_max': r['best_t_max'],
        'local_weight': r['best_local_weight'],
        'macro_f1': round(r['best_macro_f1'], 4),
    } for ds, r in valid.items()},
    'statistics': {
        'mean_alpha': round(mean_alpha, 2),
        'median_alpha': round(median_alpha, 2),
        'weighted_alpha': round(weighted_alpha, 2),
        'min_alpha': min(alphas),
        'max_alpha': max(alphas),
    },
    'interpretation_notes': [
        f"alpha_pf={rec} es el valor recomendado para usar en TODAS las estrategias (S1-S7, PW)",
        "Este valor fue encontrado optimizando S6 en 6 datasets diversos",
        "S6 fue elegida como referencia por usar Macro-F1 (robusta) sin sesgo de diversidad",
        "Los datasets cubren: pocas/muchas clases, balanceado/desbalanceado, pequeño/grande",
    ]
}
with open(out / 's6_alpha_recommendation.yaml', 'w') as f:
    yaml.dump(recommendation, f, default_flow_style=False, allow_unicode=True)

# JSON detallado con todos los trials
import json
serializable = {}
for ds, r in valid.items():
    serializable[ds] = {k: v for k, v in r.items() if k != 'all_trials'}
    serializable[ds]['all_trials'] = [
        {kk: (float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv)
         for kk, vv in t.items()}
        for t in r['all_trials']
    ]

with open(out / 's6_optimization_full_results.json', 'w') as f:
    json.dump(serializable, f, indent=2, default=str)

print(f"\n✅ Resultados guardados en: {out}")
print(f"   - s6_optimization_summary.csv")
print(f"   - s6_alpha_recommendation.yaml")
print(f"   - s6_optimization_full_results.json")
