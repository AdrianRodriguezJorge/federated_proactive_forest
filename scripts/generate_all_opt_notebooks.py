"""Generate 8 optimization notebooks: S1-S7 + PW, each with 6 dataset cells."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
# Fix: go up from scripts/ to project root
for p in Path(__file__).parents:
    if (p / 'src').exists() and (p / 'data').exists():
        ROOT = p
        break
OUT = ROOT / 'src' / 'interfaces' / 'notebooks' / 'optimization'
OUT.mkdir(parents=True, exist_ok=True)

# ── Dataset definitions ──────────────────────────────────────────────────────
DATASETS = {
    'letter': {
        'file': 'letter.csv', 'target_col': 'class', 'cat_cols': [],
        'drop_cols': [], 'name': 'Letter',
        'desc': '20,000 samples, 16 features, 26 classes (A-Z), numeric features'
    },
    'optdigits': {
        'file': 'optdigits.csv', 'target_col': 'class', 'cat_cols': [],
        'drop_cols': [], 'name': 'Optdigits',
        'desc': '5,620 samples, 64 features (8x8 pixel), 10 classes (0-9), numeric 0-16'
    },
    'spambase': {
        'file': 'spambase.csv', 'target_col': 'class', 'cat_cols': [],
        'drop_cols': [], 'name': 'Spambase',
        'desc': '4,601 samples, 57 features (word frequencies), 2 classes (spam/ham)'
    },
    'nursery': {
        'file': 'nursery.csv', 'target_col': 'class',
        'cat_cols': ['parents','has_nurs','form','children','housing','finance','social','health'],
        'drop_cols': [], 'name': 'Nursery',
        'desc': '12,960 samples, 8 categorical features, 5 classes'
    },
    'sonar': {
        'file': 'sonar.csv', 'target_col': 'Class', 'cat_cols': [],
        'drop_cols': [], 'name': 'Sonar',
        'desc': '208 samples, 60 numeric features, 2 classes (Rock/Mine) — small dataset!'
    },
    'vowel': {
        'file': 'vowel.csv', 'target_col': 'Class', 'cat_cols': [],
        'drop_cols': ['Train or Test', 'Speaker Number', 'Sex'], 'name': 'Vowel',
        'desc': '990 samples, 10 numeric features (drop 3 metadata cols), 11 classes'
    },
}

# ── Strategy configs ─────────────────────────────────────────────────────────
STRATEGIES = {
    'S1': {
        'title': 'S1 — Simple Pool',
        'desc': 'Pools ALL trees from ALL clients. No selection. Only alpha_pf + local_weight.',
        'params': ['alpha_pf', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S1'},
    },
    'S2': {
        'title': 'S2 — Global Accuracy',
        'desc': 'Ranks trees by global accuracy. Round-robin aggregation.',
        'params': ['alpha_pf', 't_max', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "t_max = trial.suggest_int('t_max', 30, 150, step=10)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S2', 't_max': 100},
    },
    'S3': {
        'title': 'S3 — Global Macro-F1',
        'desc': 'Ranks trees by global macro-F1. Round-robin aggregation.',
        'params': ['alpha_pf', 't_max', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "t_max = trial.suggest_int('t_max', 30, 150, step=10)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S3', 't_max': 100},
    },
    'S4': {
        'title': 'S4 — Global F1 + PCD',
        'desc': 'Ranks trees by combined F1 + PCD score. Global aggregation.',
        'params': ['alpha_pf', 't_max', 'f1_weight', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "t_max = trial.suggest_int('t_max', 30, 150, step=10)",
            "f1_weight = trial.suggest_float('f1_weight', 0.0, 1.0, step=0.1)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S4', 't_max': 100, 'f1_weight': 0.5, 'pcd_weight': 0.5},
    },
    'S5': {
        'title': 'S5 — Per-Client Accuracy',
        'desc': 'Ranks trees by per-client accuracy. Round-robin aggregation.',
        'params': ['alpha_pf', 't_max', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "t_max = trial.suggest_int('t_max', 30, 150, step=10)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S5', 't_max': 100},
    },
    'S6': {
        'title': 'S6 — Per-Client Macro-F1',
        'desc': 'Ranks trees by per-client macro-F1. Round-robin aggregation.',
        'params': ['alpha_pf', 't_max', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "t_max = trial.suggest_int('t_max', 30, 150, step=10)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S6', 't_max': 100},
    },
    'S7': {
        'title': 'S7 — Per-Client F1 + PCD',
        'desc': 'Ranks trees by per-client combined F1 + PCD score.',
        'params': ['alpha_pf', 't_max', 'f1_weight', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "t_max = trial.suggest_int('t_max', 30, 150, step=10)",
            "f1_weight = trial.suggest_float('f1_weight', 0.0, 1.0, step=0.1)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'S7', 't_max': 100, 'f1_weight': 0.5, 'pcd_weight': 0.5},
    },
    'PW': {
        'title': 'PW — Progressive Windows',
        'desc': 'Incremental round-robin with dynamic scoring. Multiple communication rounds.',
        'params': ['alpha_pf', 'window_size', 'max_rounds', 'alpha_pw', 'convergence_threshold', 'local_weight'],
        'suggest': [
            "alpha_pf = trial.suggest_float('alpha_pf', 0.1, 0.8, step=0.1)",
            "window_size = trial.suggest_int('window_size', 3, 15, step=1)",
            "max_rounds = trial.suggest_int('max_rounds', 10, 50, step=5)",
            "alpha_pw = trial.suggest_float('alpha_pw', 0.0, 1.0, step=0.1)",
            "convergence_threshold = trial.suggest_float('convergence_threshold', 0.001, 0.01)",
            "local_weight = trial.suggest_float('local_weight', 0.0, 1.0, step=0.1)",
        ],
        'aggregation': {'strategy': 'PW', 'window_size': 5, 'max_rounds': 20, 'alpha': 0.5, 'convergence_threshold': 0.002},
        'is_pw': True,
    },
}

N_TRIALS = 20
SEED = 42

def make_loader_code(ds_key, ds):
    """Generate dataset loader code for a notebook cell."""
    f = ds['file']
    tgt = ds['target_col']
    drops = ds['drop_cols']
    cat = ds['cat_cols']

    code = []
    code.append(f"# ── Load {ds['name']} ──────────────────────────────────────────────────")
    code.append(f"df = pd.read_csv(DATA_DIR / '{f}')")
    code.append(f"print(f'📊 Shape: {{df.shape}}')")

    if drops:
        code.append(f"# Drop metadata columns")
        code.append(f"df = df.drop(columns={drops})")

    if cat:
        code.append(f"# Encode categorical features")
        code.append(f"cat_cols = {cat}")
        code.append(f"enc = OrdinalEncoder()")
        code.append(f"X = enc.fit_transform(df[cat_cols])")
    else:
        code.append(f"X = df.drop(columns=['{tgt}']).values.astype(float)")

    code.append(f"le = LabelEncoder()")
    code.append(f"y = le.fit_transform(df['{tgt}'])")
    code.append(f"class_names = list(le.classes_)")
    code.append(f"feature_names = [c for c in df.columns if c != '{tgt}']")
    code.append(f"")
    code.append(f"# Train/test split (80/20, stratified)")
    code.append(f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)")
    code.append(f"sc = StandardScaler()")
    code.append(f"X_tr = sc.fit_transform(X_tr)")
    code.append(f"X_te = sc.transform(X_te)")
    code.append(f"")
    code.append(f"ds = DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,")
    code.append(f"                  feature_names=feature_names, class_names=class_names,")
    code.append(f"                  dataset_name='{ds_key}')")
    code.append(f"print(f'✅ Train={{ds.X_train.shape[0]}}, Test={{ds.X_test.shape[0]}}, Feats={{ds.X_train.shape[1]}}, Classes={{len(class_names)}}')")
    return '\n'.join(code)


def make_config_code(strat_key, strat):
    """Generate config code for a trial."""
    lines = []
    if strat.get('is_pw'):
        lines.append("        cfg = {")
        lines.append("            'federation': {'n_clients': N_CLIENTS, 'distribution': 'iid', 'seed': SEED},")
        lines.append("            'model': {")
        lines.append("                'n_estimators': 100, 'alpha': alpha_pf,")
        lines.append("                'split_criterion': 'entropy', 'use_progressive_stopping': True,")
        lines.append("                'convergence': 0.002, 'episode_size': 5, 'verbose': False,")
        lines.append("            },")
        lines.append("            'aggregation': {")
        lines.append("                'strategy': 'PW', 'window_size': window_size,")
        lines.append("                'max_rounds': max_rounds, 'alpha': alpha_pw,")
        lines.append("                'convergence_threshold': convergence_threshold,")
        lines.append("            },")
        lines.append("            'prediction': {'local_weight': local_weight, 'global_weight': 1.0 - local_weight},")
        lines.append("            'verbose': False, 'seed': SEED,")
        lines.append("        }")
    else:
        agg = strat['aggregation']
        lines.append("        cfg = {")
        lines.append("            'federation': {'n_clients': N_CLIENTS, 'distribution': 'iid', 'seed': SEED},")
        lines.append("            'model': {")
        lines.append("                'n_estimators': 100, 'alpha': alpha_pf,")
        lines.append("                'split_criterion': 'entropy', 'use_progressive_stopping': True,")
        lines.append("                'convergence': 0.002, 'episode_size': 5, 'verbose': False,")
        lines.append("            },")
        lines.append("            'aggregation': " + json.dumps(agg) + ",")
        lines.append("            'prediction': {'local_weight': local_weight, 'global_weight': 1.0 - local_weight},")
        lines.append("            'verbose': False, 'seed': SEED,")
        lines.append("        }")
    return '\n'.join(lines)


def make_orchestrator_code(strat):
    """Generate orchestrator run code."""
    if strat.get('is_pw'):
        return """        np.random.seed(SEED)
        orch = ProgressiveWindowsOrchestrator(config=cfg, dataset_split=ds, verbose=False)
        res = orch.run_federated_round()
        return res.global_macro_f1"""
    else:
        return """        np.random.seed(SEED)
        orch = FLEXOrchestrator.from_config(cfg)
        orch.setup_federation(ds, seed=SEED)
        res = orch.run_federated_round()
        return res.global_macro_f1"""


def make_notebook(strat_key, strat):
    """Create a full Jupyter notebook for one strategy."""
    nb = {
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.8.10"}},
        "nbformat": 4, "nbformat_minor": 5,
        "cells": []
    }

    def add_cell(source, cell_type='code'):
        nb['cells'].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": source if isinstance(source, list) else source.split('\n'),
            "execution_count": None,
            "outputs": [],
        })

    # ── Title ──
    add_cell(f"# 🔍 Optimización S6 alpha_pf — {strat['title']}\n\n"
             f"## Federated Proactive Forest\n\n"
             f"**Estrategia:** {strat['desc']}\n\n"
             f"**Hiperparámetros:** {', '.join(f'`{p}`' for p in strat['params'])}\n\n"
             f"**Datasets:** Letter, Optdigits, Spambase, Nursery, Sonar, Vowel\n\n"
             f"> ⚡ **Cada celda de dataset es independiente** — ejecuta solo la que necesites.",
             'markdown')

    # ── Setup cell ──
    setup = [
        "# ── Imports & Config ─────────────────────────────────────────────────────",
        "import sys",
        "from pathlib import Path",
        "",
        "ROOT = Path.cwd().parent.parent.parent.parent",
        "sys.path.insert(0, str(ROOT))",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import yaml",
        "import json",
        "import optuna",
        "from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler",
        "from sklearn.model_selection import train_test_split",
        "",
        "SEED = 42",
        "N_CLIENTS = 5",
        "N_TRIALS = 20",
        "DATA_DIR = ROOT / 'data'",
        "RESULTS_DIR = ROOT / 'results' / 's6_alpha_optimization'",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "from src.domain.dataset.base_adapter import DatasetSplit",
    ]
    if strat.get('is_pw'):
        setup.append("from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator")
    else:
        setup.append("from src.application.fl_orchestrator import FLEXOrchestrator")
    setup.append("")
    setup.append("print(f'✅ Project root: {ROOT}')")
    setup.append("print(f'✅ Strategy: {strat_key}')")
    add_cell('\n'.join(setup))

    # ── One cell per dataset ──
    for ds_key, ds in DATASETS.items():
        loader = make_loader_code(ds_key, ds)
        config = make_config_code(strat_key, strat)
        orch = make_orchestrator_code(strat)

        cell_src = f"""{loader}

# ── Optimization ──────────────────────────────────────────────────────────
def objective(trial):
    {'; '.join(strat['suggest'])}
{config}
{orch}

print(f"🚀 Optimizando {strat_key} en {ds['name']}... ({N_TRIALS} trials)")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

df_trials = study.trials_dataframe()
print(f"\\n{'='*70}")
print(f"📊 {strat_key} — {ds['name']} — Resultados")
print(f"{'='*70}")
print(f"   Mejor Macro-F1: {{study.best_value:.4f}}")
for p, v in study.best_params.items():
    print(f"   {{p:25s}}: {{v}}")
print(f"   Media Macro-F1:  {{df_trials['value'].mean():.4f}}")
print(f"   Std Macro-F1:    {{df_trials['value'].std():.4f}}")

# Save per-dataset results
result = {{
    'dataset': '{ds_key}',
    'strategy': '{strat_key}',
    'best_macro_f1': study.best_value,
    'mean_macro_f1': df_trials['value'].mean(),
    'std_macro_f1': df_trials['value'].std(),
    'best_params': study.best_params,
    'all_trials': df_trials.to_dict('records'),
}}
with open(RESULTS_DIR / 's6_{ds_key}_{strat_key.lower()}_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\\n✅ Resultados guardados: s6_{ds_key}_{strat_key.lower()}_results.json")"""

        add_cell(f"## Dataset: {ds['name']}\n\n{ds['desc']}", 'markdown')
        add_cell(cell_src)

    # ── Summary cell ──
    summary = [
        "# ── Resumen Global (ejecutar después de todas las celdas) ─────────────",
        "import glob",
        "",
        "results = []",
        f"for fp in sorted((RESULTS_DIR).glob('s6_*_{strat_key.lower()}_results.json')):",
        "    with open(fp) as f:",
        "        r = json.load(f)",
        "    row = {'dataset': r['dataset'], 'best_f1': round(r['best_macro_f1'], 4),",
        "           'mean_f1': round(r['mean_macro_f1'], 4), 'std': round(r['std_macro_f1'], 4)}",
        "    row.update(r['best_params'])",
        "    results.append(row)",
        "",
        "df_sum = pd.DataFrame(results)",
        "print(f\"\\n{'='*90}\")",
        f"print(f\"📋 RESUMEN GLOBAL — {strat['title']}\")",
        "print(f\"{'='*90}\")",
        "print(df_sum.to_string(index=False))",
        "",
        "# Recommend alpha_pf",
        "if 'alpha_pf' in df_sum.columns:",
        "    rec = df_sum['alpha_pf'].median()",
        "    print(f\"\\n🎯 alpha_pf recomendado (mediana): {{rec:.1f}}\")",
        "    print(f\"   Rango: [{{df_sum['alpha_pf'].min()}} — {{df_sum['alpha_pf'].max()}}]\")",
        "",
        "df_sum.to_csv(RESULTS_DIR / f'summary_{strat_key.lower()}.csv', index=False)",
        "print(f\"\\n✅ Resumen guardado: summary_{strat_key.lower()}.csv\")",
    ]
    add_cell("## 📊 Resumen Global — Comparar todos los datasets\n\n> ⚡ Ejecuta esta celda **después** de haber ejecutado todas las celdas de datasets.", 'markdown')
    add_cell('\n'.join(summary))

    # Save
    out_path = OUT / f'optuna_{strat_key.lower()}_alpha_pf.ipynb'
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✅ Created: {out_path.name}")


# ── Generate all 8 notebooks ──────────────────────────────────────────────────
for key, cfg in STRATEGIES.items():
    make_notebook(key, cfg)

print(f"\n🎉 8 notebooks generated in: {OUT}")
