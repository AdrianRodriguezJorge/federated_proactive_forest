"""Generate 8 optimization notebooks: S1-S7 + PW, each with 6 dataset cells.

All notebooks read hyperparameter ranges from search_spaces.yaml — no hardcoded ranges.
"""
import json
import yaml
from pathlib import Path

# Find project root
for p in Path(__file__).parents:
    if (p / 'src').exists() and (p / 'data').exists():
        ROOT = p
        break
else:
    ROOT = Path(__file__).parent.parent.parent.parent.parent

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

STRATEGY_INFO = {
    'S1': 'Pools ALL trees from ALL clients. No selection.',
    'S2': 'Ranks trees by global accuracy. Round-robin aggregation.',
    'S3': 'Ranks trees by global macro-F1. Round-robin aggregation.',
    'S4': 'Ranks trees by combined F1 + PCD score. Global aggregation.',
    'S5': 'Ranks trees by per-client accuracy. Round-robin aggregation.',
    'S6': 'Ranks trees by per-client macro-F1. Round-robin aggregation.',
    'S7': 'Ranks trees by per-client combined F1 + PCD score.',
    'PW': 'Incremental round-robin with dynamic scoring. Multiple communication rounds.',
}

N_TRIALS = 20
N_TRIALS_PW = 40
SEED = 42
N_CLIENTS = 3


def make_loader_code(ds):
    """Generate dataset loader code."""
    tgt = ds['target_col']
    drops = ds['drop_cols']
    cat = ds['cat_cols']

    lines = []
    lines.append(f"# ── Load {ds['name']} ──────────────────────────────────────────────────")
    lines.append(f"df = pd.read_csv(DATA_DIR / '{ds['file']}')")
    lines.append(f"print(f'📊 Shape: {{df.shape}}')")

    if drops:
        lines.append(f"df = df.drop(columns={drops})")

    if cat:
        lines.append(f"cat_cols = {cat}")
        lines.append(f"enc = OrdinalEncoder()")
        lines.append(f"X = enc.fit_transform(df[cat_cols])")
    else:
        lines.append(f"X = df.drop(columns=['{tgt}']).values.astype(float)")

    lines.append(f"le = LabelEncoder()")
    lines.append(f"y = le.fit_transform(df['{tgt}'])")
    lines.append(f"class_names = list(le.classes_)")
    lines.append(f"feature_names = [c for c in df.columns if c != '{tgt}']")
    lines.append(f"X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)")
    lines.append(f"sc = StandardScaler()")
    lines.append(f"X_tr = sc.fit_transform(X_tr)")
    lines.append(f"X_te = sc.transform(X_te)")
    lines.append(f"ds = DatasetSplit(X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,")
    lines.append(f"                  feature_names=feature_names, class_names=class_names,")
    lines.append(f"                  dataset_name='{ds['name'].lower()}')")
    lines.append(f"print(f'✅ Train={{ds.X_train.shape[0]}}, Test={{ds.X_test.shape[0]}}, Feats={{ds.X_train.shape[1]}}, Classes={{len(class_names)}}')")
    return '\n'.join(lines)


def make_suggest_from_space():
    """Generate code that reads SPACE dict at runtime and creates suggest_* calls dynamically."""
    return """    params = {}
    for _name, _cfg in SPACE.items():
        _t = _cfg.get('type', 'float')
        _low, _high = _cfg['low'], _cfg['high']
        _step = _cfg.get('step')
        _log = _cfg.get('log', False)
        if _t == 'float':
            if _log:
                params[_name] = trial.suggest_float(_name, _low, _high, log=True)
            elif _step is not None:
                params[_name] = trial.suggest_float(_name, _low, _high, step=_step)
            else:
                params[_name] = trial.suggest_float(_name, _low, _high)
        elif _t == 'int':
            params[_name] = trial.suggest_int(_name, _low, _high, step=_step or 1)
    return params"""


def make_config_code(strat_key):
    """Generate config dict using params dict from objective()."""
    if strat_key == 'PW':
        return """        cfg = {
            'federation': {'n_clients': N_CLIENTS, 'distribution': 'iid', 'seed': SEED},
            'model': {
                'n_estimators': 100, 'alpha': params['alpha_pf'],
                'split_criterion': 'entropy', 'use_progressive_stopping': True,
                'convergence': 0.002, 'episode_size': 5, 'verbose': False,
            },
            'aggregation': {
                'strategy': 'PW',
                'window_size': params['window_size'],
                'max_rounds': params['max_rounds'],
                'f1_weight': params['f1_weight'],
                'convergence_threshold': params['convergence_threshold'],
            },
            'prediction': {
                'local_weight': params['local_weight'],
                'global_weight': 1.0 - params['local_weight'],
            },
            'verbose': False, 'seed': SEED,
        }"""
    else:
        return """        agg = {'strategy': STRATEGY}
        if 't_max' in params:
            agg['t_max'] = params['t_max']
        if 'f1_weight' in params:
            agg['f1_weight'] = params['f1_weight']
            agg['pcd_weight'] = 1.0 - params['f1_weight']
        cfg = {
            'federation': {'n_clients': N_CLIENTS, 'distribution': 'iid', 'seed': SEED},
            'model': {
                'n_estimators': 100, 'alpha': params['alpha_pf'],
                'split_criterion': 'entropy', 'use_progressive_stopping': True,
                'convergence': 0.002, 'episode_size': 5, 'verbose': False,
            },
            'aggregation': agg,
            'prediction': {
                'local_weight': params['local_weight'],
                'global_weight': 1.0 - params['local_weight'],
            },
            'verbose': False, 'seed': SEED,
        }"""


def make_orchestrator_code(strat_key):
    """Generate orchestrator run code."""
    if strat_key == 'PW':
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


def make_notebook(strat_key, space_dict):
    """Create a full Jupyter notebook for one strategy, reading ranges from YAML at runtime."""
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

    param_list = list(space_dict.keys())
    info = STRATEGY_INFO.get(strat_key, '')

    # ── Title ──
    add_cell(
        f"# 🔍 Optimización — {strat_key}\n\n"
        f"## Federated Proactive Forest\n\n"
        f"**Estrategia:** {info}\n\n"
        f"**Hiperparámetros:** {', '.join(f'`{p}`' for p in param_list)}\n\n"
        f"**Datasets:** Letter, Optdigits, Spambase, Nursery, Sonar, Vowel\n\n"
        f"**N_CLIENTS:** {N_CLIENTS}\n\n"
        f"> ⚡ **Cada celda de dataset es independiente** — ejecuta solo la que necesites.\n\n"
        f"> 📋 **Rangos leídos en runtime desde** `configs/experiments/optimization/search_spaces.yaml`",
        'markdown'
    )

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
        f"N_CLIENTS = {N_CLIENTS}",
        "N_TRIALS = 20" if strat_key != 'PW' else "N_TRIALS = 40",
        f"STRATEGY = '{strat_key}'",
        "DATA_DIR = ROOT / 'data'",
        "RESULTS_DIR = ROOT / 'results' / 's6_alpha_optimization'",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "# Load search space from YAML (single source of truth)",
        "with open(ROOT / 'configs' / 'experiments' / 'optimization' / 'search_spaces.yaml') as f:",
        "    all_spaces = yaml.safe_load(f)",
        "SPACE = all_spaces[STRATEGY]",
        "",
        "from src.domain.dataset.base_adapter import DatasetSplit",
    ]
    if strat_key == 'PW':
        setup.append("from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator")
    else:
        setup.append("from src.application.fl_orchestrator import FLEXOrchestrator")
    setup.append("")
    setup.append("print(f'✅ Project root: {ROOT}')")
    setup.append(f"print(f'✅ Strategy: {{STRATEGY}}')")
    setup.append(f"print(f'✅ N_CLIENTS: {{N_CLIENTS}}')")
    setup.append("print(f'✅ Search space: {list(SPACE.keys())}')")
    add_cell('\n'.join(setup))

    # ── Shared code blocks ──
    suggest_code = make_suggest_from_space()
    config_code = make_config_code(strat_key)
    orch_code = make_orchestrator_code(strat_key)

    # ── One cell per dataset ──
    for ds_key, ds in DATASETS.items():
        loader = make_loader_code(ds)

        cell_src = f"""{loader}

# ── Optimization ──────────────────────────────────────────────────────────
def objective(trial):
{suggest_code}
{config_code}
{orch_code}

print(f"🚀 Optimizando {strat_key} en {{ds.dataset_name}}... ({{N_TRIALS}} trials)")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

df_trials = study.trials_dataframe()
print(f"\\n{{'='*70}}")
print(f"📊 {strat_key} — {{ds.dataset_name}} — Resultados")
print(f"{{'='*70}}")
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
        "results = []",
        f"for fp in sorted(RESULTS_DIR.glob('s6_*_{strat_key.lower()}_results.json')):",
        "    with open(fp) as f:",
        "        r = json.load(f)",
        "    row = {'dataset': r['dataset'], 'best_f1': round(r['best_macro_f1'], 4),",
        "           'mean_f1': round(r['mean_macro_f1'], 4), 'std': round(r['std_macro_f1'], 4)}",
        "    row.update(r['best_params'])",
        "    results.append(row)",
        "",
        "df_sum = pd.DataFrame(results)",
        "print(f\"\\n{'='*90}\")",
        f"print(f\"📋 RESUMEN GLOBAL — {strat_key}\")",
        "print(f\"{'='*90}\")",
        "if df_sum.empty:",
        "    print('⚠️ No hay resultados. Ejecuta al menos una celda de dataset primero.')",
        "else:",
        "    print(df_sum.to_string(index=False))",
        "    if 'alpha_pf' in df_sum.columns:",
        "        rec = df_sum['alpha_pf'].median()",
        "        print(f\"\\n🎯 alpha_pf recomendado (mediana): {rec:.1f}\")",
        "        print(f\"   Rango: [{df_sum['alpha_pf'].min()} — {df_sum['alpha_pf'].max()}]\")",
        "    df_sum.to_csv(RESULTS_DIR / f'summary_{strat_key.lower()}.csv', index=False)",
        "    print(f\"\\n✅ Resumen guardado: summary_{strat_key.lower()}.csv\")",
    ]
    add_cell(
        "## 📊 Resumen Global — Comparar todos los datasets\n\n"
        "> ⚡ Ejecuta esta celda **después** de haber ejecutado todas las celdas de datasets.",
        'markdown'
    )
    add_cell('\n'.join(summary))

    # Save
    out_path = OUT / f'optuna_{strat_key.lower()}_alpha_pf.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅ {out_path.name}")


# ── Load search_spaces.yaml and generate notebooks ────────────────────────────
spaces_path = ROOT / 'configs' / 'experiments' / 'optimization' / 'search_spaces.yaml'
with open(spaces_path, encoding='utf-8') as f:
    all_spaces = yaml.safe_load(f)

for strat_key in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'PW']:
    if strat_key not in all_spaces:
        print(f"⚠️ Skipping {strat_key}: not in search_spaces.yaml")
        continue
    make_notebook(strat_key, all_spaces[strat_key])

print(f"\n🎉 8 notebooks generated in: {OUT}")
