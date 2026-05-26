import os
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# 1. Definir los resultados de la Tesis de Nayma (Baseline Centralizado) extraídos del CSV
NAYMA_F1 = {
    "Iris": 0.954981,
    "Car": 0.945782,
    "Nursery": 0.954848,
    "Vowel": 0.968468,
    "Optdigits": 0.982218,
    "Sonar": 0.823483,
    "Spambase": 0.952755,
}

NAYMA_ACC = {
    "Iris": 0.956000,
    "Car": 0.976626,
    "Nursery": 0.995911,
    "Vowel": 0.971919,
    "Optdigits": 0.983236,
    "Sonar": 0.848299,
    "Spambase": 0.953880,
}

NAYMA_PCD = {
    "Iris": 0.112000,
    "Car": 0.309630,
    "Nursery": 0.396870,
    "Vowel": 0.937000,
    "Optdigits": 0.583070,
    "Sonar": 0.907250,
    "Spambase": 0.330740,
}

def run_single_fold_analysis() -> None:
    """Loads comparative F1, Accuracy, and PCD JSON and runs Friedman and Wilcoxon tests."""
    json_path = os.path.join("results", "benchmark_single_fold_results.json")
    
    if not os.path.exists(json_path):
        print(f"Error: No se encontró el archivo de benchmark {json_path}")
        return

    # Leer resultados del benchmark federado
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    results = json_data["results"]
    common_datasets = list(NAYMA_F1.keys())

    # Extraer registros federados
    records = []
    for dataset, strategies in results.items():
        if dataset not in common_datasets:
            continue
        for strategy, details in strategies.items():
            if details.get("success"):
                metrics = details.get("metrics", {})
                records.append({
                    "Dataset": dataset,
                    "Strategy": strategy,
                    "Accuracy": metrics.get("accuracy"),
                    "F1": metrics.get("f1"),
                    "PCD": metrics.get("pcd")
                })
    df_fed = pd.DataFrame(records)

    # Pivotear la tabla de resultados para F1
    df_f1 = df_fed.pivot(index="Dataset", columns="Strategy", values="F1")
    df_f1["Centralized"] = df_f1.index.map(NAYMA_F1)
    df_f1 = df_f1.loc[common_datasets]

    # Preparar datos para Friedman
    strategies = [col for col in df_f1.columns if col != "Centralized"]
    columns_friedman = [df_f1["Centralized"].values] + [df_f1[strat].values for strat in strategies]

    print("=" * 80)
    print("--- ANÁLISIS ESTADÍSTICO DE COMPARACIÓN DE EFICACIA (F1-score) ---")
    print("=" * 80)
    print(f"Datasets analizados ({len(common_datasets)}): {', '.join(common_datasets)}")
    print(f"Resultados de benchmark cargados de: {json_path}\n")

    # Friedman Test
    stat_global, p_friedman_global = friedmanchisquare(*columns_friedman)
    print("PASO 1: Test de Friedman (Diferencias Globales)")
    print(f"  Estadístico: {stat_global:.4f}")
    print(f"  p-value: {p_friedman_global:.6f}")
    if p_friedman_global < 0.05:
        print("  Conclusión: EXISTEN diferencias significativas globales (p < 0.05).")
    else:
        print("  Conclusión: NO existen diferencias significativas globales.")
    print("-" * 80)

    # Wilcoxon Tests
    print("PASO 2: Análisis Post-hoc (Centralizado vs Cada Estrategia)")
    print(f"{'Estrategia':<25} | {'Mean F1':<10} | {'p-value':<10} | {'Diferencia':<10} | {'Significativa':<12}")
    print("-" * 80)
    
    mean_cent = df_f1["Centralized"].mean()
    print(f"{'* Centralized PF *':<25} | {mean_cent:.6f} | {'-':<10} | {'-':<10} | Baseline")
    
    sorted_strategies = sorted(strategies, key=lambda s: df_f1[s].mean(), reverse=True)
    for strat in sorted_strategies:
        mean_strat = df_f1[strat].mean()
        diff = mean_strat - mean_cent
        try:
            _, p_w = wilcoxon(df_f1["Centralized"].values, df_f1[strat].values)
            sig = "Sí (p<0.05)" if p_w < 0.05 else "No"
            p_w_str = f"{p_w:.6f}"
        except Exception:
            p_w_str = "Error"
            sig = "N/A"
        print(f"{strat:<25} | {mean_strat:.6f} | {p_w_str:<10} | {diff:+.6f} | {sig:<12}")
        
    print("=" * 80)

if __name__ == "__main__":
    run_single_fold_analysis()
