import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 1. Definir los resultados de la Tesis de Nayma (Baseline Centralizado)
nayma_results = {
    "Iris": 0.9549,
    "Car": 0.9457,
    "Nursery": 0.9548,
    "Vowel": 0.9684,
    "Letter": 0.9652,
    "Optdigits": 0.9822,
    "Sonar": 0.8234,
    "Spambase": 0.9527
}

# 2. Cargar mis resultados federados
file_path = "results/results_final_benchmark.csv"
try:
    df = pd.read_csv(file_path)
    if 'rep' not in df.columns:
        headers = ["rep", "strategy", "dataset", "f1_mean", "f1_std", "acc_mean", "acc_std", "recall_mean", "recall_std", "prec_mean", "prec_std", "pcd_mean", "timestamp"]
        df = pd.read_csv(file_path, names=headers)
    
    df = df[df['rep'].apply(lambda x: str(x).isdigit())]
    df['f1_mean'] = pd.to_numeric(df['f1_mean'])
    
    # Promediar reps si las hay (en este caso es 1 rep, 10-fold)
    df_pivot = df.pivot_table(index='dataset', columns='strategy', values='f1_mean')
    
    # 3. Integrar la Tesis como una "estrategia" más
    df_pivot['Nayma_PF_Centralized'] = df_pivot.index.map(nayma_results)
    
    # Filtrar solo los datasets comunes y estrategias principales
    common_datasets = list(nayma_results.keys())
    df_pivot = df_pivot.loc[common_datasets].dropna()
    
    print("--- Tabla de F1-Scores para Pruebas Estadísticas ---")
    print(df_pivot)
    
    # 4. Prueba de Friedman (General)
    # Convertimos a matriz para Friedman
    data_matrix = df_pivot.values
    stat, p_value = stats.friedmanchisquare(*[df_pivot[col] for col in df_pivot.columns])
    
    print(f"\n--- Test de Friedman ---")
    print(f"Estadístico: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("RESULTADO: Existen diferencias significativas entre las estrategias (p < 0.05).")
    else:
        print("RESULTADO: No se encontraron diferencias significativas globales.")

    # 5. Comparación Wilcoxon: Nayma Centralized vs Cada Estrategia
    print("\n--- Test de Wilcoxon (Nayma Centralized vs Tus Estrategias) ---")
    comparisons = []
    for col in df_pivot.columns:
        if col == 'Nayma_PF_Centralized': continue
        
        # Wilcoxon signed-rank test
        w_stat, w_p = stats.wilcoxon(df_pivot['Nayma_PF_Centralized'], df_pivot[col])
        comparisons.append({'Strategy': col, 'p-value': w_p, 'Mean_F1': df_pivot[col].mean()})
        
    comp_df = pd.DataFrame(comparisons).sort_values('p-value')
    print(comp_df)
    
    # 6. Conclusión Técnica
    print("\n--- Conclusión Estadística ---")
    best_fed = comp_df.iloc[comp_df['Mean_F1'].idxmax()]
    print(f"Tu mejor estrategia federada es: {best_fed['Strategy']} (Mean F1: {best_fed['Mean_F1']:.4f})")
    
    significant_diffs = comp_df[comp_df['p-value'] < 0.05]
    if not significant_diffs.empty:
        print(f"Hay {len(significant_diffs)} estrategias con diferencias significativas vs la Tesis.")
    else:
        print("No hay diferencias estadísticamente significativas vs la Tesis (tus resultados son comparables).")

except Exception as e:
    print(f"Error: {e}")
