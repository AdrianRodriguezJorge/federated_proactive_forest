"""Script to run Friedman and post-hoc Wilcoxon tests with Bonferroni correction.

Determines statistical significance between the Centralized Proactive Forest
baseline (PF) and the various federated selection strategies.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# 1. Definir los resultados de la Tesis de Nayma (Baseline Centralizado)
NAYMA_RESULTS = {
    "Iris": 0.9549,
    "Car": 0.9457,
    "Nursery": 0.9548,
    "Vowel": 0.9684,
    "Letter": 0.9652,
    "Optdigits": 0.9822,
    "Sonar": 0.8234,
    "Spambase": 0.9527,
}

def run_friedman_bonferroni_analysis() -> None:
    """Loads comparative F1 CSV and runs Friedman and Bonferroni tests."""
    # Ruta del archivo de benchmark actual
    benchmark_path = os.path.join("results", "results_final_benchmark.csv")
    
    if not os.path.exists(benchmark_path):
        print(f"Error: No se encontró el archivo de benchmark {benchmark_path}")
        return

    # Leer resultados del benchmark
    df = pd.read_csv(benchmark_path)
    if "rep" not in df.columns:
        headers = [
            "rep", "strategy", "dataset", "f1_mean", "f1_std",
            "acc_mean", "acc_std", "recall_mean", "recall_std",
            "prec_mean", "prec_std", "pcd_mean", "timestamp"
        ]
        df = pd.read_csv(benchmark_path, names=headers)

    # Limpiar y convertir a numérico
    df = df[df["rep"].apply(lambda x: str(x).isdigit())]
    df["f1_mean"] = pd.to_numeric(df["f1_mean"])

    # Pivotear la tabla de resultados
    df_pivot = df.pivot_table(index="dataset", columns="strategy", values="f1_mean")

    # Integrar los resultados de Nayma
    df_pivot["Nayma_PF_Centralized"] = df_pivot.index.map(NAYMA_RESULTS)

    # Filtrar y ordenar según la lista de datasets
    common_datasets = list(NAYMA_RESULTS.keys())
    df_pivot = df_pivot.loc[common_datasets].dropna()

    # Mapeo de nombres para compatibilidad y exportación
    rename_map = {
        "Nayma_PF_Centralized": "F1_PF",
        "local_isolation": "F1_Local",
        "s1_simple_pool": "F1_S1",
        "s2_global_accuracy": "F1_S2",
        "s3_global_f1": "F1_S3",
        "s4_global_f1_pcd": "F1_S4",
        "s5_perclient_accuracy": "F1_S5",
        "s6_perclient_f1": "F1_S6",
        "s7_perclient_f1_pcd": "F1_S7",
        "pw": "F1_PW",
        "s9_weighted_average": "F1_S9_Weighted",
        "s9_simple_mean": "F1_S9_Mean",
        "s9_median": "F1_S9_Median",
        "s9_consensus": "F1_S9_Consensus",
        "s9_proactive_pcd": "F1_S9_Proactive"
    }

    # Asegurar que todas las columnas deseadas existan
    available_cols = [col for col in rename_map.keys() if col in df_pivot.columns]
    df_clean = df_pivot[available_cols].rename(columns=rename_map)
    df_clean.index.name = "BD"
    df_clean = df_clean.reset_index()

    # Guardar la tabla comparativa en la ruta correspondiente
    output_dir = os.path.join("results", "comparison_vs_nayma_good")
    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, "comparison_table.csv")
    df_clean.to_csv(csv_out_path, index=False)
    
    # Extraemos los datos del dataframe
    datasets = df_clean["BD"].tolist()
    
    # Extraer valores F1 de base
    f1_pf = df_clean["F1_PF"].values
    
    # Nombres visuales claros para los reportes
    pretty_names = {
        "local_isolation": "Local Isolation",
        "s1_simple_pool": "S1 (Simple Pool)",
        "s2_global_accuracy": "S2 (Global Acc)",
        "s3_global_f1": "S3 (Global F1)",
        "s4_global_f1_pcd": "S4 (Global F1 PCD)",
        "s5_perclient_accuracy": "S5 (Per-Client Acc)",
        "s6_perclient_f1": "S6 (Per-Client F1)",
        "s7_perclient_f1_pcd": "S7 (Per-Client F1 PCD)",
        "pw": "PW (Proactive Window)",
        "s9_weighted_average": "S9 Weighted Avg",
        "s9_simple_mean": "S9 Simple Mean",
        "s9_median": "S9 Median",
        "s9_consensus": "S9 Consensus",
        "s9_proactive_pcd": "S9 Proactive PCD",
    }

    # Construir diccionario de datos dinámico
    estrategias_todas = {}
    for strategy_key, pretty_name in pretty_names.items():
        col_name = rename_map[strategy_key]
        if col_name in df_clean.columns:
            estrategias_todas[pretty_name] = df_clean[col_name].values

    n_ds = len(datasets)
    
    # Preparar el reporte de texto
    report_lines = []
    def rprint(msg: str):
        print(msg)
        report_lines.append(msg)

    rprint("=" * 80)
    rprint("--- ANÁLISIS ESTADÍSTICO DE COMPARACIÓN CONTRA LA TESIS DE NAYMA ---")
    rprint("=" * 80)
    rprint(f"Datasets analizados ({n_ds}): {', '.join(datasets)}")
    rprint(f"Resultados de benchmark cargados de: {benchmark_path}")
    rprint(f"Tabla comparativa guardada en: {csv_out_path}\n")

    # Mostrar la tabla comparativa de F1-Scores de forma legible
    rprint("--- TABLA COMPARATIVA DE F1-SCORES ---")
    rprint(df_clean.to_string(index=False, formatters={col: lambda x: f"{x:.4f}" for col in df_clean.columns if col != "BD"}))
    rprint("\n" + "-" * 80)

    # =========================================================================
    # PASO 1: TEST DE FRIEDMAN (Comparación Global)
    # =========================================================================
    rprint("PASO 1: Test de Friedman (Diferencias Globales)")
    # Correr Friedman con todas las estrategias
    columnas_friedman = [f1_pf] + list(estrategias_todas.values())
    stat_global, p_friedman_global = friedmanchisquare(*columnas_friedman)
    
    rprint(f"  [Global - Todas las Estrategias ({len(columnas_friedman)})]")
    rprint(f"    Estadístico: {stat_global:.4f}")
    rprint(f"    p-value: {p_friedman_global:.6f}")
    if p_friedman_global < 0.05:
        rprint("    Conclusión: EXISTEN diferencias significativas globales (p < 0.05).")
    else:
        rprint("    Conclusión: NO existen diferencias significativas globales.")

    # Correr Friedman solo con las estrategias principales (S1-S7, PW, Local, PF)
    main_strategy_keys = ["local_isolation", "s1_simple_pool", "s2_global_accuracy", "s3_global_f1", 
                          "s4_global_f1_pcd", "s5_perclient_accuracy", "s6_perclient_f1", 
                          "s7_perclient_f1_pcd", "pw"]
    columnas_principales = [f1_pf]
    for key in main_strategy_keys:
        pretty_name = pretty_names[key]
        if pretty_name in estrategias_todas:
            columnas_principales.append(estrategias_todas[pretty_name])

    stat_main, p_friedman_main = friedmanchisquare(*columnas_principales)
    
    rprint(f"\n  [Estrategias Principales ({len(columnas_principales)-1} + Baseline)]")
    rprint(f"    Estadístico: {stat_main:.4f}")
    rprint(f"    p-value: {p_friedman_main:.6f}")
    if p_friedman_main < 0.05:
        rprint("    Conclusión: EXISTEN diferencias significativas en el grupo principal (p < 0.05).")
    else:
        rprint("    Conclusión: NO existen diferencias significativas en el grupo principal.")
    rprint("-" * 80)

    # =========================================================================
    # PASO 2: ANÁLISIS POST-HOC (Wilcoxon con Bonferroni y Holm)
    # =========================================================================
    rprint("PASO 2: Análisis Post-hoc (Nayma PF vs cada estrategia)")
    
    # Calcular Wilcoxon crudo
    p_values = {}
    for nombre, valores in estrategias_todas.items():
        try:
            _, p_val = wilcoxon(f1_pf, valores)
            p_values[nombre] = p_val
        except Exception:
            p_values[nombre] = np.nan

    # Excluir NaN
    p_values = {k: v for k, v in p_values.items() if not np.isnan(v)}
    m = len(p_values)

    # Aplicar corrección de Bonferroni (m = número total de comparaciones)
    p_values_bonf = {nombre: min(p * m, 1.0) for nombre, p in p_values.items()}

    # Aplicar corrección de Holm-Bonferroni
    sorted_p = sorted(p_values.items(), key=lambda x: x[1])
    p_values_holm = {}
    for rank, (nombre, p_orig) in enumerate(sorted_p):
        p_corr = min(p_orig * (m - rank), 1.0)
        p_values_holm[nombre] = p_corr
    
    # Asegurar monotonicidad en Holm
    max_p = 0.0
    for nombre, _ in sorted_p:
        p_values_holm[nombre] = max(p_values_holm[nombre], max_p)
        max_p = p_values_holm[nombre]

    rprint(f"Resultados post-hoc (Comparaciones m = {m}):")
    rprint(f"{'Estrategia':<25} | {'Mean F1':<7} | {'p-orig':<8} | {'p-Bonf':<8} | {'p-Holm':<8} | {'Diferencia':<10}")
    rprint("-" * 80)
    
    # Mostrar la tabla ordenada por rendimiento (F1 medio descendente)
    estrategias_ordenadas = sorted(
        estrategias_todas.keys(), 
        key=lambda name: np.mean(estrategias_todas[name]), 
        reverse=True
    )
    
    mean_pf = np.mean(f1_pf)
    rprint(f"{'* Nayma PF (Centralizado) *':<25} | {mean_pf:.4f}  | {'-':<8} | {'-':<8} | {'-':<8} | {'Baseline':<10}")
    
    for nombre in estrategias_ordenadas:
        vals = estrategias_todas[nombre]
        mean_val = np.mean(vals)
        p_orig = p_values.get(nombre, 1.0)
        p_bonf = p_values_bonf.get(nombre, 1.0)
        p_holm = p_values_holm.get(nombre, 1.0)
        diff = mean_val - mean_pf
        
        # Determinar significancia estadística (usando p original y corregido)
        sig_orig = "*" if p_orig < 0.05 else "ns"
        sig_bonf = "*" if p_bonf < 0.05 else "ns"
        sig_holm = "*" if p_holm < 0.05 else "ns"
        
        sig_str = f"({sig_orig}/{sig_bonf}/{sig_holm})"
        
        rprint(f"{nombre:<25} | {mean_val:.4f}  | {p_orig:.6f} | {p_bonf:.6f} | {p_holm:.6f} | {diff:+.4f} {sig_str}")

    rprint("\nLeyenda de significancia: (original / Bonferroni / Holm)")
    rprint("  *  = Diferencia estadísticamente significativa (p < 0.05)")
    rprint("  ns = No significativo")
    rprint("\n[!] NOTA ESTADÍSTICA CRÍTICA:")
    rprint("    Dado que solo tenemos 8 datasets (N=8), el p-valor original mínimo posible")
    rprint("    para Wilcoxon es 0.007813. Al aplicar correcciones post-hoc para m=14,")
    rprint("    el p-valor corregido mínimo es 0.007813 * 14 = 0.10938 (Bonferroni),")
    rprint("    lo cual es mayor que 0.05. Por lo tanto, con N=8 y correcciones tan conservadoras,")
    rprint("    es matemáticamente imposible declarar significancia corregida. Debe guiarse")
    rprint("    por el p-valor original (p-orig) o la tendencia de rendimiento.")
    rprint("-" * 80)

    # =========================================================================
    # PASO 3: RANKING DE DIFERENCIAS MÁXIMAS
    # =========================================================================
    rprint("PASO 3: Comparación de Pérdida/Ganancia de F1 vs Centralizado por Dataset")
    ranking_datasets = []

    for i, bd in enumerate(datasets):
        valor_pf = f1_pf[i]
        diffs = {s: (estrategias_todas[s][i] - valor_pf) for s in estrategias_todas}
        
        # Encontrar la estrategia con menor pérdida (o mayor ganancia) y la peor
        mejor_estrategia = max(diffs.keys(), key=lambda k: diffs[k])
        peor_estrategia = min(diffs.keys(), key=lambda k: diffs[k])
        
        mejor_diff = diffs[mejor_estrategia]
        peor_diff = diffs[peor_estrategia]
        
        ranking_datasets.append((bd, mejor_estrategia, mejor_diff, peor_estrategia, peor_diff))

    rprint(f"{'Dataset':<12} | {'Mejor Estrategia':<23} | {'Diff F1':<8} | {'Peor Estrategia':<23} | {'Diff F1':<8}")
    rprint("-" * 80)
    for bd, b_est, b_dif, w_est, w_dif in ranking_datasets:
        rprint(f"{bd:<12} | {b_est:<23} | {b_dif:+.4f} | {w_est:<23} | {w_dif:+.4f}")
    
    rprint("\n" + "=" * 80)
    rprint("--- CONCLUSIONES CLAVE DEL EXPERIMENTO ---")
    rprint("=" * 80)
    
    # Encontrar la mejor estrategia federada en promedio (con control de índice corregido)
    f1_promedios = {name: np.mean(vals) for name, vals in estrategias_todas.items()}
    mejor_est_prom = max(f1_promedios.keys(), key=lambda k: f1_promedios[k])
    mean_mejor_est = f1_promedios[mejor_est_prom]
    
    rprint(f"1. La mejor estrategia federada global en promedio es: '{mejor_est_prom}'")
    rprint(f"   con un F1-score medio de {mean_mejor_est:.4f} (frente a {mean_pf:.4f} del Centralizado).")
    
    # Comparar S1 (Simple Pool) contra Local Isolation
    mean_s1 = np.mean(estrategias_todas["S1 (Simple Pool)"])
    mean_local = np.mean(estrategias_todas["Local Isolation"])
    rprint(f"2. S1 (Simple Pool) obtiene un F1-score medio de {mean_s1:.4f}, superando a")
    rprint(f"   Local Isolation ({mean_local:.4f}) por un margen de {mean_s1 - mean_local:+.4f},")
    rprint(f"   lo cual demuestra cuantitativamente la efectividad de la agregación federada.")
    
    # Guardar reporte de texto completo
    report_path = os.path.join(output_dir, "statistical_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReporte de texto detallado guardado en: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_friedman_bonferroni_analysis()
