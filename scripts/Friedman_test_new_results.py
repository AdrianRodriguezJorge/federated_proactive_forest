import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

def parse_markdown_tables(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    tables = {}
    current_table_name = None
    table_lines = []
    
    def parse_lines_to_df(t_lines):
        header_line = t_lines[0]
        headers = [h.strip().replace("**", "") for h in header_line.split("|")[1:-1]]
        
        data = []
        for line in t_lines[2:]:  # skip header and separator
            row = [cell.strip().replace("**", "") for cell in line.split("|")[1:-1]]
            if row:
                data.append(row)
                
        df = pd.DataFrame(data, columns=headers)
        df = df.set_index(headers[0])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    for line in lines:
        line_str = line.strip()
        if line_str.startswith("**Tabla"):
            if current_table_name and table_lines:
                tables[current_table_name] = parse_lines_to_df(table_lines)
                table_lines = []
            if "Exactitud" in line_str or "accuracy" in line_str.lower():
                current_table_name = "Accuracy"
            elif "Macro-F1" in line_str or "f1" in line_str.lower():
                current_table_name = "F1"
            elif "PCD" in line_str:
                current_table_name = "PCD"
            else:
                current_table_name = line_str
        elif line_str.startswith("|"):
            table_lines.append(line_str)
            
    if current_table_name and table_lines:
        tables[current_table_name] = parse_lines_to_df(table_lines)
        
    return tables

def analyze_metric(metric_name, df):
    baseline_col = "PF Cent."
    if baseline_col not in df.columns:
        raise ValueError(f"La columna de baseline '{baseline_col}' no se encuentra en el DataFrame.")
        
    strategies = [col for col in df.columns if col != baseline_col]
    columns_friedman = [df[baseline_col].values] + [df[strat].values for strat in strategies]
    
    print("=" * 80)
    print(f"--- ANÁLISIS ESTADÍSTICO PARA LA MÉTRICA: {metric_name} ---")
    print("=" * 80)
    print(f"Datasets analizados ({len(df.index)}): {', '.join(df.index)}\n")
    
    # Friedman Test
    stat_global, p_friedman_global = friedmanchisquare(*columns_friedman)
    print("PASO 1: Test de Friedman (Diferencias Globales)")
    print(f"  Estadístico: {stat_global:.4f}")
    print(f"  p-value: {p_friedman_global:.6e}")
    if p_friedman_global < 0.05:
        print("  Conclusión: EXISTEN diferencias significativas globales (p < 0.05).")
    else:
        print("  Conclusión: NO existen diferencias significativas globales.")
    print("-" * 80)
    
    # Wilcoxon Tests
    print("PASO 2: Análisis Post-hoc (Centralizado vs Cada Estrategia)")
    print(f"{'Estrategia':<25} | {'Mean ' + metric_name:<12} | {'p-value':<12} | {'Diferencia':<10} | {'Significativa':<12}")
    print("-" * 80)
    
    mean_cent = df[baseline_col].mean()
    print(f"{'* ' + baseline_col + ' *':<25} | {mean_cent:.6f} | {'-':<12} | {'-':<10} | Baseline")
    
    # Decidir orden de las estrategias basándose en la métrica.
    sorted_strategies = sorted(strategies, key=lambda s: df[s].mean(), reverse=True)
    
    for strat in sorted_strategies:
        mean_strat = df[strat].mean()
        diff = mean_strat - mean_cent
        try:
            _, p_w = wilcoxon(df[baseline_col].values, df[strat].values)
            sig = "Sí (p<0.05)" if p_w < 0.05 else "No"
            p_w_str = f"{p_w:.6f}"
        except Exception as e:
            p_w_str = "Error/Empate"
            sig = "N/A"
        print(f"{strat:<25} | {mean_strat:.6f} | {p_w_str:<12} | {diff:+.6f} | {sig:<12}")
    print("=" * 80 + "\n")

def main():
    possible_paths = [
        "tablas.md",
        "../tablas.md",
        os.path.join(os.path.dirname(__file__), "tablas.md") if "__file__" in locals() or "__file__" in globals() else "",
        os.path.join(os.path.dirname(__file__), "..", "tablas.md") if "__file__" in locals() or "__file__" in globals() else "",
    ]
    
    filepath = None
    for path in possible_paths:
        if path and os.path.exists(path):
            filepath = path
            break
            
    if not filepath:
        print("Error: No se encontró 'tablas.md' en los directorios esperados.")
        return
        
    print(f"Cargando datos desde: {filepath}\n")
    try:
        tables = parse_markdown_tables(filepath)
    except Exception as e:
        print(f"Error al analizar el archivo markdown: {e}")
        return
        
    for name in ["Accuracy", "F1", "PCD"]:
        if name in tables:
            analyze_metric(name, tables[name])
        else:
            print(f"Advertencia: No se encontró la tabla para la métrica '{name}'.")

if __name__ == "__main__":
    main()
