import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import os

# =============================================================================
# CONFIGURACIÓN Y CARGA DE DATOS
# =============================================================================

# Definimos la ruta al archivo de resultados
csv_path = os.path.join("results", "comparison_vs_nayma_good", "comparison_table.csv")

# Cargamos los datos usando pandas
df = pd.read_csv(csv_path)

# Extraemos los nombres de las bases de datos (columna 'BD')
datasets = df['BD'].tolist()

# Extraemos las métricas F1 para cada estrategia
# El CSV tiene columnas como F1_PF, F1_S1, ..., F1_PW
f1_pf = df['F1_PF'].values
f1_s1 = df['F1_S1'].values
f1_s2 = df['F1_S2'].values
f1_s3 = df['F1_S3'].values
f1_s4 = df['F1_S4'].values
f1_s5 = df['F1_S5'].values
f1_s6 = df['F1_S6'].values
f1_s7 = df['F1_S7'].values
f1_pw = df['F1_PW'].values

# Diccionario organizado para facilitar las comparaciones post-hoc y el ranking
estrategias = {
    "S1": f1_s1,
    "S2": f1_s2,
    "S3": f1_s3,
    "S4": f1_s4,
    "S5": f1_s5,
    "S6": f1_s6,
    "S7": f1_s7,
    "PW": f1_pw
}

print(f"--- Análisis Estadístico de Resultados (Datasets: {len(datasets)}) ---")
print(f"Datos cargados desde: {csv_path}\n")

# =============================================================================
# 1. TEST DE FRIEDMAN (Diferencias Globales)
# =============================================================================
# El test de Friedman determina si existen diferencias significativas entre 
# todos los grupos comparados simultáneamente.

print("PASO 1: Test de Friedman (Comparación Global)")
stat, p_friedman = friedmanchisquare(f1_pf, f1_s1, f1_s2, f1_s3, f1_s4, f1_s5, f1_s6, f1_s7, f1_pw)

print(f"Estadístico de Friedman: {stat:.4f}")
print(f"p-value: {p_friedman:.6f}")

if p_friedman < 0.05:
    print("Conclusión: EXISTEN diferencias estadísticamente significativas entre al menos dos estrategias.")
else:
    print("Conclusión: No se detectan diferencias significativas globales.")
print("-" * 50)

# =============================================================================
# 2. ANÁLISIS POST-HOC (Wilcoxon con Bonferroni)
# =============================================================================
# Si el test de Friedman es significativo, comparamos PF individualmente 
# contra cada una de las otras estrategias.

print("PASO 2: Análisis Post-hoc (PF vs cada estrategia)")
p_values = {}
for nombre, valores in estrategias.items():
    # Test de Wilcoxon: prueba no paramétrica para muestras relacionadas
    _, p_val = wilcoxon(f1_pf, valores)
    p_values[nombre] = p_val

# Aplicamos la corrección de Bonferroni (multiplicar p por el número de comparaciones)
m = len(p_values)  # Número de comparaciones (8 estrategias contra PF)
p_values_corr = {nombre: min(p * m, 1.0) for nombre, p in p_values.items()}

print(f"Resultados post-hoc (Corrección de Bonferroni para m={m}):")
for nombre in estrategias.keys():
    sig = "(*)" if p_values_corr[nombre] < 0.05 else "(ns)"
    print(f"  PF vs {nombre}: p-original = {p_values[nombre]:.6f}, p-corregido = {p_values_corr[nombre]:.6f} {sig}")
print("-" * 50)

# =============================================================================
# 3. RANKING DE DIFERENCIAS MÁXIMAS
# =============================================================================
# Calculamos la diferencia absoluta máxima de F1 entre PF y cualquier otra 
# estrategia para cada dataset, para identificar dónde el impacto es mayor.

print("PASO 3: Ranking de Diferencias Máximas contra PF")
ranking = []

for i, bd in enumerate(datasets):
    valor_pf = f1_pf[i]
    # Diferencias de todas las estrategias contra PF en este dataset
    diffs = [abs(valor_pf - estrategias[s][i]) for s in estrategias]
    max_diff_dataset = max(diffs)
    ranking.append((bd, max_diff_dataset))

# Ordenamos el ranking de mayor diferencia a menor
ranking.sort(key=lambda x: x[1], reverse=True)

print("Impacto de la estrategia (Diferencia máxima F1 vs PF) por dataset:")
for bd, diff in ranking:
    print(f"  {bd:12}: {diff:.6f}")

# Resumen final de la mayor diferencia encontrada
max_total_bd, max_total_val = ranking[0]
print(f"\nMayor diferencia absoluta global encontrada en '{max_total_bd}': {max_total_val:.6f}")
print("=" * 50)
