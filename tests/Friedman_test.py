# import numpy as np
# from scipy.stats import friedmanchisquare

# # Datos de F1 por dataset y estrategia
# f1_pf = [0.945782, 0.954981, 0.965253, 0.954848, 0.982218, 0.823483, 0.952755, 0.968468]
# f1_s1 = [0.904483, 0.933222, 0.918025, 0.954997, 0.96739, 0.811073, 0.926888, 0.79329]
# f1_s2 = [0.885108, 0.933333, 0.921603, 0.945319, 0.968012, 0.739781, 0.926023, 0.79816]
# f1_s3 = [0.899373, 0.933333, 0.917082, 0.946854, 0.964404, 0.749217, 0.922867, 0.781829]
# f1_s4 = [0.904628, 0.922027, 0.923927, 0.958173, 0.969193, 0.779753, 0.919131, 0.77291]
# f1_s5 = [0.916909, 0.910944, 0.921582, 0.948766, 0.964434, 0.772043, 0.921395, 0.788009]
# f1_s6 = [0.903957, 0.944417, 0.919125, 0.946047, 0.968813, 0.769588, 0.922264, 0.783479]
# f1_s7 = [0.923092, 0.922139, 0.917727, 0.950467, 0.969467, 0.782231, 0.920005, 0.760975]
# f1_pw = [0.876685, 0.922139, 0.923843, 0.953607, 0.970025, 0.770655, 0.922127, 0.775966]

# # Test de Friedman: PF vs todas las estrategias
# stat, p = friedmanchisquare(f1_pf, f1_s1, f1_s2, f1_s3, f1_s4, f1_s5, f1_s6, f1_s7, f1_pw)

# print("Estadístico de Friedman:", stat)
# print("p-value:", p)

# if p < 0.05:
#     print("Conclusión: existen diferencias estadísticamente significativas entre PF y al menos una estrategia.")
# else:
#     print("Conclusión: no se detectan diferencias significativas.") 


import numpy as np
from scipy.stats import wilcoxon

# Datos de F1: cada lista comienza con F1_PF seguido de las estrategias
data = {
    "Car":       [0.945782, 0.904483, 0.885108, 0.899373, 0.904628, 0.916909, 0.903957, 0.923092, 0.876685],
    "Iris":      [0.954981, 0.933222, 0.933333, 0.933333, 0.922027, 0.910944, 0.944417, 0.922139, 0.922139],
    "Letter":    [0.965253, 0.918025, 0.921603, 0.917082, 0.923927, 0.921582, 0.919125, 0.917727, 0.923843],
    "Nursery":   [0.954848, 0.954997, 0.945319, 0.946854, 0.958173, 0.948766, 0.946047, 0.950467, 0.953607],
    "Optdigits": [0.982218, 0.96739, 0.968012, 0.964404, 0.969193, 0.964434, 0.968813, 0.969467, 0.970025],
    "Sonar":     [0.823483, 0.811073, 0.739781, 0.749217, 0.779753, 0.772043, 0.769588, 0.782231, 0.770655],
    "Spambase":  [0.952755, 0.926888, 0.926023, 0.922867, 0.919131, 0.921395, 0.922264, 0.920005, 0.922127],
    "Vowel":     [0.968468, 0.79329, 0.79816, 0.781829, 0.77291, 0.788009, 0.783479, 0.760975, 0.775966]
}

# Convertir a arrays por estrategia
f1_pf = [values[0] for values in data.values()]
f1_s1 = [values[1] for values in data.values()]
f1_s2 = [values[2] for values in data.values()]
f1_s3 = [values[3] for values in data.values()]
f1_s4 = [values[4] for values in data.values()]
f1_s5 = [values[5] for values in data.values()]
f1_s6 = [values[6] for values in data.values()]
f1_s7 = [values[7] for values in data.values()]
f1_pw = [values[8] for values in data.values()]

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

# Comparaciones PF vs cada estrategia
p_values = {}
for nombre, valores in estrategias.items():
    stat, p = wilcoxon(f1_pf, valores)
    p_values[nombre] = p

# Corrección de Bonferroni
m = len(p_values)  # número de comparaciones
p_values_corr = {nombre: min(p*m, 1.0) for nombre, p in p_values.items()}

print("Resultados post-hoc (Wilcoxon con Bonferroni):")
for nombre in estrategias.keys():
    print(f"{nombre}: p-value original = {p_values[nombre]:.6f}, p-value corregido = {p_values_corr[nombre]:.6f}")

# import numpy as np
# from scipy.stats import friedmanchisquare

# # Datos de F1 por dataset y estrategia
# f1_pf = [0.945782, 0.954981, 0.965253, 0.954848, 0.982218, 0.823483, 0.952755, 0.968468]
# f1_s1 = [0.904483, 0.933222, 0.918025, 0.954997, 0.96739, 0.811073, 0.926888, 0.79329]
# f1_s2 = [0.885108, 0.933333, 0.921603, 0.945319, 0.968012, 0.739781, 0.926023, 0.79816]
# f1_s3 = [0.899373, 0.933333, 0.917082, 0.946854, 0.964404, 0.749217, 0.922867, 0.781829]
# f1_s4 = [0.904628, 0.922027, 0.923927, 0.958173, 0.969193, 0.779753, 0.919131, 0.77291]
# f1_s5 = [0.916909, 0.910944, 0.921582, 0.948766, 0.964434, 0.772043, 0.921395, 0.788009]
# f1_s6 = [0.903957, 0.944417, 0.919125, 0.946047, 0.968813, 0.769588, 0.922264, 0.783479]
# f1_s7 = [0.923092, 0.922139, 0.917727, 0.950467, 0.969467, 0.782231, 0.920005, 0.760975]
# f1_pw = [0.876685, 0.922139, 0.923843, 0.953607, 0.970025, 0.770655, 0.922127, 0.775966]

# # Test de Friedman: PF vs todas las estrategias
# stat, p = friedmanchisquare(f1_pf, f1_s1, f1_s2, f1_s3, f1_s4, f1_s5, f1_s6, f1_s7, f1_pw)

# print("Estadístico de Friedman:", stat)
# print("p-value:", p)

# if p < 0.05:
#     print("Conclusión: existen diferencias estadísticamente significativas entre PF y al menos una estrategia.")
# else:
#     print("Conclusión: no se detectan diferencias significativas.")


# import numpy as np

# # Datos de F1: cada lista comienza con F1_PF seguido de las estrategias
# data = {
#     "Car":       [0.945782, 0.904483, 0.885108, 0.899373, 0.904628, 0.916909, 0.903957, 0.923092, 0.876685],
#     "Iris":      [0.954981, 0.933222, 0.933333, 0.933333, 0.922027, 0.910944, 0.944417, 0.922139, 0.922139],
#     "Letter":    [0.965253, 0.918025, 0.921603, 0.917082, 0.923927, 0.921582, 0.919125, 0.917727, 0.923843],
#     "Nursery":   [0.954848, 0.954997, 0.945319, 0.946854, 0.958173, 0.948766, 0.946047, 0.950467, 0.953607],
#     "Optdigits": [0.982218, 0.96739, 0.968012, 0.964404, 0.969193, 0.964434, 0.968813, 0.969467, 0.970025],
#     "Sonar":     [0.823483, 0.811073, 0.739781, 0.749217, 0.779753, 0.772043, 0.769588, 0.782231, 0.770655],
#     "Spambase":  [0.952755, 0.926888, 0.926023, 0.922867, 0.919131, 0.921395, 0.922264, 0.920005, 0.922127],
#     "Vowel":     [0.968468, 0.79329, 0.79816, 0.781829, 0.77291, 0.788009, 0.783479, 0.760975, 0.775966]
# }

# max_diff = 0
# max_info = None
# ranking = []

# for bd, values in data.items():
#     f1_pf = values[0]
#     diffs = [abs(f1_pf - v) for v in values[1:]]  # diferencias contra PF
#     local_max = max(diffs)
#     ranking.append((bd, local_max))
#     if local_max > max_diff:
#         max_diff = local_max
#         max_info = (bd, local_max)

# # Ordenar ranking de mayor a menor diferencia
# ranking.sort(key=lambda x: x[1], reverse=True)

# print("Mayor diferencia absoluta global:", max_info)
# print("\nRanking de diferencias máximas por dataset:")
# for bd, diff in ranking:
#     print(f"{bd}: {diff:.6f}")
