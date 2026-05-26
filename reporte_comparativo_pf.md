# Reporte Comparativo: Proactive Forest Centralizado vs. Federado

Este reporte presenta la comparación de los resultados del modelo centralizado de Proactive Forest frente al experimento federado (`results/benchmark_single_fold_results.json`) para los 7 conjuntos de datos coincidentes.

---

## 1. Tabla Comparativa de Eficacia (F1-score & Exactitud)

Para cada conjunto de datos se compara el rendimiento del modelo centralizado frente a la **mejor estrategia federada** obtenida en el experimento. Los valores en negrita indican el mejor rendimiento absoluto.

| Dataset | Centralized F1 | Best Federated F1 | F1 Winner | Centralized Acc | Best Federated Acc | Acc Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | **0.954981** | 0.910662 (s9_weighted_average) | Centralized | **0.956000** | 0.911111 (s9_weighted_average) | Centralized |
| **Car** | 0.945782 | **0.946906 (s4_global_f1_pcd)** | Federated | **0.976626** | 0.971098 (s4_global_f1_pcd) | Centralized |
| **Nursery** | 0.954848 | **0.959833 (s3_global_f1)** | Federated | **0.995911** | 0.975823 (s2_global_accuracy) | Centralized |
| **Vowel** | **0.968468** | 0.858628 (s1_simple_pool) | Centralized | **0.971919** | 0.858586 (s1_simple_pool) | Centralized |
| **Optdigits** | **0.982218** | 0.966097 (s1_simple_pool) | Centralized | **0.983236** | 0.966192 (s1_simple_pool) | Centralized |
| **Sonar** | 0.823483 | **0.918181 (s6_perclient_f1)** | Federated | 0.848299 | **0.920635 (s6_perclient_f1)** | Federated |
| **Spambase** | **0.952755** | 0.935041 (s7_perclient_f1_pcd) | Centralized | **0.953880** | 0.938539 (s7_perclient_f1_pcd) | Centralized |

*Nota: Para el modelo federado, la estrategia con mejor resultado para esa métrica en particular se indica entre paréntesis.*

---

## 2. Tabla Comparativa de Diversidad (PCD)

Comparación de la diversidad correctiva porcentual (PCD) entre el modelo centralizado y la estrategia federada con mayor diversidad.

| Dataset | Centralized PCD | Best Federated PCD | Diversity Winner |
| :--- | :---: | :---: | :---: |
| **Iris** | 0.112000 | **0.333333 (s1_simple_pool)** | Federated |
| **Car** | 0.309630 | **0.369942 (s1_simple_pool)** | Federated |
| **Nursery** | 0.396870 | **0.438272 (pw)** | Federated |
| **Vowel** | 0.937000 | **0.979798 (s1_simple_pool)** | Federated |
| **Optdigits** | **0.583070** | 0.551601 (s9_median) | Centralized |
| **Sonar** | 0.907250 | **1.000000 (s1_simple_pool)** | Federated |
| **Spambase** | 0.330740 | **0.360087 (pw)** | Federated |

---

## 3. Análisis Estadístico de Eficacia (F1-score)

Resultados de las pruebas estadísticas aplicadas sobre los valores de F1-score:

### A. Test de Friedman (Diferencias Globales)
* **Estadístico**: 47.4135
* **p-value**: 0.000008
* **Conclusión**: **Existen diferencias estadísticas significativas** globales entre el modelo centralizado y el conjunto de estrategias federadas (p < 0.05).

### B. Pruebas Post-hoc de Wilcoxon (Centralizado vs. Cada Estrategia)

| Estrategia | Media F1 | p-value (Wilcoxon) | Diferencia vs. Base | Diferencia Significativa (p < 0.05) |
| :--- | :---: | :---: | :---: | :---: |
| **Centralized PF (Nayma)** | 0.940362 | - | - | *Línea de base* |
| **s6_perclient_f1** | 0.914572 | 0.296875 | -0.025790 | **No** (Desempeño comparable) |
| **s3_global_f1** | 0.908175 | 0.218750 | -0.032187 | **No** (Desempeño comparable) |
| **s5_perclient_accuracy** | 0.906819 | 0.156250 | -0.033543 | **No** (Desempeño comparable) |
| **s2_global_accuracy** | 0.902943 | 0.218750 | -0.037419 | **No** (Desempeño comparable) |
| **s7_perclient_f1_pcd** | 0.902673 | 0.156250 | -0.037689 | **No** (Desempeño comparable) |
| **s1_simple_pool** | 0.901846 | 0.015625 | -0.038517 | **Sí** (Rendimiento inferior) |
| **s4_global_f1_pcd** | 0.898981 | 0.046875 | -0.041381 | **Sí** (Rendimiento inferior) |
| **pw (Progressive Windows)** | 0.865113 | 0.015625 | -0.075249 | **Sí** (Rendimiento inferior) |
| **s9_median** | 0.829128 | 0.015625 | -0.111234 | **Sí** (Rendimiento inferior) |
| **s9_proactive_pcd** | 0.827864 | 0.015625 | -0.112498 | **Sí** (Rendimiento inferior) |
| **s9_consensus** | 0.827856 | 0.015625 | -0.112506 | **Sí** (Rendimiento inferior) |
| **s9_simple_mean** | 0.827856 | 0.015625 | -0.112506 | **Sí** (Rendimiento inferior) |
| **s9_weighted_average** | 0.827856 | 0.015625 | -0.112506 | **Sí** (Rendimiento inferior) |
