# Análisis de Hiperparámetros de las Estrategias de Agregación Federada (Estandarizado)

Este documento detalla y clasifica de manera precisa todos los hiperparámetros asociados a las 12 estrategias de agregación federada implementadas en el repositorio de `federated_proactive_forest` utilizando la nomenclatura oficial y estandarizada del codebase.

---

## 1. Hiperparámetros Comunes Globales
Estos parámetros definen el comportamiento de entrenamiento del modelo base local (**Proactive Forest**) o la combinación para la predicción en el cliente. Están activos en las 12 estrategias analizadas:

*   **`alpha_pf`:**
    *   **Tipo de dato:** `float`
    *   **Rol:** Parámetro de proactividad / tolerancia de Proactive Forest local. Regula el nivel de desvío aleatorio en la selección de características para cada división (split) de los árboles de decisión locales. Un valor bajo promueve mayor aleatoriedad, mientras que un valor alto favorece la ganancia de información estricta.
    *   **Valor por defecto:** `0.1` (definido en `ModelConfig.alpha_pf` y configuraciones `configs/experiments/exp_*.yaml`).
    *   **Rango de optimización:** `[0.05, 0.9]` con paso de `0.05` (excluido en la optimización unificada actual para mantener el baseline propuesto).
*   **`local_convergence_threshold`:**
    *   **Tipo de dato:** `float`
    *   **Rol:** Umbral de convergencia local para el entrenamiento progresivo de los bosques en los clientes. Determina la ganancia mínima en precisión necesaria en episodios sucesivos para continuar añadiendo estimadores al bosque local de cada cliente en cada ronda.
    *   **Valor por defecto:** `0.002` (en `flex_train_pf.py` y `flex_s8_progressive.py`).
    *   **Rango de optimización:** `[0.001, 0.002, 0.005]` (discreto y acotado).
*   **`local_weight` (y `global_weight` complementario):**
    *   **Tipo de dato:** `float`
    *   **Rol:** Peso ponderado asignado al clasificador local frente al global durante el voto híbrido en inferencia (`HybridForest` / `HybridPredictor`):
        $$\text{score} = \text{local\_weight} \cdot \text{votos\_locales} + \text{global\_weight} \cdot \text{votos\_globales}$$
    *   **Valor por defecto:** `local_weight: 0.4` / `global_weight: 0.6` (definido en `PredictionConfig`).
    *   **Rango de optimización:** `local_weight` en `[0.0, 1.0]` con paso de `0.1`, con `global_weight = 1.0 - local_weight`.
*   **`use_weighted`:**
    *   **Tipo de dato:** `bool`
    *   **Rol:** Determina si se aplica la ponderación específica por origen (`local_weight`) o si cada árbol en el bosque híbrido tiene el mismo peso (`1.0`), desactivando la distinción de origen.
    *   **Valor por defecto:** `True`.
    *   **Rango de optimización:** `[True, False]`.
*   **`n_estimators`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Especifica el tamaño base de estimadores construidos localmente en el cliente para S1-S7. En la estrategia S8, representa el número total final de árboles en el bosque local ($max\_rounds \times window\_size$).
    *   **Valor por defecto:** `100` (definido en `ModelConfig`).
    *   **Rango de optimización:** Fijo (no se optimiza).

---

## 2. Hiperparámetros Comunes por Subgrupos

### 2.1. Comunes a las Estrategias de Progressive Forest (S2 - S7)
Estrategias que rankean y seleccionan progresivamente árboles en el servidor evaluando sobre un dataset de validación central y aplicando parada temprana:

*   **`max_trees`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Límite superior o número máximo de árboles permitidos en el bosque global agregado resultante.
    *   **Valor por defecto:** `100` (definido en `GlobalProgressiveStrategy.T_MAX` y `PerClientProgressiveStrategy.T_MAX`). Si se omite, el orquestador usa `n_estimators`.
    *   **Rango de optimización:** `[50, 150]` con paso de `20`.
*   **`global_convergence_threshold`:**
    *   **Tipo de dato:** `float`
    *   **Rol:** Umbral de parada temprana en el proceso de agregación progresiva. La selección progresiva en el servidor finaliza si la mejora de precisión del bosque global sobre el dataset de validación central entre dos episodios consecutivos es menor que este valor durante 2 episodios seguidos.
    *   **Valor por defecto:** `0.002` (en `AggregationConfig.global_convergence_threshold`).
    *   **Rango de optimización:** `[0.001, 0.002, 0.005]`.
*   **`min_episodes`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Cantidad mínima de episodios de selección global que deben ejecutarse obligatoriamente antes de permitir que actúe la parada temprana.
    *   **Valor por defecto:** `5`.
    *   **Rango de optimización:** Fijo.
*   **`max_rounds`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Límite máximo de rondas de comunicación del experimento federado.
    *   **Valor por defecto:** `20`.
    *   **Rango de optimización:** Fijo para S1-S7.

### 2.2. Comunes a las Estrategias Progresivas Globales (S2 - S4)
*   **`global_episode_size`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Número de árboles incorporados y evaluados en bloque en cada episodio de la selección progresiva desde la lista única global jerarquizada.
    *   **Valor por defecto:** `5`.
    *   **Rango de optimización:** Evaluado a través de las 5 parejas acopladas con `trees_per_client_per_episode` para garantizar comparaciones equitativas.

### 2.3. Comunes a las Estrategias Progresivas por Cliente (S5 - S7)
*   **`trees_per_client_per_episode`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Número de árboles que aporta individualmente cada cliente en cada episodio de la selección progresiva. El tamaño efectivo del episodio evaluado en el servidor se calcula como: $episode\_size = trees\_per\_client\_per\_episode \times n\_clients$.
    *   **Valor por defecto:** `1` (en `flex_aggregate_pf.py`).

### 2.4. Comunes a las Variantes de Global Attribute Roulette (S8)
*   **`local_roulette_weight`:**
    *   **Tipo de dato:** `float`
    *   **Rol:** Factor de interpolación lineal para fusionar la ruleta local de características entrenada por el cliente con la ruleta agregada global:
        $$\text{Roulette}_{new} = \text{local\_roulette\_weight} \cdot \text{Roulette}_{local} + (1.0 - \text{local\_roulette\_weight}) \cdot \text{Roulette}_{global}$$
    *   **Valor por defecto:** `0.1`.
    *   **Rango de optimización:** `[0.0, 1.0]` con paso de `0.1`.
*   **`window_size`:**
    *   **Tipo de dato:** `int`
    *   **Rol:** Número de árboles entrenados de manera local en el cliente en cada ronda de comunicación de S8 (tamaño del lote o ventana local).
    *   **Valor por defecto:** `5`.
    *   **Rango de optimización:** Acoplado directamente con `max_rounds` (`5_30`, `10_15`, `15_10`).
*   **`max_rounds` (S8):**
    *   **Tipo de dato:** `int`
    *   **Rol:** Límite máximo de rondas de comunicación para el intercambio de vectores roulette.
    *   **Valor por defecto:** `20`.

---

## 3. Hiperparámetros Específicos por Estrategia o Variante

*   **`f1_weight` y `pcd_weight` (Específicos para S4 y S7):**
    *   **Tipo de dato:** `float`
    *   **Rol:** Factores de ponderación en la función de puntuación multiobjetivo para el ranking de árboles:
        $$\text{score} = \text{f1\_weight} \cdot \text{macro\_f1} + \text{pcd\_weight} \cdot \text{pcd}$$
        Si se especifica `f1_weight`, `pcd_weight` se calcula automáticamente como $1.0 - f1\_weight$.
    *   **Valor por defecto:** `f1_weight: 0.5` / `pcd_weight: 0.5`.
    *   **Rango de optimización:** `f1_weight` en `[0.0, 1.0]` con paso de `0.1`.
*   **`variant` (Específico para variantes de Roulette S8):**
    *   **Tipo de dato:** Categorical (`str`)
    *   **Rol:** Determina la fórmula de agregación empleada por el servidor para unificar las ruletas de los clientes en un único vector global:
        *   `S8_MEAN`: Promedio simple de todos los vectores de ruleta.
        *   `S8_MEDIAN`: Mediana de probabilidad componente por componente.
        *   `S8_CONSENSUS`: Promedio ponderado basado en el rendimiento local Macro-F1 de cada cliente evaluado en el servidor.
        *   `S8_WEIGHTED`: Promedio ponderado por la cantidad de muestras de entrenamiento del cliente.
        *   `S8_PROACTIVE_PCD`: Promedio ponderado según la métrica de diversidad PCD.
    *   **Valor por defecto:** Según la variante elegida (ej. `S8_CONSENSUS` para `s8_consensus`).

---

## 4. Tabla de Referencia Resumida

A continuación se muestra la matriz comparativa de las 12 estrategias y los hiperparámetros que se encuentran activos e influyen en su ejecución utilizando la nueva nomenclatura:

| Estrategia | alpha_pf | local_conv_threshold | local_weight | use_weighted | max_trees | global_conv_threshold | global_episode_size | f1_weight / pcd_weight | local_roulette_weight | window_size | max_rounds | variante |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S1: Simple Pool** | 0.1 | 0.002 | 0.4 | True | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 20 | *N/A* |
| **S2: Global Accuracy** | 0.1 | 0.002 | 0.4 | True | 100 | 0.002 | 5 | *N/A* | *N/A* | *N/A* | 20 | *N/A* |
| **S3: Global Macro-F1** | 0.1 | 0.002 | 0.4 | True | 100 | 0.002 | 5 | *N/A* | *N/A* | *N/A* | 20 | *N/A* |
| **S4: Global F1 + PCD** | 0.1 | 0.002 | 0.4 | True | 100 | 0.002 | 5 | 0.5 / 0.5 | *N/A* | *N/A* | 20 | *N/A* |
| **S5: Per-Client Accuracy** | 0.1 | 0.002 | 0.4 | True | 100 | 0.002 | $1 \times W$ | *N/A* | *N/A* | *N/A* | 20 | *N/A* |
| **S6: Per-Client Macro-F1** | 0.1 | 0.002 | 0.4 | True | 100 | 0.002 | $1 \times W$ | *N/A* | *N/A* | *N/A* | 20 | *N/A* |
| **S7: Per-Client F1 + PCD** | 0.1 | 0.002 | 0.4 | True | 100 | 0.002 | $1 \times W$ | 0.5 / 0.5 | *N/A* | *N/A* | 20 | *N/A* |
| **S8_MEAN** | 0.1 | 0.002 | 0.4 | True | *N/A* | *N/A* | *N/A* | *N/A* | 0.1 | 5 | 20 | `S8_MEAN` |
| **S8_MEDIAN** | 0.1 | 0.002 | 0.4 | True | *N/A* | *N/A* | *N/A* | *N/A* | 0.1 | 5 | 20 | `S8_MEDIAN` |
| **S8_CONSENSUS** | 0.1 | 0.002 | 0.4 | True | *N/A* | *N/A* | *N/A* | *N/A* | 0.1 | 5 | 20 | `S8_CONSENSUS` |
| **S8_WEIGHTED** | 0.1 | 0.002 | 0.4 | True | *N/A* | *N/A* | *N/A* | *N/A* | 0.1 | 5 | 20 | `S8_WEIGHTED` |
| **S8_PROACTIVE_PCD** | 0.1 | 0.002 | 0.4 | True | *N/A* | *N/A* | *N/A* | *N/A* | 0.1 | 5 | 20 | `S8_PROACTIVE_PCD` |

*Nota: $W$ representa el número de clientes participantes en la simulación federada.*
