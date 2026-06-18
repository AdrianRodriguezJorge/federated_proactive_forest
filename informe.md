# Optimización de Hiperparámetros del Framework Federado

## 1. Objetivo

Se llevó a cabo un proceso de optimización bayesiana unificada con el fin de encontrar una única configuración de hiperparámetros robusta que maximizara el accuracy medio de los clientes de forma generalizable, evitando el sobreajuste a un dataset o estrategia de agregación particular.

## 2. Metodología

Se empleó el algoritmo **Tree-Structured Parzen Estimator (TPE)** implementado en Optuna, con la opción multivariante activada para explotar las dependencias entre hiperparámetros. Se fijó una semilla de `42` para garantizar la reproducibilidad de los resultados.

### 2.1. Función Objetivo

Cada ensayo (trial) de la optimización evaluó el conjunto de hiperparámetros propuesto sobre todas las **12 estrategias de agregación** (S1–S7 y las 5 variantes de S8) y **4 datasets representativos** (Sonar, Vowel, Spambase y Nursery), seleccionados por cubrir un espectro amplio de características: datasets numéricos y categóricos, binarios y multiclase, de tamaño reducido y elevado.

La métrica optimizada fue el **accuracy medio** de los clientes federados. El score final de cada ensayo correspondió a la **media aritmética del accuracy** obtenido en todas las combinaciones estrategia–dataset evaluadas.

### 2.2. Configuración del Experimento

Las evaluaciones se realizaron bajo una distribución de datos **IID** con **3 clientes** federados y un máximo de 150 estimadores como límite superior de seguridad.

## 3. Espacio de Búsqueda

Se definieron los siguientes hiperparámetros a optimizar, agrupados según su ámbito de aplicación:

### 3.1. Hiperparámetros del Modelo Local

| Hiperparámetro | Tipo | Rango / Valores | Descripción |
|:---|:---:|:---:|:---|
| Umbral de convergencia local (`local_convergence_threshold`) | Categórico | {0.001, 0.002, 0.005} | Umbral de varianza mínima en el accuracy local para activar la parada temprana del cliente. |

### 3.2. Hiperparámetros de Agregación en el Servidor

| Hiperparámetro | Tipo | Rango / Valores | Descripción |
|:---|:---:|:---:|:---|
| Tamaño máximo del pool global (`max_trees`) | Entero | [50, 150], paso 20 | Límite superior de árboles permitidos en el pool global agregado. |
| Umbral de convergencia global (`global_convergence_threshold`) | Categórico | {0.001, 0.002, 0.005} | Umbral de mejora mínima entre episodios consecutivos del servidor para continuar la agregación. |
| Peso de F1 en selección híbrida (`f1_weight`) | Float | [0.0, 1.0], paso 0.1 | Balance entre F1-score y diversidad PCD en las estrategias S4 y S7. El peso de PCD es el complemento ($1 - w_{F1}$). |
| Tamaño de episodio global (`global_episode_size`) | Categórico | {3, 5, 6, 9, 10} | Número de árboles que se añaden al pool global en cada paso de evaluación antes de verificar la convergencia. |
| Árboles por cliente por episodio (`trees_per_client_per_episode`) | Categórico | {1, 2, 3} | Número de árboles que cada cliente contribuye por episodio en las estrategias con selección per-client. |

### 3.3. Hiperparámetros de la Estrategia S8 (Global Attribute Roulette)

| Hiperparámetro | Tipo | Rango / Valores | Descripción |
|:---|:---:|:---:|:---|
| Tamaño de ventana local (`window_size`) | Categórico | {5, 10, 15} | Número de árboles que cada cliente entrena localmente antes de recalcular y transmitir su vector de importancia de atributos al servidor. |
| Máximo de rondas federadas (`max_rounds`) | Categórico | {10, 15, 30} | Límite superior del número de rondas de comunicación entre clientes y servidor. |
| Peso del vector de atributos local (`local_roulette_weight`) | Float | [0.0, 1.0], paso 0.1 | Ponderación del vector de importancia local del cliente frente al vector global del servidor al actualizar las probabilidades de selección de atributos. |

### 3.4. Hiperparámetros de Predicción Híbrida

| Hiperparámetro | Tipo | Rango / Valores | Descripción |
|:---|:---:|:---:|:---|
| Peso del modelo local (`local_weight`) | Float | [0.0, 1.0], paso 0.1 | Contribución del bosque local del cliente en la predicción híbrida. El peso global es el complemento ($1 - w_{local}$). |
| Votación ponderada (`use_weighted`) | Categórico | {True, False} | Activa o desactiva la ponderación de los votos de los árboles según el rendimiento del cliente. |

### 3.5. Hiperparámetros Fijos

Durante todo el proceso de optimización, se mantuvieron constantes los siguientes parámetros para servir como base de control:

| Hiperparámetro | Valor fijo | Descripción |
|:---|:---:|:---|
| Parámetro de diversidad del bosque (`alpha`) | `0.1` | Controla la penalización por redundancia en el crecimiento local del bosque. |
| Criterio de división del nodo (`split_criterion`) | `"entropy"` | Criterio de ganancia de información utilizado para la construcción de cada árbol de decisión. |
| Esquema de votación del ensamble (`voting`) | `"soft"` | Votación basada en el promedio ponderado de las probabilidades de clase predichas. |
| Parada progresiva activa (`use_progressive_stopping`) | `True` | Habilita la comprobación dinámica de convergencia local y global. |
| Tamaño de episodio local (`local_episode_size`) | `5` | Frecuencia de árboles evaluados localmente por el cliente para el cálculo de convergencia. |
| Mínimo de episodios evaluados (`min_episodes`) | `4` | Número mínimo de comprobaciones globales obligatorias antes de permitir parada temprana. |
| Mínimo de rondas federadas (`min_rounds`) | `4` | Número mínimo de rondas de comunicación obligatorias en la estrategia S8 antes de finalizar. |

## 4. Resultados de la Optimización

Se completaron un total de **63 ensayos** (de 68 lanzados; 2 fallidos y 3 incompletos). A continuación se presenta la configuración óptima identificada, seguida del análisis por hiperparámetro.

### 4.1. Configuración Óptima

La optimización bayesiana identificó la siguiente combinación como la configuración con mayor accuracy medio sobre el conjunto de estrategias y datasets evaluados:

| Hiperparámetro | Valor óptimo |
|:---|:---:|
| Peso del modelo local (`local_weight`) | 0.4 |
| Tamaño máximo del pool global (`max_trees`) | 110 |
| Umbral de convergencia local (`local_convergence_threshold`) | 0.002 |
| Umbral de convergencia global (`global_convergence_threshold`) | 0.002 |
| Peso de F1 en selección híbrida (`f1_weight`) | 0.5 |
| Peso del vector de atributos local (`local_roulette_weight`) | 0.6 |
| Tamaño de episodio global (`global_episode_size`) | 10 |
| Árboles por cliente por episodio (`trees_per_client_per_episode`) | 3 |
| Tamaño de ventana local (`window_size`) | 10 |
| Máximo de rondas federadas (`max_rounds`) | 15 |
| Votación ponderada (`use_weighted`) | Sí |

Esta configuración fue la utilizada en el experimento comparativo final.

### 4.2. Análisis por Hiperparámetro

#### Votación ponderada (`use_weighted`)

| Valor  | Ensayos |
|:------:|:-------:|
| True   | 50      |
| False  | 13      |

La totalidad de los mejores ensayos identificados en la exploración utilizaron votación ponderada, demostrando ser una opción de configuración dominante. Esto confirma que ponderar el peso de los estimadores por el rendimiento y representatividad local mejora la robustez colectiva.

#### Tamaño de episodio global (`global_episode_size`)

| Valor | Ensayos |
|:-----:|:-------:|
| 10    | 23      |
| 3     | 26      |
| 9     | 8       |
| 6     | 3       |
| 5     | 3       |

Los episodios más amplios (como 10) mostraron mejores resultados de forma consistente. Esto indica que evaluar lotes de mayor tamaño en el servidor estabiliza las métricas, reduciendo el ruido en las comprobaciones de convergencia progresiva y evitando paradas tempranas prematuras.

#### Árboles por cliente por episodio (`trees_per_client_per_episode`)

| Valor | Ensayos |
|:-----:|:-------:|
| 3     | 31      |
| 1     | 29      |
| 2     | 3       |

El optimizador mostró una marcada preferencia por contribuciones de 3 árboles por etapa progresiva local, lo que proporciona estimadores suficientes para el control de convergencia a nivel cliente.

#### Tamaño de ventana local (`window_size`) (Estrategia S8)

| Valor | Ensayos |
|:-----:|:-------:|
| 10    | 41      |
| 5     | 11      |
| 15    | 11      |

La búsqueda se concentró predominantemente en ventanas de 10 árboles, aunque el comportamiento global se mantuvo sumamente estable ante las variaciones de frecuencia de comunicación, indicando una gran tolerancia a esta granularidad.

#### Máximo de rondas federadas (`max_rounds`) (Estrategia S8)

| Valor | Ensayos |
|:-----:|:-------:|
| 15    | 41      |
| 30    | 11      |
| 10    | 11      |

El desempeño de la ruleta de atributos se estabiliza de manera rápida en las primeras fases de la comunicación federada, por lo que rondas moderadas (como 15) resultaron óptimas y computacionalmente eficientes.

#### Tamaño máximo del pool global (`max_trees`)

| Pool global | Ensayos |
|:-----------:|:-------:|
| 130         | 21      |
| 110         | 12      |
| 90          | 10      |
| 150         | 10      |
| 70          | 6       |
| 50          | 4       |

El rango de búsqueda preferido se ubicó entre 110 y 130 árboles globales agregados. Tamaños inferiores restringen la capacidad de representación y generalización del ensamble, mientras que tamaños superiores no aportaron ventajas significativas, debido a la redundancia de los estimadores añadidos tardíamente.

#### Umbrales de convergencia

**Umbral de convergencia local (`local_convergence_threshold`):**

| Valor | Ensayos |
|:-----:|:-------:|
| 0.002 | 31      |
| 0.001 | 21      |
| 0.005 | 11      |

**Umbral de convergencia global (`global_convergence_threshold`):**

| Valor | Ensayos |
|:-----:|:-------:|
| 0.001 | 31      |
| 0.005 | 25      |
| 0.002 | 7       |

La exigencia de umbrales estrictos (0.001 y 0.002) resultó prioritaria. El uso de criterios excesivamente permisivos (como 0.005) tiende a detener el crecimiento progresivo de forma prematura antes de que el modelo converja adecuadamente.

#### Peso del modelo local (`local_weight`)

Los ensayos más óptimos de la búsqueda bayesiana se agruparon en el rango de 0.4 a 0.5 para el peso del modelo local, lo que valida la importancia de una predicción híbrida equilibrada frente a esquemas puramente locales o colectivos.

#### Peso del vector de atributos local (`local_roulette_weight`) (S8)

El análisis de correlación de Spearman arrojó una relación positiva y estadísticamente significativa entre este peso y el desempeño del modelo, evidenciando que priorizar la contribución local del cliente al actualizar las probabilidades de la ruleta de atributos mejora el desempeño global en la agregación distribuida.
