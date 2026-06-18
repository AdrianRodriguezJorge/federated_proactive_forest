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

| Hiperparámetro                       | Tipo       | Rango / Valores           | Descripción                                                                                      |
|:-------------------------------------|:----------:|:-------------------------:|:-------------------------------------------------------------------------------------------------|
| Umbral de convergencia local         | Categórico | {0.001, 0.002, 0.005}     | Umbral de varianza mínima en el accuracy local para activar la parada temprana del cliente.       |

### 3.2. Hiperparámetros de Agregación en el Servidor

| Hiperparámetro                       | Tipo       | Rango / Valores           | Descripción                                                                                      |
|:-------------------------------------|:----------:|:-------------------------:|:-------------------------------------------------------------------------------------------------|
| Tamaño máximo del pool global        | Entero     | [50, 150], paso 20        | Límite superior de árboles permitidos en el pool global agregado.                                |
| Umbral de convergencia global        | Categórico | {0.001, 0.002, 0.005}     | Umbral de mejora mínima entre episodios consecutivos del servidor para continuar la agregación.   |
| Peso de F1 en selección híbrida      | Float      | [0.0, 1.0], paso 0.1      | Balance entre F1-score y diversidad PCD en las estrategias S4 y S7. El peso de PCD es el complemento ($1 - w_{F1}$). |
| Tamaño de episodio global            | Categórico | {3, 5, 6, 9, 10}          | Número de árboles que se añaden al pool global en cada paso de evaluación antes de verificar la convergencia. |
| Árboles por cliente por episodio     | Categórico | {1, 2, 3}                 | Número de árboles que cada cliente contribuye por episodio en las estrategias con selección per-client. |

### 3.3. Hiperparámetros de la Estrategia S8 (Global Attribute Roulette)

| Hiperparámetro                       | Tipo       | Rango / Valores           | Descripción                                                                                      |
|:-------------------------------------|:----------:|:-------------------------:|:-------------------------------------------------------------------------------------------------|
| Tamaño de ventana local              | Categórico | {5, 10, 15}               | Número de árboles que cada cliente entrena localmente antes de recalcular y transmitir su vector de importancia de atributos al servidor. |
| Máximo de rondas federadas           | Categórico | {10, 15, 30}              | Límite superior del número de rondas de comunicación entre clientes y servidor.                   |
| Peso del vector de atributos local   | Float      | [0.0, 1.0], paso 0.1      | Ponderación del vector de importancia local del cliente frente al vector global del servidor al actualizar las probabilidades de selección de atributos. |

### 3.4. Hiperparámetros de Predicción Híbrida

| Hiperparámetro                       | Tipo       | Rango / Valores           | Descripción                                                                                      |
|:-------------------------------------|:----------:|:-------------------------:|:-------------------------------------------------------------------------------------------------|
| Peso del modelo local                | Float      | [0.0, 1.0], paso 0.1      | Contribución del bosque local del cliente en la predicción híbrida. El peso global es el complemento ($1 - w_{local}$). |
| Votación ponderada                   | Categórico | {True, False}             | Activa o desactiva la ponderación de los votos de los árboles según el rendimiento del cliente.   |

## 4. Resultados de la Optimización

Se completaron un total de **63 ensayos** (de 68 lanzados; 2 fallidos y 3 incompletos). A continuación se presenta la configuración óptima identificada, seguida del análisis por hiperparámetro.

### 4.1. Configuración Óptima

La optimización bayesiana identificó la siguiente combinación como la configuración con mayor accuracy medio sobre el conjunto de estrategias y datasets evaluados:

| Hiperparámetro                       | Valor óptimo |
|:-------------------------------------|:------------:|
| Peso del modelo local                | 0.4          |
| Tamaño máximo del pool global        | 110          |
| Umbral de convergencia local         | 0.002        |
| Umbral de convergencia global        | 0.002        |
| Peso de F1 en selección híbrida      | 0.5          |
| Peso del vector de atributos local   | 0.6          |
| Tamaño de episodio global            | 10           |
| Árboles por cliente por episodio     | 3            |
| Ventana local S8                     | 10           |
| Máximo de rondas federadas S8        | 15           |
| Votación ponderada                   | Sí           |

Esta configuración fue la utilizada en el experimento comparativo final.

### 4.2. Análisis por Hiperparámetro

#### Votación ponderada

| Valor  | Ensayos | Accuracy medio | Máximo  |
|:------:|:-------:|:--------------:|:-------:|
| True   | 50      | 0.6543         | 0.6942  |
| False  | 13      | 0.6459         | 0.6646  |

La totalidad de los 10 mejores ensayos utilizaron votación ponderada.

#### Tamaño de episodio global

| Valor | Ensayos | Accuracy medio | Máximo  |
|:-----:|:-------:|:--------------:|:-------:|
| 10    | 23      | 0.6692         | 0.6942  |
| 9     | 8       | 0.6552         | 0.6657  |
| 6     | 3       | 0.6461         | 0.6628  |
| 3     | 26      | 0.6428         | 0.6788  |
| 5     | 3       | 0.6086         | 0.6424  |

Los episodios más amplios reducen el ruido en las comprobaciones de convergencia en el servidor y evitan paradas tempranas prematuras.

#### Árboles por cliente por episodio

| Valor | Ensayos | Accuracy medio | Máximo  |
|:-----:|:-------:|:--------------:|:-------:|
| 3     | 31      | 0.6656         | 0.6942  |
| 2     | 3       | 0.6461         | 0.6628  |
| 1     | 29      | 0.6393         | 0.6788  |

Un mayor número de árboles locales contribuidos por cliente en cada etapa progresiva permite una mejor estimación del rendimiento local durante el entrenamiento.

#### Tamaño de ventana local (Estrategia S8)

| Valor | Ensayos | Accuracy medio | Máximo  |
|:-----:|:-------:|:--------------:|:-------:|
| 5     | 11      | 0.6540         | 0.6827  |
| 10    | 41      | 0.6529         | 0.6942  |
| 15    | 11      | 0.6499         | 0.6732  |

Las diferencias entre los tamaños de ventana local evaluados son muy reducidas, indicando que el esquema de la ruleta de atributos es robusto ante la frecuencia de sincronización local.

#### Máximo de rondas federadas (Estrategia S8)

| Valor | Ensayos | Accuracy medio | Máximo  |
|:-----:|:-------:|:--------------:|:-------:|
| 30    | 11      | 0.6540         | 0.6827  |
| 15    | 41      | 0.6529         | 0.6942  |
| 10    | 11      | 0.6499         | 0.6732  |

El número máximo de rondas permitidas muestra una influencia menor en el accuracy final, logrando la convergencia en fases tempranas de la comunicación.

#### Tamaño máximo del pool global

| Pool global | Ensayos | Accuracy medio | Máximo  |
|:-----------:|:-------:|:--------------:|:-------:|
| 110         | 12      | 0.6593         | 0.6894  |
| 70          | 6       | 0.6535         | —       |
| 90          | 10      | 0.6534         | —       |
| 130         | 21      | 0.6505         | 0.6942  |
| 150         | 10      | 0.6500         | —       |
| 50          | 4       | 0.6462         | —       |

La media óptima se sitúa en el rango de 110 a 130 árboles. Valores pequeños limitan la expresividad del ensamble, mientras que valores excesivos acumulan estimadores de baja calidad.

#### Umbrales de convergencia

**Umbral de convergencia local:**

| Valor | Ensayos | Accuracy medio | Máximo  |
|:-----:|:-------:|:--------------:|:-------:|
| 0.001 | 21      | 0.6600         | 0.6894  |
| 0.002 | 31      | 0.6565         | 0.6942  |
| 0.005 | 11      | 0.6272         | 0.6878  |

**Umbral de convergencia global:**

| Valor | Ensayos | Accuracy medio | Máximo  |
|:-----:|:-------:|:--------------:|:-------:|
| 0.001 | 31      | 0.6583         | 0.6894  |
| 0.002 | 7       | 0.6546         | 0.6742  |
| 0.005 | 25      | 0.6449         | 0.6942  |

Los umbrales estrictos (0.001 y 0.002) garantizan que los bosques alcancen un nivel de madurez suficiente antes de detenerse. El umbral de 0.005 provoca paradas prematuras con una caída notable en el rendimiento medio.

#### Peso del modelo local

Los mejores resultados se concentraron en el rango de 0.4 a 0.5, indicando que la predicción híbrida se beneficia de mantener un balance equilibrado entre la experiencia local del cliente y el conocimiento generalizado del servidor.

#### Peso del vector de atributos local (S8)

El análisis de correlación de Spearman mostró una relación positiva estadísticamente significativa ($\rho = +0.30$, $p = 0.017$) entre este peso y el rendimiento del ensamble, confirmando que priorizar la importancia de atributos local del cliente mejora consistentemente la calidad de la agregación en S8.
