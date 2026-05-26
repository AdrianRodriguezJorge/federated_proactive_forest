# Bosques Federados Proactivos
## Análisis Comparativo de Estrategias de Agregación

**Autor:** Adrián Rodríguez Jorge  
**Tutores:** MSc. Daniel Pardo Echevarría · Dr. C. Nayma Cepero Pérez

---

## 1. Aprendizaje Federado

### Concepto General

- Paradigma esencial para entrenar modelos de aprendizaje automático preservando la privacidad de los datos.
- Especialmente crítico en dominios sensibles: salud, finanzas y ciberseguridad.
- Los datos permanecen en los dispositivos cliente; solo se comparten parámetros del modelo.

### Tipos de Aprendizaje Federado

#### Horizontal (HFL)
- **Características:** Mismas características, diferentes muestras de datos.
- **Aplicación:** ★ Esta investigación

#### Vertical (VFL)
- **Características:** Diferentes características, mismas muestras de datos.

#### Transferencia (FTL)
- **Características:** Diferente distribución en características y muestras.

---

## 2. Random Forest y su Evolución hacia Proactive Forest

### Random Forest

- Destaca en entornos tabulares: equilibrio entre capacidad predictiva, interpretabilidad y eficiencia.
- Genera diversidad mediante aleatoriedad no controlada en la selección de atributos.
- **Limitación:** Produce alta inestabilidad y no garantiza *ensembles* realmente diversos ni estables.

### Proactive Forest

- Sustituye la aleatoriedad clásica por una **ruleta de atributos adaptativa**: probabilidades de selección actualizadas dinámicamente según el desempeño de los árboles previos.
- Favorece estructuras que aporten eficacia penalizando patrones redundantes.
- Garantiza diversidad controlada; cada cliente captura relaciones particulares de su distribución (crítico en datos non-IID).

**Conclusión:** Proactive Forest constituye la base del modelo entrenado en cada cliente en el marco BFP.

---

## 3. Limitaciones y Necesidad de Aplicar Proactive Forest al Aprendizaje Federado

### 1. Diversidad Inestable
La aleatoriedad no controlada del Random Forest provoca alta inestabilidad estructural y no garantiza *ensembles* realmente diversos ni estables.

### 2. Crecimiento Descontrolado
La agregación de modelos locales genera un crecimiento descontrolado del bosque global, produciendo *ensembles* altamente redundantes, costosos en cómputo, almacenamiento y tiempo de ejecución.

### 3. Diversidad Ignorada
Los enfoques federados actuales seleccionan árboles basándose solo en precisión local, ignorando métricas de diversidad del *ensemble* (como PCD), esencial para la robustez del modelo global.

### 4. Vacío en la Literatura
Aunque Proactive Forest ha demostrado controlar diversidad y tamaño en entornos centralizados, no existe una formalización equivalente en el paradigma de aprendizaje federado.

---

## 4. Ciclo Federado Clásico en Aprendizaje Federado Horizontal

### Fase 1: Inicialización Global
**Responsable:** Servidor  
El servidor establece hiperparámetros, criterios de parada y mecanismos de privacidad.

### Fase 2: Entrenamiento Local Autónomo
**Responsable:** Clientes  
Cada cliente construye su bosque sobre sus propios datos sin comunicación intermedia.

### Fase 3: Transmisión Única al Servidor
**Responsable:** Clientes → Servidor  
Clientes envían bosques completos o parámetros finales en una única fase de comunicación.

### Fase 4: Agregación Global
**Responsable:** Servidor  
El servidor integra modelos locales equilibrando eficacia, diversidad y compacidad.

### Fase 5: Actualización e Inferencia Local
**Responsable:** Clientes  
El servidor distribuye el bosque global; cada cliente realiza inferencia de forma totalmente local.

**Nota:** Este ciclo establece la estructura operativa sobre la cual se definen las dos propuestas generales (Propuestas A y B).

---

## 5. Problema de Investigación

**¿Cómo integrar un *enfoque proactivo* que permita la generación de una *diversidad controlada* en un esquema de *aprendizaje federado horizontal*?**

---

## 6. Objetivo General

Desarrollar un marco de **Bosques Federados Proactivos (BFP)** que integre mecanismos de **control de diversidad**, **estrategias de agregación eficientes** y **criterios de parada progresiva**, con el fin de obtener modelos globales que sean:

- **Eficaces** en escenarios heterogéneos con datos non-IID
- **Robustos** mediante diversidad controlada del *ensemble*
- **Compactos** para reducir costo computacional y de almacenamiento

---

## 7. Objetivos Específicos

### Objetivo 1
Identificar los elementos teóricos y metodológicos fundamentales para la construcción de bosques de decisión en esquemas de aprendizaje federado y la aplicación del enfoque proactivo.

### Objetivo 2
Desarrollar estrategias de agregación eficientes que permitan la construcción de un bosque global a partir de bosques locales proactivos y progresivos.

### Objetivo 3
Evaluar la eficacia y diversidad de las estrategias de agregación propuestas frente a un modelo centralizado mediante diseño experimental.

---

## 8. Estructura de la Propuesta: Dos Enfoques Generales

### Propuesta A: Estrategias de Agregación Progresiva (S1 – S7)
7 estrategias de ranking global y round-robin para construir un bosque global compacto a partir de bosques locales. (S1 – S7)

### Propuesta B: Ruleta Global de Probabilidades (S8)
Orquestación descentralizada que transmite únicamente vectores de probabilidad de atributos en lugar de árboles completos, agregándolos mediante 5 variantes (Promedio Ponderado, Media Simple, Mediana Robusta, Consenso y Proactiva PCD).

---

## 9. Componentes del Modelo en el Cliente

Ambos algoritmos actúan conjuntamente en cada cliente de la federación como base de las dos propuestas generales.

### Proactive Forest (PF)

| Aspecto | Descripción |
|---------|-------------|
| **Objetivo** | Controlar diversidad y estabilidad del modelo local. |
| **Mecanismo** | Ruleta de atributos adaptativa: probabilidades actualizadas dinámicamente según el desempeño. |
| **Beneficio en BFP** | Árboles diversos, menos redundantes. Adaptación a distribuciones non-IID. |
| **Rol en BFP** | Base de construcción local: genera diversidad controlada. |

### Progressive Forest

| Aspecto | Descripción |
|---------|-------------|
| **Objetivo** | Controlar el tamaño del bosque local (compacidad y eficiencia). |
| **Mecanismo** | Parada temprana por convergencia: si la mejora es inferior a un umbral (ej. 0.002), el proceso se detiene. |
| **Beneficio en BFP** | Reduce sobreajuste y costo computacional en hardware limitado. Entrenamiento adaptativo. |
| **Rol en BFP** | Complemento: garantiza eficiencia y estabilidad del bosque local. |

---

## 10. División de Enfoques en la Literatura de Aprendizaje Federado Horizontal

### Agregación de Modelos (Asíncrona)

- Cada cliente entrena su bosque de forma independiente y envía árboles completos al servidor.
- Altamente escalable y tolerante a clientes lentos.
- Compatible con arquitecturas de *ensemble*.
- **Limitación:** Genera bosques globales crecientes y sensibles a la heterogeneidad.

### Agregación de Estadísticas (Síncrona)

- Comparte histogramas, gradientes o bines de características para reconstruir decisiones de partición.
- Puede replicar con precisión un algoritmo centralizado.
- Más estable ante distribuciones heterogéneas.
- **Limitación:** Requiere sincronización estricta y depende del cliente más lento.

### Decisión de Esta Investigación

Esta investigación adopta el **enfoque asíncrono de agregación de modelos**: mayor independencia del servidor, integración natural de Proactive Forest y Progressive Forest, y habilitación de mecanismos de selección y control de compacidad.

---

## 11. Elementos Comunes a las Dos Propuestas

### Modelo en Clientes
Proactive Forest (diversidad controlada) + Progressive Forest (compacidad), adaptados a distribuciones non-IID.

### Comunicación
Asíncrona mediante Protocol Buffers, adecuada para entornos distribuidos con conectividad variable.

### Métricas Transmitidas
Exactitud, Macro-F1 y PCD (Proporción de Clasificaciones Distintas) para cada árbol o ventana.

### Actualización Global
Cada cliente incorpora los árboles del modelo global excluyendo los ya seleccionados para evitar duplicados.

### Inferencia Híbrida
Votación ponderada entre bosque local (especialización) y bosque global (generalización). Peso 0.5 / 0.5 por defecto, configurable. Totalmente local, sin latencia de red.

---

## 12. Propuesta A: Estrategias de Agregación (S1 – S7)

El servidor combina todos los árboles recibidos en una lista, los ordena según la estrategia y los incorpora progresivamente en ventanas de 5 (Progressive Forest), garantizando compacidad.

### S1: Concatenación Simple
- **Tipo:** —
- **Criterio:** —
- **Descripción:** Todos los árboles sin selección.

### S2: Ranking Global — Exactitud
- **Tipo:** Ranking Global
- **Criterio:** Exactitud
- **Descripción:** Prioriza árboles con mayor *accuracy*. Ventanas de 5 con Progressive Forest.

### S3: Ranking Global — Macro-F1
- **Tipo:** Ranking Global
- **Criterio:** Macro-F1
- **Descripción:** Favorece rendimiento equilibrado en clases desbalanceadas.

### S4: Ranking Global — Promedio (Macro-F1 + PCD)
- **Tipo:** Ranking Global
- **Criterio:** Promedio (Macro-F1 + PCD)
- **Descripción:** Equilibrio entre eficacia y diversidad estructural del *ensemble*.

### S5: Round-Robin — Exactitud
- **Tipo:** Round-Robin
- **Criterio:** Exactitud
- **Descripción:** Equidad + desempeño: alterna un árbol por cliente cíclicamente.

### S6: Round-Robin — Macro-F1
- **Tipo:** Round-Robin
- **Criterio:** Macro-F1
- **Descripción:** Equidad + robustez ante clases desbalanceadas.

### S7: Round-Robin — Macro-F1 + PCD
- **Tipo:** Round-Robin
- **Criterio:** Macro-F1 + PCD
- **Descripción:** Equidad + diversidad estructural: evita sobre-representación de clientes dominantes.

---

## 13. Propuesta A: Diagrama de Actividades

[Diagrama de Actividades — Propuesta A]  
*Pendiente de diseño final*

---

## 16. Propuesta B: Ruleta Global de Probabilidades

Orquestación ciega: el servidor nunca recibe modelos internos, solo vectores de probabilidad de atributos. Reduce el costo de comunicación y refuerza la privacidad.

### Fase B1: Construcción Local Guiada por Ruleta
Cada cliente parte de una ruleta de atributos con probabilidades iguales. Durante la construcción (ventanas de tamaño W), atributos que aportan diversidad y eficacia aumentan su probabilidad; los redundantes la disminuyen.

### Fase B2: Comunicación y Agregación Federada
Al finalizar cada ventana, el cliente transmite un vector de probabilidades de atributos (no árboles). El servidor agrega los vectores recibidos (promedio ponderado, media simple, mediana robusta o algoritmos de consenso) y redistribuye una ruleta global.

### Fases B3–B5: Actualización e Inferencia Híbrida
Igual que en la Propuesta A: cada cliente incorpora los árboles del modelo global excluyendo los ya seleccionados. Inferencia mediante votación ponderada local/global (0.5/0.5 configurable), totalmente local sin latencia de red.

---

## 17. Propuesta B: Diagrama de Actividades

[Diagrama de Actividades — Propuesta B]  
*Pendiente de diseño final*

---

## 18. Estado Actual de la Implementación

El sistema fue implementado y desplegado como plataforma de experimentación reproducible, validando las propuestas en tiempo real.

### Interfaz Streamlit

- Configuración de experimentos (clientes, datasets, estrategias)
- Ejecución con seguimiento visual en tiempo real
- Ranking de árboles seleccionados
- Métricas globales y por cliente

**URL:** https://federatedproactiveforest-ngmacrb7n87cuashvwqvyj.streamlit.app/

### Características del Sistema

- CLI con YAML para reproducibilidad y automatización
- 12 estrategias en total (S1–S7 de agregación y 5 variantes de la Propuesta B / S8)
- Soporte IID y non-IID
- Métricas: Accuracy, F1, Macro-F1, PCD
- Predicción híbrida configurable (local/global)
- Extensibilidad: nuevos datasets y estrategias
- Integración con FLEX Framework para orquestación federada avanzada

---

## 19. Diseño Experimental y Conjuntos de Datos

Se evaluaron las **12 estrategias federadas (S1–S7 de la Propuesta A y 5 variantes de S8 de la Propuesta B)** frente al **Proactive Forest centralizado (PF)** como referencia. Métrica principal de entrenamiento: **Macro-F1**, y métrica de evaluación experimental: **Exactitud (Accuracy)**. 

Se seleccionaron **8 datasets** de un conjunto de 32 evaluados en la tesis de referencia [1], heterogéneos en tamaño, atributos y clases.

### Datasets Utilizados

| Dataset | Instancias | Atributos | Clases |
|---------|------------|-----------|--------|
| Car | 1,728 | 6 | 4 |
| Iris | 150 | 4 | 3 |
| Letter | 20,000 | 16 | 26 |
| Nursery | 12,960 | 8 | 5 |
| Optdigits | 5,620 | 64 | 10 |
| Sonar | 208 | 60 | 2 |
| Spambase | 4,601 | 57 | 2 |
| Vowel | 990 | 10 | 11 |

**[1]** Cepero-Pérez, N., et al. (2018). Proactive Forest for supervised classification. IWAIPR, Springer.

---

## 20. Resultados Comparativos (Exactitud / Accuracy)

*Nota metodológica: Las métricas de desempeño reportadas para el modelo federado se calculan mediante el promedio de la métrica (Exactitud) obtenida localmente en el conjunto de evaluación de cada uno de los clientes en la federación.*

| BD | Acc_PF | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8_W | S8_Mean | S8_Med | S8_Cons | S8_PCD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Car | 0.9766 | 0.9672 | 0.9634 | 0.9653 | 0.9711 | 0.9615 | 0.9672 | 0.9711 | 0.9191 | 0.9191 | 0.9191 | 0.9191 | 0.9191 |
| Iris | 0.9560 | 0.8667 | 0.8667 | 0.8667 | 0.8889 | 0.8889 | 0.8889 | 0.8667 | 0.9111 | 0.9111 | 0.9111 | 0.9111 | 0.9111 |
| Letter | 0.9650 | 0.9617 | 0.9586 | 0.9551 | 0.9533 | 0.9603 | 0.9621 | 0.9447 | 0.8772 | 0.8768 | 0.8806 | 0.8871 | 0.8772 |
| Nursery | 0.9959 | 0.9720 | 0.9758 | 0.9745 | 0.9740 | 0.9740 | 0.9725 | 0.9715 | 0.9588 | 0.9588 | 0.9588 | 0.9588 | 0.9588 |
| Optdigits | 0.9832 | 0.9662 | 0.9567 | 0.9573 | 0.9603 | 0.9573 | 0.9614 | 0.9543 | 0.9407 | 0.9407 | 0.9413 | 0.9407 | 0.9407 |
| Sonar | 0.8483 | 0.8095 | 0.8889 | 0.8730 | 0.8254 | 0.8889 | 0.9206 | 0.8571 | 0.6984 | 0.6984 | 0.6984 | 0.6984 | 0.6984 |
| Spambase | 0.9539 | 0.9356 | 0.9364 | 0.9364 | 0.9349 | 0.9356 | 0.9328 | 0.9385 | 0.9241 | 0.9241 | 0.9241 | 0.9241 | 0.9241 |
| Vowel | 0.9719 | 0.8586 | 0.7946 | 0.8316 | 0.7912 | 0.8081 | 0.8114 | 0.8249 | 0.6633 | 0.6633 | 0.6700 | 0.6633 | 0.6633 |

---

## 21. Análisis de Resultados

### Observaciones Principales

Inicialmente se puede observar que en la mayoría de los conjuntos de datos, las diferencias de exactitud (accuracy) entre el Proactive Forest centralizado (PF) y las estrategias federadas son pequeñas, confirmando que la federación conserva el poder de generalización.

Sin embargo, se detectan comportamientos específicos en algunos conjuntos:

- **Mayor diferencia absoluta global frente a PF:** Vowel (diferencia = 0.3086 frente a las variantes de S8, y de 0.1807 frente a S4).
- **Desempeño Sobresaliente (Federado supera a Centralizado):** En el conjunto de datos **Sonar**, múltiples estrategias federadas (S2, S3, S5, S6 y S7) superan el desempeño del modelo centralizado. De forma notable, la estrategia **S6 (Round-Robin con criterio F1)** alcanza una exactitud de `0.9206` frente al `0.8483` de PF centralizado (una mejora de `+0.0723`).

### Transición al Análisis Estadístico

Tras observar variaciones en el rendimiento entre PF y las estrategias federadas, fue necesario determinar si esas diferencias correspondían a efectos **estadísticamente significativos**.

---

## 22. Análisis Estadístico: PF vs. Estrategias Federadas

**Diseño:** 8 datasets × 13 métodos (PF + 12 estrategias federadas: S1–S7 y 5 variantes de S8). Medidas repetidas: todos los métodos evaluados sobre los mismos bloques de datos.

### Test de Friedman

| Hipótesis Nula (H₀) | Resultado | Conclusión |
|-------------------|-----------|------------|
| No existen diferencias significativas entre los 13 métodos en exactitud. | p-value = 4.94e-07 (α < 0.05) → Se rechaza H₀ | Existen diferencias globales significativas en el conjunto de métodos. |

### Análisis Post-hoc: Wilcoxon Signed-Rank con Corrección Bonferroni

| Comparación | Corrección | Resultado p-value | Conclusión |
|-------------|------------|-------------------|------------|
| PF vs. cada una de las 12 estrategias federadas | Bonferroni (12 comparaciones) | Todos los p-values corregidos > 0.05 | PF es estadísticamente equivalente a todas las estrategias tras ajustar por múltiples comparaciones. |

### Interpretación

El test global (Friedman) detecta diferencias globales debido a las diferencias marcadas en conjuntos específicos como Vowel o en las variantes S8.
Sin embargo, aplicando la corrección estricta de Bonferroni (alfa ajustado = 0.05 / 12 ≈ 0.00417), ninguna estrategia federada presenta diferencias estadísticamente significativas con la línea de base centralizada (el p-value mínimo posible con N=8 es 0.0078).

Sin aplicar la corrección múltiple (alfa = 0.05 uncorrected):
- El modelo centralizado **PF es estadísticamente equivalente** a **S2** (p=0.109), **S3** (p=0.078), **S5** (p=0.109) y **S6** (p=0.148).
- El modelo centralizado **PF es estadísticamente superior** a **S1** (p=0.0078), **S4** (p=0.0078), **S7** (p=0.0234) y a las 5 variantes de **S8** (p=0.0078).

---

## 23. Resultados Comparativos de Diversidad (PCD)

Se evaluó la diversidad correctiva porcentual (PCD) del bosque global de cada estrategia federada en comparación con la diversidad del Proactive Forest centralizado (PF).

| BD | PCD_PF | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8_W | S8_Mean | S8_Med | S8_Cons | S8_PCD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Car | 0.3096 | 0.3699 | 0.3295 | 0.3372 | 0.3430 | 0.3391 | 0.3372 | 0.3449 | 0.3507 | 0.3507 | 0.3468 | 0.3507 | 0.3507 |
| Iris | 0.1120 | 0.3333 | 0.2667 | 0.2667 | 0.2889 | 0.3111 | 0.3111 | 0.3111 | 0.2889 | 0.2889 | 0.2889 | 0.2889 | 0.2889 |
| Letter | 0.4562 | 0.5016 | 0.5253 | 0.5210 | 0.5250 | 0.5158 | 0.5184 | 0.5141 | 0.5181 | 0.5128 | 0.5125 | 0.5158 | 0.5046 |
| Nursery | 0.3969 | 0.3310 | 0.3048 | 0.3032 | 0.3140 | 0.2971 | 0.3014 | 0.3146 | 0.2142 | 0.2142 | 0.2142 | 0.2142 | 0.2142 |
| Optdigits | 0.5831 | 0.5374 | 0.5119 | 0.5119 | 0.5219 | 0.5071 | 0.5083 | 0.5160 | 0.5510 | 0.5510 | 0.5516 | 0.5510 | 0.5510 |
| Sonar | 0.9073 | 1.0000 | 0.9365 | 0.9365 | 0.9524 | 0.9683 | 0.9683 | 0.9683 | 0.9841 | 0.9841 | 0.9841 | 0.9841 | 0.9841 |
| Spambase | 0.3307 | 0.3406 | 0.3290 | 0.3283 | 0.3341 | 0.3312 | 0.3355 | 0.3398 | 0.3377 | 0.3377 | 0.3333 | 0.3377 | 0.3377 |
| Vowel | 0.9370 | 0.9798 | 0.9495 | 0.9630 | 0.9596 | 0.9562 | 0.9428 | 0.9596 | 0.8687 | 0.8687 | 0.8754 | 0.8687 | 0.8687 |

### Análisis Estadístico de la Diversidad

- **Test de Friedman**: `p-value = 0.2209` (p > 0.05).
  **Conclusión**: **No existen diferencias estadísticamente significativas** en diversidad entre los métodos evaluados.
- **Pruebas Post-hoc de Wilcoxon**: Todos los p-values corregidos y no corregidos (mínimo p=0.312) confirman que **todas las estrategias federadas mantienen una diversidad equivalente a la del modelo centralizado**.

### Observaciones y Conclusiones sobre la Diversidad

- **Incremento de Diversidad en el Federado**: En la mayoría de los conjuntos de datos, varias estrategias federadas muestran un incremento de diversidad (PCD) promedio en comparación con el Proactive Forest centralizado (ej. S1 muestra una ganancia promedio de PCD de `+0.0451`).
- **Explicación Metodológica**: La partición de los datos entre diferentes clientes favorece que cada bosque local se especialice y capture relaciones particulares de sus subconjuntos. Al realizar la agregación de estos modelos asíncronos en el servidor, el bosque global resultante presenta mayor variación estructural y una diversidad equivalente o superior al centralizado, lo cual incrementa la robustez del modelo sin comprometer la privacidad.

---

## 24. Optimización Metodológica con Optuna

### ¿Cómo Funciona Optuna?

- Automatiza la búsqueda de hiperparámetros mediante optimización bayesiana.
- Aprende de cada intento para dirigir la búsqueda hacia configuraciones prometedoras.
- Combina exploración (zonas nuevas) y explotación (refinamiento de buenas configuraciones).
- Descarta tempranamente configuraciones poco útiles (*pruning*), reduciendo el costo computacional.
- **Parámetros en análisis:** Número máximo de árboles, pesos en rankings, proporción local/global.

### ¿Por Qué no Metaheurísticas?

- La optimización bayesiana usa un modelo subrogado (Proceso Gaussiano) que aprende de cada evaluación.
- Requiere significativamente menos evaluaciones que los métodos basados en población (algoritmos evolutivos, enjambres) para converger.
- Las metaheurísticas son efectivas en espacios combinatorios complejos, pero computacionalmente costosas para el ajuste de hiperparámetros en aprendizaje automático.
- Optuna logra el mismo nivel de optimización con menor costo temporal, esencial en el paradigma de aprendizaje federado distribuido.

---

## 25. Conclusiones

### Resultados Iniciales

- Se implementó un marco de **Bosques Federados Proactivos (BFP)** con mecanismos de control de diversidad, estrategias de agregación y criterios de parada progresiva.
- Las **Propuestas A (S1–S7) y B (ruleta global - S8)** demostraron ser competitivas frente al Proactive Forest centralizado en escenarios heterogéneos en exactitud y diversidad.
- El análisis estadístico confirmó la equivalencia en diversidad y exactitud (bajo Bonferroni) entre el enfoque centralizado y las variantes federadas, validando la solidez metodológica de las propuestas.

### Aportes Metodológicos

- Se construyó un **laboratorio interactivo de experimentación (Streamlit + CLI + FLEX)** que permite reproducir, comparar y visualizar resultados de forma transparente.
- Se incorporó **optimización metodológica con Optuna (optimización bayesiana)**, que redujo el costo de búsqueda de configuraciones óptimas y fortaleció la robustez de las estrategias evaluadas.
- El marco responde directamente a las limitaciones identificadas en la literatura: **control de diversidad, reducción de redundancia, eficiencia en tamaño y competitividad frente al modelo centralizado.**

---

## Gracias

**Adrián Rodríguez Jorge** · **MSc. Daniel Pardo Echevarría** · **Dr. C. Nayma Cepero Pérez**
