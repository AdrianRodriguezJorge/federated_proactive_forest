# Round Robin Dynamic Scoring (RR-DS) Strategy

## Descripción General

**Round Robin Dynamic Scoring (RR-DS)** es una estrategia de agregación para Federated Proactive Forest que implementa un enfoque de 6 fases con selección secuencial de árboles basada en un score dinámico que balancea rendimiento y diversidad.

## Arquitectura de 6 Fases

### Fase 1 — Entrenamiento local en los clientes

Cada cliente entrena su modelo **Proactive Forest (PF)** de manera independiente:
- **Ventanas de tamaño fijo (W)**: Típicamente 5 árboles por ventana
- **Mecanismo PF**: Actualización dinámica de probabilidades para selección de atributos
- **Meta-aprendizaje**: Regula diversidad y tamaño del bosque
- **Métricas reportadas**: Accuracy, Macro-F1, y diversidad interna (PCD)

```python
# Cada cliente entrena con ventanas de W árboles
for window in range(total_trees // window_size):
    trees = client_train_window(X_local, y_local, window_size=W)
    metrics = evaluate_window(trees, X_val, y_val)
    send_to_server(window, trees, metrics)
```

### Fase 2 — Comunicación de ventanas al servidor

Una vez completada cada ventana, el cliente transmite:
- **Estructura completa de los W árboles** (nodos, umbrales, etiquetas)
- **Métricas de desempeño y diversidad** asociadas

Formato de transmisión: **Protobuf** (eficiente, interoperable, seguro)

### Fase 3 — Agregación global con Round Robin

El servidor construye el bosque federado mediante proceso **secuencial e incremental**:

#### Score Dinámico
```
Score(T) = α · F1(T) + (1 - α) · Diversidad(T | G)
```

Donde:
- **F1(T)**: Macro-F1 del árbol T (validación local)
- **Diversidad(T|G)**: PCD (Partition-Coverage Distance) respecto al bosque global actual G
- **α**: Peso entre rendimiento y diversidad (default: 0.5)

#### Proceso Round Robin

1. **Generar permutación aleatoria** de clientes (mitiga sesgo de posición)
2. **Para cada cliente en turno**:
   - Calcular scores dinámicos para todos los árboles en ventana
   - Seleccionar árbol con mayor score
   - Incorporar **inmediatamente** al bosque global
   - Actualizar diversidad para siguiente cliente
3. **Ronda completada**: k árboles añadidos (uno por cliente)

### Fase 4 — Criterio de parada (Progressive Global)

Algoritmo **Progressive Forest** a nivel de servidor:

| Criterio | Descripción |
|----------|-------------|
| **Convergencia** | Mejora ≤ 0.002 durante 2 rondas consecutivas |
| **Máximo rondas** | Límite superior (default: 20 rondas) |
| **T_MAX** | Máximo total de árboles (default: 100) |

Ejemplo: 5 clientes × 20 rondas = máx 100 árboles globales

### Fase 5 — Actualización del modelo en los clientes

Tras agregación global, servidor distribuye ventanas seleccionadas:

```python
# Cada cliente recibe árboles globales
global_trees = receive_from_server()

# Merge sin duplicados: excluye árboles propios ya seleccionados
local_no_selected = [t for i, t in enumerate(local_trees) 
                     if i not in selected_local_ids]
hybrid_forest = local_no_selected + global_trees
```

**Resultado**: Bosque local con generalización global + especialización local

### Fase 6 — Inferencia

**Votación híbrida ponderada** (local, sin comunicación):

```
ŷ = argmax_c ( λ · p_local(c|x) + (1-λ) · p_global(c|x) )
```

- **λ (lambda)**: Peso local vs global (default: 0.5)
- **Ajustable**: Según grado de heterogeneidad de datos

## Hiperparámetros

| Parámetro | Símbolo | Default | Descripción |
|-----------|---------|---------|-------------|
| Window Size | W | 5 | Árboles por ventana |
| Max Rounds | R_MAX | 20 | Máximo rondas Round Robin |
| Alpha Score | α | 0.5 | Balance F1 vs Diversidad |
| Lambda | λ | 0.5 | Peso local vs global (inferencia) |
| Convergencia | δ | 0.002 | Umbral convergencia |
| Max Trees | T_MAX | 100 | Límite árboles globales |

## Comparación con Otras Estrategias

| Estrategia | Enfoque | Selección | Parada |
|------------|---------|-----------|--------|
| **S1** | Simple Pool | Todas, sin orden | N/A |
| **S2-S4** | Global | Accuracy/F1/PCD | Progressive |
| **S5-S7** | Per-Client | Ranking por cliente | Progressive |
| **RR_DS** | **Round Robin** | **Score dinámico** | **Progressive Global** |

## Uso en Streamlit

1. Ir a **Configuración del Experimento**
2. Seleccionar estrategia: `RR_DS — Round Robin Dynamic Scoring`
3. Ajustar parámetros:
   - 🪟 Tamaño ventana (W)
   - 🔁 Máximo rondas (R_MAX)
   - α Score (F1 vs Diversidad)

## Archivo de Configuración

```yaml
aggregation:
  strategy: rr_dynamic
  alpha: 0.5          # Balance F1 vs Diversidad
  window_size: 5      # W: árboles por ventana
  max_rounds: 20      # R_MAX: máximo rondas
  convergence_threshold: 0.002
  t_max: 100          # Máximo árboles globales

prediction:
  local_weight: 0.5   # λ: peso local
  global_weight: 0.5  # (1-λ): peso global
```

## Ventajas

✅ **Equidad**: Round Robin mitiga sesgo de posición  
✅ **Dinámico**: Score se actualiza con estado del bosque global  
✅ **Eficiente**: Comunicación por ventanas reduce overhead  
✅ **Balanceado**: α controla trade-off rendimiento/diversidad  
✅ **Adaptable**: Progressive Forest detecta convergencia temprana  

## Referencias

- **Proactive Forest (PF)**: Mecanismo de selección proactiva de atributos
- **PCD**: Partition-Coverage Distance para diversidad estructural
- **Progressive Forest**: Criterio de parada basado en convergencia
