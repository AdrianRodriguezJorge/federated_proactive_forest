# Implementación Round Robin Dynamic Scoring (RR-DS) - Resumen

## ✅ Archivos Creados/Modificados

### Nuevos Archivos

| Archivo | Descripción |
|---------|-------------|
| `src/domain/aggregation/strategies/round_robin_dynamic/__init__.py` | Package init |
| `src/domain/aggregation/strategies/round_robin_dynamic/round_robin_dynamic_strategy.py` | Implementación completa RR-DS (523 líneas) |
| `configs/experiments/exp008_rr_dynamic.yaml` | Configuración experimento |
| `RR_DS_README.md` | Documentación detallada |
| `IMPLEMENTATION_SUMMARY.md` | Este resumen |

### Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `src/domain/aggregation/aggregation_factory.py` | + Import RR_DS, + registro en factory |
| `src/interfaces/streamlit/pages_manual/page_config.py` | + UI config RR_DS, + parámetros window_size/max_rounds/alpha |
| `src/application/fl_orchestrator.py` | + Soporte RR_DS en aggregate_kwargs |

## 📁 Estructura de Carpetas

```
federated_proactive_forest/
├── src/
│   ├── domain/
│   │   └── aggregation/
│   │       ├── strategies/
│   │       │   └── round_robin_dynamic/        [NUEVA CARPETA]
│   │       │       ├── __init__.py
│   │       │       └── round_robin_dynamic_strategy.py
│   │       └── aggregation_factory.py          [MODIFICADO]
│   └── interfaces/
│       └── streamlit/
│           └── pages_manual/
│               └── page_config.py              [MODIFICADO]
├── configs/
│   └── experiments/
│       └── exp008_rr_dynamic.yaml              [NUEVO]
├── RR_DS_README.md                             [NUEVO]
└── IMPLEMENTATION_SUMMARY.md                   [NUEVO]
```

## 🎯 Características Implementadas

### Fase 1: Entrenamiento Local
- ✅ Ventanas de tamaño fijo (W = 5 árboles)
- ✅ Proactive Forest con actualización de probabilidades
- ✅ Métricas: Accuracy, Macro-F1, PCD

### Fase 2: Comunicación
- ✅ Transmisión de ventanas al servidor
- ✅ Formato eficiente (listo para Protobuf)

### Fase 3: Agregación Round Robin
- ✅ Score dinámico: `Score(T) = α·F1(T) + (1-α)·Diversidad(T|G)`
- ✅ Permutación aleatoria por ronda (mitiga sesgo)
- ✅ Selección secuencial con actualización inmediata
- ✅ Cálculo PCD para diversidad estructural

### Fase 4: Criterio de Parada
- ✅ Progressive Forest global
- ✅ Convergencia: δ ≤ 0.002 por 2 rondas consecutivas
- ✅ Máximo rondas: R_MAX = 20 (configurable)
- ✅ T_MAX: Máximo 100 árboles globales

### Fase 5: Actualización Clientes
- ✅ ClientUpdater existente (merge sin duplicados)
- ✅ Exclusión árboles locales ya seleccionados

### Fase 6: Inferencia
- ✅ HybridPredictor existente
- ✅ Votación híbrida: `λ·local + (1-λ)·global`
- ✅ λ configurable (default: 0.5)

## 🔧 Hiperparámetros

| Parámetro | Default | Rango | Descripción |
|-----------|---------|-------|-------------|
| `window_size` | 5 | 2-20 | Árboles por ventana |
| `max_rounds` | 20 | 5-50 | Máximo rondas Round Robin |
| `alpha` | 0.5 | 0.0-1.0 | Balance F1 vs Diversidad |
| `convergence_threshold` | 0.002 | 0.0001-0.01 | Umbral convergencia |
| `t_max` | 100 | - | Máximo árboles globales |
| `local_weight` (λ) | 0.5 | 0.0-1.0 | Peso local en inferencia |

## 🖥️ Interfaz Streamlit

### Configuración RR-DS

Al seleccionar la estrategia `RR_DS` en Streamlit, se muestran:

1. **🪟 Tamaño ventana (W)**: Número de árboles por ventana
2. **🔁 Máximo rondas (R_MAX)**: Límite de rondas Round Robin
3. **α Score (F1 vs Diversidad)**: Balance en score dinámico

### Comparación Estrategias

```
S1 — Simple Pool (todos los árboles, sin ordenar)
S2 — Global, orden por Accuracy + Progressive
S3 — Global, orden por Macro-F1 + Progressive
S4 — Global, orden por α·F1 + β·PCD + Progressive
S5 — Per-Client, orden por Accuracy + Progressive
S6 — Per-Client, orden por Macro-F1 + Progressive
S7 — Per-Client, orden por α·F1 + β·PCD + Progressive
RR_DS — Round Robin Dynamic Scoring (ventanas + score dinámico F1+Diversidad) [NUEVA]
```

## 🧪 Tests Realizados

```bash
# Test 1: Creación de estrategia desde factory
✅ AggregationFactory.create_strategy('RR_DS')

# Test 2: Imports en Streamlit
✅ STRATEGY_LABELS incluye 'rr_dynamic'

# Test 3: Configuración desde YAML
✅ exp008_rr_dynamic.yaml válido
```

## 📊 Fórmulas Clave

### Score Dinámico
```
Score(T) = α · F1(T) + (1 - α) · Diversidad(T | G)
```

### Diversidad (PCD)
```
Diversidad(T | G) = promedio(PCD(T, T_i) for T_i in G)
```

### Inferencia Híbrida
```
ŷ = argmax_c ( λ · p_local(c|x) + (1-λ) · p_global(c|x) )
```

## 🚀 Uso Rápido

### Desde Streamlit
1. Ejecutar: `streamlit run src/interfaces/streamlit/app.py`
2. Ir a **Configuración del Experimento**
3. Seleccionar: `RR_DS — Round Robin Dynamic Scoring`
4. Ajustar parámetros según necesidad
5. Guardar configuración
6. Ejecutar experimento en **Run Experiment**

### Desde CLI (próximamente)
```bash
python -m src.interfaces.cli.main --config configs/experiments/exp008_rr_dynamic.yaml
```

## 📈 Ventajas vs Otras Estrategias

| Ventaja | Descripción |
|---------|-------------|
| **Equidad** | Round Robin mitiga sesgo de posición |
| **Dinámico** | Score se actualiza con estado del bosque |
| **Eficiente** | Comunicación por ventanas reduce overhead |
| **Balanceado** | α controla trade-off rendimiento/diversidad |
| **Adaptable** | Progressive Forest detecta convergencia |

## 🔍 Próximos Pasos (Opcionales)

1. **Visualización**: Agregar panel de métricas específicas para RR_DS
2. **Protobuf**: Implementar serialización eficiente
3. **Tests unitarios**: Crear tests específicos para RR_DS
4. **Benchmarking**: Comparar rendimiento vs S1-S7
5. **Ablación**: Estudiar impacto de α, W, R_MAX

## 📚 Referencias

- **Proactive Forest**: Mecanismo de selección proactiva
- **PCD**: Partition-Coverage Distance
- **Progressive Forest**: Criterio de parada por convergencia
- **FLEX Framework**: Orquestación federada

---

**Estado**: ✅ Implementación Completa  
**Versión**: 1.0  
**Fecha**: 2026-04-01
