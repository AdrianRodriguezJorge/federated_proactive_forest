# 🔗 FLEX Framework Integration Guide

**Documento**: Integración completa de FLEX Framework con Federated Proactive Forest  
**Versión**: 2.0  
**Última actualización**: Marzo 12, 2026

---

## 📚 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Configuración FLEX](#configuración-flex)
3. [Distribuciones de datos](#distribuciones-de-datos)
4. [Ejemplos de uso](#ejemplos-de-uso)
5. [Capacidades avanzadas](#capacidades-avanzadas)
6. [Solución de problemas](#solución-de-problemas)

---

## 🎯 Introducción

**FLEX Framework** es un framework de Federred Learning flexible que proporciona:

✅ **Distribución de datos**: IID y Non-IID Dirichlet  
✅ **Comunicación federada**: Cliente-Servidor y P2P  
✅ **Abstracción agnóstica**: Funciona con cualquier framework ML  
✅ **Escalabilidad**: Soporte para múltiples clientes y rondas

**En este proyecto**, FLEX se integra para:
- ✅ Particionar datos de forma IID o heterogénea
- ✅ Simular comunicación federada
- ✅ Facilitar experimentación con múltiples estrategias

---

## ⚙️ Configuración FLEX

### Archivo: `src/application/fl_orchestrator.py`

La clase `FLEXOrchestrator` proporciona integración completa con FLEX.

### Configuración Básica

```python
config = {
    # Datos y Federación
    'n_clients': 5,                              # Número de clientes
    'distribution': 'iid',                       # 'iid' o 'noniid_dirichlet'
    'alpha': 0.5,                                # Parámetro Dirichlet (si non-IID)
    
    # Modelo ProactiveForest
    'n_estimators': 100,                         # Árboles por cliente
    'alpha_pf': 0.1,                             # Parámetro de diversidad CPF
    'verbose': False,                            # Logs de entrenamiento
    
    # Agregación
    'strategy': 'S1',                            # S1-S7
    
    # Metadata
    'metadata': {'validation_split': 0.2},       # Validación local
}

from src.application.fl_orchestrator import FLEXOrchestrator
orchestrator = FLEXOrchestrator.from_config(config)
```

---

## 📊 Distribuciones de Datos

### 1. Distribución IID (Uniforme)

**Uso**: Experimento baseline - datos distribuidos uniformemente.

```python
config = {
    'n_clients': 5,
    'distribution': 'iid',  # ← Distribución uniforme
}

orchestrator = FLEXOrchestrator.from_config(config)
orchestrator.setup_federation(dataset_split)
```

**Características**:
- Cada cliente recibe ~20% del dataset (n_clients=5)
- Distribución de clases uniforme por cliente
- Útil para baseline y estudios de escalabilidad

---

### 2. Distribución Non-IID Dirichlet

**Uso**: Simular heterogeneidad realista - clientes tienen distribuciones de clases diferentes.

```python
config = {
    'n_clients': 5,
    'distribution': 'noniid_dirichlet',  # ← Heterogeneidad Dirichlet
    'alpha': 0.5,  # Parámetro Dirichlet
}

orchestrator = FLEXOrchestrator.from_config(config)
orchestrator.setup_federation(dataset_split)
```

**Parámetro `alpha`**:

| Valor | Heterogeneidad | Caso de Uso |
|-------|---|---|
| `alpha = 10.0` | ✅ Baja (cercano a IID) | Datos moderadamente diferentes |
| `alpha = 1.0` | ⚠️ Moderada | Heterogeneidad realista típica |
| `alpha = 0.5` | 🔴 Alta | Heterogeneidad severa |
| `alpha = 0.1` | 🔴🔴 Muy alta | Caso extremo (cada cliente, 1-2 clases) |

**Ejemplo con alpha=0.5:**

```
Cliente 0: [70%, 20%, 10%, 0%, 0%]  (clases desbalanceadas)
Cliente 1: [10%, 60%, 20%, 10%, 0%]
Cliente 2: [5%, 5%, 70%, 15%, 5%]
Cliente 3: [15%, 15%, 10%, 50%, 10%]
Cliente 4: [0%, 0%, 10%, 10%, 80%]
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: IID + S1 (Simple Pool)

```python
from src.infrastructure.dataset import IrisAdapter
from src.application.fl_orchestrator import FLEXOrchestrator

# Cargar dataset
adapter = IrisAdapter("data/iris.csv")
dataset_split = adapter.load_and_split()

# Configurar FL
config = {
    'n_clients': 5,
    'distribution': 'iid',
    'strategy': 'S1',
    'n_estimators': 50,
}

# Ejecutar
orchestrator = FLEXOrchestrator.from_config(config)
orchestrator.setup_federation(dataset_split)
results = orchestrator.run_federated_round()

print(f"Accuracy global: {results.global_accuracy:.4f}")
print(f"Clientes: {results.client_ids}")
```

### Ejemplo 2: Non-IID Dirichlet + S7 (Per-Client F1+PCD)

```python
config = {
    'n_clients': 10,
    'distribution': 'noniid_dirichlet',
    'alpha': 0.5,  # ← Alta heterogeneidad
    'strategy': 'S7',  # ← Agregación per-cliente
    'n_estimators': 100,
    'alpha_pf': 0.1,
}

orchestrator = FLEXOrchestrator.from_config(config)
orchestrator.setup_federation(dataset_split)
results = orchestrator.run_federated_round()

print(f"Comunicación estimada: {results.communication_cost:.2f} MB")
```

### Ejemplo 3: Múltiples Rondas (Convergencia)

```python
config = {...}
orchestrator = FLEXOrchestrator.from_config(config)
orchestrator.setup_federation(dataset_split)

# Ejecutar 5 rondas
results_list = orchestrator.run_multiple_rounds(num_rounds=5)

# Analizar convergencia
accuracies = [r.global_accuracy for r in results_list]
print(f"Ronda 1: {accuracies[0]:.4f}")
print(f"Ronda 5: {accuracies[-1]:.4f}")
```

---

## 🚀 Capacidades Avanzadas

### 1. Estrategias de Agregación (S1-S7)

```python
# Todas están soportadas
strategies = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

for strategy in strategies:
    config['strategy'] = strategy
    orchestrator = FLEXOrchestrator.from_config(config)
    orchestrator.setup_federation(dataset_split)
    results = orchestrator.run_federated_round()
    print(f"{strategy}: {results.global_accuracy:.4f}")
```

### 2. Tracking de Comunicación

```python
results = orchestrator.run_federated_round()

print(f"Datos transferidos: {results.communication_cost:.2f} MB")
print(f"Rondas ejecutadas: {results.num_rounds}")
```

### 3. Integración con Streamlit

```python
# En page_config.py - selector de distribución
distribution = st.selectbox(
    "Distribución",
    ["IID", "Non-IID Dirichlet"]
)

if distribution == "Non-IID Dirichlet":
    alpha = st.slider("Alpha (Dirichlet)", 0.1, 10.0, 0.5)
    config['distribution'] = 'noniid_dirichlet'
    config['alpha'] = alpha
else:
    config['distribution'] = 'iid'
```

---

## 🔧 Solución de Problemas

### Problema 1: ImportError "FLEX not installed"

```python
# Solución: Instalar FLEX
pip install flexible-fl
```

### Problema 2: Distribución Non-IID no funciona

```python
# Verificar: import correcto
from flex.data import FedDataDistribution, FedDatasetConfig

# Verificar: alpha válido (0.1 - 10.0)
config = {
    'alpha': 0.5,  # OK
    # 'alpha': -1.0,  # ERROR
}
```

### Problema 3: Clientes sin datos

```python
# Solución: Verificar cliente_partitions no está vacío
if len(orchestrator.client_partitions) == 0:
    print("ERROR: setup_federation() no fue llamado")
else:
    for client_id, (X, y) in orchestrator.client_partitions.items():
        print(f"{client_id}: {len(X)} muestras ({len(np.unique(y))} clases)")
```

---

## 📊 Matriz de Compatibilidad

| Feature | FLEX | FLEXOrchestrator | Soporte |
|---------|------|---|---|
| IID Distribution | ✅ | ✅ | Completo |
| Non-IID Dirichlet | ✅ | ✅ | Completo |
| FlexPool | ✅ | ⏳ | Parcial* |
| Decoradores | ✅ | ⏳ | Futuro |
| P2P Architecture | ✅ | ⏳ | Futuro |

\* FlexPool disponible en `flex_pool_factory.py`, integración completa en v3.0

---

## 📖 Referencias

- **FLEX GitHub**: https://github.com/FLEXible-FL/FLEXible
- **FLEX Documentation**: https://flexible.readthedocs.io
- **flex-trees**: https://github.com/FLEXible-FL/flex-trees

---

## ✅ Checklist de Verificación

Antes de ejecutar experimentos asegúrate de:

- [ ] FLEX está instalado: `pip check | grep flexible`
- [ ] Dataset está en la ruta correcta
- [ ] Configuración tiene `n_clients >= 2`
- [ ] `alpha` está en rango 0.1-10.0 si non-IID
- [ ] Strategy está en S1-S7
- [ ] Numpy y Scikit-learn actualizados

---

**Autor**: GitHub Copilot  
**Versión**: 2.0  
**Fecha**: Marzo 12, 2026
