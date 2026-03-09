# 🌲 Federated Proactive Forest

**Federated Proactive Forest** es un sistema de **aprendizaje federado horizontal** que implementa y compara **7 estrategias de agregación** basadas en **Proactive Forest** (Cepero, 2023). 

El proyecto combina un marco de FL puro (sin dependencias de librerías FL externas) con interfaces interactivas (CLI y Streamlit Web), permitiendo investigar diferentes criterios de selección y ranking de árboles en entornos distribuidos.

## 🎯 Características Principales

- ✅ **7 estrategias de agregación** (S1-S7) con criterios variados
- ✅ **Aprendizaje Federado Horizontal** con N clientes configurables
- ✅ **Detección de Intrusiones** usando NSL-KDD o Iris
- ✅ **Interfaz Web** con Streamlit (4 páginas interactivas)
- ✅ **CLI** con configuración YAML
- ✅ **Arquitectura Hexagonal** (Application → Domain → Infrastructure)
- ✅ **Extensible**: Agregar nuevos datasets fácilmente

## 📋 Tabla de Contenidos

- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Datasets Soportados](#-datasets-soportados)
- [Estrategias de Agregación](#-estrategias-de-agregación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Interfaz Streamlit](#-interfaz-streamlit)
- [CLI](#-cli)
- [Agregar Nuevos Datasets](#-agregar-nuevos-datasets)
- [Documentación Completa](#-documentación-completa)

## 🚀 Instalación

### Requisitos
- Python 3.8+
- pip

### Pasos

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu_usuario/federated_proactive_forest.git
   cd federated_proactive_forest
   ```

2. **Crear entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o en Windows:
   venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Uso Rápido

### Interfaz Web (Streamlit)

```bash
streamlit run src/interfaces/streamlit/app.py
```

Abre tu navegador en `http://localhost:8501`

**Flujo de trabajo:**
1. **⚙️ Configuración** — Selecciona dataset, #clientes, estrategia, pesos
2. **▶️ Ejecutar** — Inicia la ronda federada
3. **🏆 Ranking** — Visualiza árboles seleccionados vs descartados
4. **📊 Métricas** — Resultados: Accuracy, F1, PCD, matriz confusión

### CLI

```bash
python -m src.interfaces.cli.main --config configs/experiments/exp007_s7_perclient_f1_pcd.yaml
```

Salida: Resultados en consola + tabla resumen

## 📊 Datasets Soportados

| Dataset | Muestras | Features | Clases | Adaptador | Tiempo FL |
|---------|----------|----------|--------|-----------|-----------|
| **NSL-KDD** | 5000+ | 42 (39 num + 3 cat) | 5 | `nslkdd_adapter.py` | 4-8 min |
| **Iris** | 150 | 4 (todos num) | 3 | `iris_adapter.py` | 1-2 min |
| **CSV Genérico** | Variable | Variable | Variable | `csv_adapter.py` | Variable |

### Descargar NSL-KDD

```bash
# Descargar desde https://www.unb.ca/cic/datasets/nsl.html
# Extraer KDDTrain+.csv y KDDTest+.csv en data/
mkdir -p data
# Copiar archivos CSV a data/
```

### Iris (incluido)

Iris ya está en `data/iris.csv`. Úsalo para pruebas rápidas.

## 🎯 Estrategias de Agregación

| ID | Nombre | Ámbito | Criterio | Per-Client | Tiempo |
|----|--------|--------|----------|-----------|--------|
| **S1** | Simple Pool | - | - | ❌ | <5 seg |
| **S2** | Global Accuracy | Global | Accuracy | ❌ | 10-30 seg |
| **S3** | Global F1 | Global | Macro-F1 | ❌ | 10-30 seg |
| **S4** | Global F1+PCD | Global | α·F1+β·PCD | ❌ | 15-40 seg |
| **S5** | Per-Client Accuracy | Per-Client | Accuracy | ✅ | 15-40 seg |
| **S6** | Per-Client F1 | Per-Client | Macro-F1 | ✅ | 15-40 seg |
| **S7** | Per-Client F1+PCD | Per-Client | α·F1+β·PCD | ✅ | 20-50 seg |

**Tiempo Total**: ~4-8 min (NSL-KDD, 5 clientes, 100 árboles)

### Detalles de Estrategias

- **S1**: Baseline (sin selección)
- **S2-S4**: Ranking global + Progressive stopping (todos vs todos)
- **S5-S7**: Ranking per-cliente + Round-robin (diversidad garantizada)
- **Progressive Stopping**: Detiene agregación si converge

## 🏗️ Estructura del Proyecto

```
federated_proactive_forest/
├── README.md                          # Este archivo
├── DATASETS_GUIDE.md                  # Guía: cómo agregar datasets
├── pyproject.toml                     # Configuración Python
├── requirements.txt                   # Dependencias
├── .gitignore                         # Ignorar archivos
│
├── src/
│   ├── application/
│   │   └── fl_orchestrator.py        # Orquestador principal (6 pasos FL)
│   │
│   ├── domain/
│   │   ├── aggregation/              # Estrategias S1-S7
│   │   │   ├── base_strategy.py
│   │   │   ├── strategies.py         # 7 implementaciones
│   │   │   ├── tree_ranker.py        # Ranking de árboles
│   │   │   └── cpf_stopper.py        # Progressive stopping
│   │   ├── dataset/
│   │   │   ├── base_adapter.py       # Interfaz IDatasetAdapter
│   │   ├── metadata/
│   │   │   └── client_metadata.py    # Metadatos cliente
│   │   ├── metrics/
│   │   │   └── forest_evaluator.py   # Cálculo de métricas
│   │   ├── prediction/
│   │   │   └── hybrid_predictor.py   # Predicción local+global
│   │   └── update/
│   │       └── client_updater.py     # No-Repeat Merge
│   │
│   ├── infrastructure/
│   │   └── dataset/
│   │       ├── nslkdd_adapter.py     # Adaptador NSL-KDD
│   │       ├── iris_adapter.py       # Adaptador Iris ✨
│   │       └── csv_adapter.py        # Adaptador genérico
│   │
│   └── interfaces/
│       ├── cli/
│       │   └── main.py               # Interfaz CLI
│       └── streamlit/
│           ├── app.py                # Entrypoint
│           └── pages/
│               ├── page_config.py    # ⚙️ Configuración
│               ├── page_run.py       # ▶️ Ejecutar
│               ├── page_ranking.py   # 🏆 Ranking
│               └── page_metrics.py   # 📊 Métricas
│
├── proactive_forest/                 # Código original Cepero 2023
│   ├── estimator.py
│   ├── tree.py
│   ├── tree_builder.py
│   ├── newalg.py                     # ComparativeProgressiveForest (CPF)
│   ├── criteria_and_splits.py
│   ├── selection_and_diversity.py
│   ├── sampling_and_voting.py
│   ├── utils.py
│   └── probabilites.py
│
├── configs/
│   └── experiments/                  # Configuraciones YAML
│       ├── exp001_s1_simple.yaml
│       └── exp007_s7_perclient_f1_pcd.yaml
│
└── data/
    ├── iris.csv                      # ✅ Incluido (150 muestras)
    ├── NSL-KDD_train.csv             # Descargar
    └── NSL-KDD_test.csv              # Descargar
```

## 🎨 Interfaz Streamlit

### Página 1: ⚙️ Configuración
- Seleccionar dataset (NSL-KDD, Iris, CSV personalizado)
- Configurar #clientes, distribución (IID/Non-IID)
- Parámetros modelo (Proactive Forest)
- Estrategia (S1-S7) y pesos
- Predicción híbrida (pesos locales/globales)

### Página 2: ▶️ Ejecutar
- Log en vivo con progreso
- Resumen: Accuracy, F1, #árboles, estrategia
- Tabla: Clientes con métricas locales y extendidas

### Página 3: 🏆 Ranking de Árboles
- Tabla filtrable: todos los árboles con scores
- Filtrado por cliente(s)
- Mostrar solo seleccionados
- Ordenamiento flexible

### Página 4: 📊 Métricas
- Cards: Accuracy, F1, PCD globales
- Tabla: Métricas por cliente
- Matriz de confusión interactiva
- Gráficos F1 por clase

## 💻 CLI

### Uso

```bash
python -m src.interfaces.cli.main --config <ruta_yaml>
```

### Ejemplo

```bash
python -m src.interfaces.cli.main --config configs/experiments/exp007_s7_perclient_f1_pcd.yaml
```

### Formato YAML

```yaml
experiment:
  name: "exp007_s7"
  seed: 42

dataset:
  type: "NSL-KDD"  # o "Iris"
  train_path: "data/NSL-KDD_train.csv"
  test_path: "data/NSL-KDD_test.csv"

federation:
  n_clients: 5
  distribution: "noniid_dirichlet"
  dirichlet_alpha: 0.5

model:
  n_estimators: 100
  use_progressive_stopping: true
  convergence: 0.002

aggregation:
  strategy: "s7_perclient_f1_pcd"
  f1_weight: 0.5
  pcd_weight: 0.5
```

## 📚 Agregar Nuevos Datasets

### Paso 1: Crear Adaptador

```python
# src/infrastructure/dataset/mi_adapter.py

from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MiDatasetAdapter(IDatasetAdapter):
    def __init__(self, data_path, scale=True, scaler_type="standard"):
        self.data_path = data_path
        self.scale = scale
        self._scaler = StandardScaler() if scale else None
        self._label_encoder = LabelEncoder()
    
    @property
    def name(self) -> str:
        return "mi_dataset"
    
    def load(self) -> DatasetSplit:
        df = pd.read_csv(self.data_path)
        # ... cargar, escalar, codificar ...
        return DatasetSplit(...)
```

### Paso 2: Actualizar UI

```python
# src/interfaces/streamlit/pages/page_config.py

# En selectbox:
dataset_type = st.selectbox("Tipo de dataset",
    ["NSL-KDD", "Iris", "Mi Dataset", "CSV personalizado"])

# En _load_dataset():
elif d["type"] == "Mi Dataset":
    from src.infrastructure.dataset.mi_adapter import MiDatasetAdapter
    adapter = MiDatasetAdapter(d["train_path"], ...)
```

Ver [DATASETS_GUIDE.md](DATASETS_GUIDE.md) para guía detallada.

## 📖 Documentación Completa

- [DATASETS_GUIDE.md](DATASETS_GUIDE.md) — Guía completa para agregar datasets
- `src/application/fl_orchestrator.py` — Documentación de 6 pasos FL
- `src/domain/aggregation/strategies.py` — Detalles estrategias S1-S7

## 🔬 Ejemplo de Ejecución: Iris

```bash
# 1. Iniciar Streamlit
streamlit run src/interfaces/streamlit/app.py

# 2. En navegador (http://localhost:8501):
#    ⚙️ Configuración:
#      - Dataset: "Iris"
#      - Clientes: 3
#      - Estrategia: "s7_perclient_f1_pcd"
#      - Guardar ✅

#    ▶️ Ejecutar:
#      - Ejecutar ronda → Esperar 1-2 min ⏱️

#    🏆 Ranking:
#      - Ver árboles seleccionados
#      - Filtrar por cliente

#    📊 Métricas:
#      - Resultados finales
```

**Resultado esperado:**
- Global Accuracy: ~0.95+
- Tiempo: 1-2 min (vs 4-8 min para NSL-KDD)

## 🛠️ Tecnologías

| Tech | Versión | Rol |
|------|---------|-----|
| Python | 3.8+ | Lenguaje |
| NumPy | ≥1.24 | Computación numérica |
| Pandas | ≥2.0 | Manipulación datos |
| scikit-learn | ≥1.3 | ML: DecisionTree, métricas |
| PyYAML | ≥6.0 | Configuración YAML |
| Streamlit | ≥1.35 | UI Web |
| Plotly | ≥5.20 | Gráficos |

## 📈 Tomar Decisiones: ¿Cuál Estrategia Usar?

| Caso | Recomendación | Razón |
|------|---------------|-------|
| Investigación pura | S7 (Per-Client F1+PCD) | Máxima diversidad + métrica equilibrada |
| Performance máximo | S4 (Global F1+PCD) | Ranking global + PCD |
| Velocidad máxima | S1 (Simple Pool) | Sin overhead de ranking |
| Datasets pequeños | S5-S7 | Garantiza todos los clientes aportan |
| Datasets grandes | S2-S4 | Más eficiente |

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'src'`
```bash
# Asegurar que estás en el directorio raíz
cd federated_proactive_forest
pip install -e .
```

### Error: Dataset no carga
- Verificar rutas en configuración
- Verificar que nombres de columnas coincidan
- Revisar [DATASETS_GUIDE.md](DATASETS_GUIDE.md)

### Streamlit lento
- Reducir `n_estimators` (100 → 50)
- Reducir `n_clients` (5 → 3)
- Usar `convergence=0.005` (más agresivo)

## 📝 Licencia

[Especificar licencia - MIT, Apache 2.0, etc.]

## 👤 Autor

- **Adrián Rodríguez** — Desarrollo, integración, UI Streamlit

**Base:** Proactive Forest (Cepero, 2023)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias: [correo/issue en GitHub]

---

**Hecho con ❤️ para investigación en Aprendizaje Federado**

## Estructura

```
federated-proactive-forest/
├── proactive_forest/        # Código original (Cepero 2023) — mín. modificaciones
├── src/
│   ├── domain/              # Lógica FL pura (sin FLEX, sin Streamlit)
│   ├── application/         # Orquestador de ronda
│   ├── infrastructure/      # Adaptadores de dataset
│   └── interfaces/
│       └── streamlit/       # UI independiente
├── configs/experiments/     # 7 YAMLs (uno por estrategia)
└── data/nslkdd/             # CSVs del dataset
```
