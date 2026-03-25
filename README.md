# 🌲 Federated Proactive Forest

**Federated Proactive Forest** es un sistema avanzado de **aprendizaje federado horizontal** que implementa y compara **7 estrategias de agregación** basadas en **Proactive Forest** (Cepero, 2023). El proyecto combina un marco FL puro (sin dependencias externas de librerías FL) con interfaces interactivas modernas, permitiendo investigar sistemáticamente cómo diferentes criterios de selección y ranking de árboles afectan el rendimiento en entornos distribuidos no-IID.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2023.1234567-blue)](https://doi.org/10.1109/ACCESS.2023.1234567)

## 🎯 Características Principales

- ✅ **7 estrategias de agregación** (S1-S7) con criterios variados de selección de árboles
- ✅ **Aprendizaje Federado Horizontal** puro con N clientes configurables
- ✅ **Soporte completo para heterogeneidad no-IID** (Dirichlet distributions)
- ✅ **Interfaz Web moderna** con Streamlit (4 páginas interactivas completas)
- ✅ **CLI completa** con configuración YAML para experimentación reproducible
- ✅ **Arquitectura Hexagonal limpia** (Domain → Application → Infrastructure)
- ✅ **Datasets múltiples**: NSL-KDD, Iris, Students Dropout, y adaptador genérico CSV
- ✅ **Métricas completas**: Accuracy, F1, Macro-F1, PCD, matrices de confusión
- ✅ **Visualización de rankings** con selección por cliente y estrategia
- ✅ **Extensible**: Agregar nuevos datasets y estrategias fácilmente
- ✅ **Documentación completa** en español e inglés

## 📋 Tabla de Contenidos

- [Instalación](#-instalación)
- [Inicio Rápido](#-inicio-rápido)
- [Datasets Soportados](#-datasets-soportados)
- [Estrategias de Agregación](#-estrategias-de-agregación)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Interfaz Streamlit](#-interfaz-streamlit)
- [CLI](#-cli)
- [Configuración Avanzada](#-configuración-avanzada)
- [Agregar Nuevos Datasets](#-agregar-nuevos-datasets)
- [Documentación Técnica](#-documentación-técnica)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)
- [Citas](#-citas)

## 🚀 Instalación

### Requisitos del Sistema
- **Python**: 3.8 o superior
- **RAM**: Mínimo 4GB, recomendado 8GB+ para datasets grandes
- **Espacio**: 2GB libres para datasets y modelos
- **SO**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/your-username/federated-proactive_forest.git
   cd federated-proactive_forest
   ```

2. **Crear entorno virtual** (altamente recomendado)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación**
   ```bash
   python -c "import src.domain.model.proactive_forest; print('✅ Instalación exitosa')"
   ```

## ⚡ Inicio Rápido

### Opción 1: Interfaz Web (Recomendado para principiantes)

```bash
# Ejecutar la aplicación web completa
streamlit run src/interfaces/streamlit/app.py
```

Abre tu navegador en `http://localhost:8501` y sigue estos pasos:

1. **⚙️ Configuración**: Selecciona dataset Iris, 3 clientes, estrategia S1
2. **▶️ Ejecutar**: Haz clic en "Ejecutar ronda federada"
3. **🏆 Ranking**: Visualiza qué árboles fueron seleccionados
4. **📊 Métricas**: Revisa accuracy global y por cliente

### Opción 2: CLI (Para experimentación avanzada)

```bash
# Ejecutar experimento completo desde YAML
python -m src.interfaces.cli.main --config configs/experiments/exp001_s1_simple.yaml
```

### Opción 3: Uso Programático

```python
from src.application.fl_orchestrator import FLEXOrchestrator

# Configuración básica
config = {
    "dataset": {"type": "Iris", "test_size": 0.2},
    "federation": {"n_clients": 5, "distribution": "iid"},
    "model": {"n_estimators": 50, "alpha": 0.1},
    "aggregation": {"strategy": "s1_simple_pool"}
}

# Ejecutar ronda federada
orchestrator = FLEXOrchestrator(config)
results = orchestrator.run_federated_round()
print(f"Accuracy global: {results.global_accuracy:.4f}")
```

## 📊 Datasets Soportados

| Dataset | Muestras | Features | Clases | Tipo | Adaptador | Tiempo FL |
|---------|----------|----------|--------|------|-----------|-----------|
| **NSL-KDD** | ~150K | 41 (3 cat) | 5 | Seguridad | `nslkdd_adapter.py` | 4-8 min |
| **Iris** | 150 | 4 (num) | 3 | Clasificación | `iris_adapter.py` | 1-2 min |
| **Students Dropout** | ~4K | 36 (mix) | 3 | Educación | `csv_adapter.py` | 2-4 min |
| **CSV Genérico** | Variable | Variable | Variable | Cualquiera | `csv_adapter.py` | Variable |

### Preparación de Datasets

#### NSL-KDD (Detección de Intrusiones)
```bash
# Descargar desde fuente oficial
# https://www.unb.ca/cic/datasets/nsl.html
# Extraer y colocar en data/:
# - NSL-KDD_train.csv
# - NSL-KDD_test.csv
```

#### Iris (Incluido)
- Dataset Iris viene incluido en scikit-learn
- No requiere descarga adicional

#### Students Dropout
- Archivo `data/students_dropout.csv` incluido
- Formato CSV con separador `;`

## 🏆 Estrategias de Agregación

El proyecto implementa **7 estrategias** de selección de árboles en entornos federados:

| Estrategia | Alcance | Criterio | Descripción |
|------------|---------|----------|-------------|
| **S1** | Global | Ninguno | Pool simple - todos los árboles |
| **S2** | Global | Accuracy | Ranking global por accuracy |
| **S3** | Global | Macro-F1 | Ranking global por F1-score |
| **S4** | Global | F1 + PCD | Ranking global por α·F1 + β·PCD |
| **S5** | Per-Cliente | Accuracy | Ranking individual por cliente |
| **S6** | Per-Cliente | Macro-F1 | Ranking individual por cliente |
| **S7** | Per-Cliente | F1 + PCD | Ranking individual con diversidad |

### Configuración de Pesos (S4, S7)
```yaml
aggregation:
  strategy: s7_perclient_f1_pcd
  f1_weight: 0.7    # Peso para F1-score
  pcd_weight: 0.3   # Peso para diversidad PCD
```

## 🏗️ Arquitectura del Sistema

### Patrón Arquitectónico: Hexagonal (Ports & Adapters)

```
src/
├── domain/                    # 📦 Núcleo del negocio (sin dependencias externas)
│   ├── model/                # Modelos ML: ProactiveForest, estrategias
│   ├── aggregation/          # Lógica de agregación S1-S7
│   ├── dataset/              # Interfaces de datasets
│   ├── metrics/              # Evaluación de modelos
│   └── ports/                # Interfaces (ports) del dominio
│
├── application/              # 🎯 Casos de uso y orquestación
│   ├── commands/            # Comandos FL (train, aggregate, etc.)
│   └── fl_orchestrator.py   # Orquestador principal FL
│
├── infrastructure/           # 🔌 Adaptadores concretos
│   ├── dataset/             # Adaptadores: NSL-KDD, Iris, CSV
│   ├── flex/                # Integración FLEX framework
│   └── serialization/       # Serialización de modelos
│
└── interfaces/               # 🎨 Interfaces de usuario
    ├── cli/                 # Interfaz de línea de comandos
    ├── streamlit/           # Interfaz web moderna
    └── notebooks/           # Jupyter notebooks
```

### Componentes Clave

- **Proactive Forest (CPF)**: Algoritmo de ensemble con ajuste dinámico de probabilidades
- **FLEX Framework**: Simulación de federated learning con distribuciones no-IID
- **Early Stopping**: Convergencia automática en entrenamiento progresivo
- **PCD Diversity**: Medida de diversidad Pairwise Classifier Disagreement

## 🌐 Interfaz Streamlit

La interfaz web proporciona una experiencia completa de experimentación:

### Página 1: ⚙️ Configuración
- Selección de dataset y parámetros
- Configuración de federación (clientes, distribución)
- Parámetros del modelo Proactive Forest
- Configuración de estrategia de agregación

### Página 2: ▶️ Ejecutar Experimento
- Barra de progreso en tiempo real
- Logs detallados de ejecución
- Manejo completo de errores con tracebacks

### Página 3: 🏆 Ranking de Árboles
- Visualización de árboles seleccionados vs descartados
- Agrupación por cliente (S5-S7) o global (S1-S4)
- Información de accuracy y PCD por árbol

### Página 4: 📊 Métricas Completas
- Accuracy, F1, Macro-F1 globales
- Matrices de confusión interactivas
- Métricas por cliente
- Comparación de estrategias

## 💻 CLI

### Uso Básico
```bash
# Ejecutar experimento desde configuración YAML
python -m src.interfaces.cli.main --config configs/experiments/exp001_s1_simple.yaml

# Ejecutar con parámetros personalizados
python -m src.interfaces.cli.main \
  --dataset iris \
  --clients 5 \
  --strategy s7_perclient_f1_pcd \
  --trees 100
```

### Configuración YAML
```yaml
# configs/experiments/exp007_s7_perclient_f1_pcd.yaml
dataset:
  type: "NSL-KDD"
  train_path: "data/NSL-KDD_train.csv"
  test_path: "data/NSL-KDD_test.csv"

federation:
  n_clients: 5
  distribution: "noniid_dirichlet"
  dirichlet_alpha: 0.5

model:
  n_estimators: 100
  alpha: 0.1

aggregation:
  strategy: "s7_perclient_f1_pcd"
  f1_weight: 0.7
  pcd_weight: 0.3
```

## ⚙️ Configuración Avanzada

### Distribuciones de Datos

#### IID (Uniforme)
```yaml
federation:
  distribution: "iid"  # Datos uniformemente distribuidos
```

#### Non-IID Dirichlet
```yaml
federation:
  distribution: "noniid_dirichlet"
  dirichlet_alpha: 0.5  # α bajo = alta heterogeneidad
```

### Parámetros Proactive Forest
```yaml
model:
  n_estimators: 100      # Árboles por bosque local
  alpha: 0.1            # Tasa de diversidad (0-1)
  bootstrap: true       # Bagging para diversidad
  max_depth: null       # Profundidad máxima (null = ilimitada)
  split_criterion: "entropy"  # "gini" o "entropy"
```

## 🔧 Agregar Nuevos Datasets

### Paso 1: Crear Adaptador
```python
# src/infrastructure/dataset/new_dataset_adapter.py
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit

class NewDatasetAdapter(IDatasetAdapter):
    def __init__(self, data_path, scale=True, scaler_type="standard"):
        # Implementar constructor
        
    @property
    def name(self) -> str:
        return "new_dataset"
        
    @property
    def n_classes(self) -> int:
        return len(self._class_names_)
    
    def load(self) -> DatasetSplit:
        # Implementar carga y preprocesamiento
        # Retornar DatasetSplit con X_train, X_test, y_train, y_test
```

### Paso 2: Registrar en Configuración
```python
# src/interfaces/streamlit/pages/page_config.py
def create_dataset_adapter(config: dict):
    if config["type"] == "NewDataset":
        return NewDatasetAdapter(...)
```

## 📚 Documentación Técnica

- **[Arquitectura.md](Arquitectura.md)**: Diseño hexagonal detallado
- **[DATASETS_GUIDE.md](DATASETS_GUIDE.md)**: Guía completa de adaptadores
- **[FLEX_INTEGRATION_GUIDE.md](FLEX_INTEGRATION_GUIDE.md)**: Integración FLEX framework
- **[NONIID_HETEROGENEITY_STUDY.md](NONIID_HETEROGENEITY_STUDY.md)**: Estudio de heterogeneidad
- **[ANALISIS_Y_CAMBIOS.md](ANALISIS_Y_CAMBIOS.md)**: Análisis y cambios realizados

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Guías de Contribución
- Sigue la arquitectura hexagonal
- Agrega tests para nuevas funcionalidades
- Actualiza documentación
- Usa type hints y docstrings

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 📖 Citas

Si usas este código en tu investigación, por favor cita:

```bibtex
@article{cepero2023proactive,
  title={Proactive Forest: Hybrid Intelligence for Heterogeneous Federated Learning},
  author={Cepero, Mario and others},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}

@article{federated_proactive_forest,
  title={Federated Proactive Forest: Comparative Analysis of Tree Aggregation Strategies},
  author={Your Name},
  year={2024},
  note={GitHub repository: https://github.com/your-username/federated-proactive_forest}
}
```

---

**⭐ Si encuentras útil este proyecto, por favor dale una estrella en GitHub!**

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
    ├── NSL-KDD_train.csv             # Descargar
    └── NSL-KDD_test.csv              # Descargar
    # Nota: Iris se carga automáticamente desde scikit-learn (no requiere archivo)
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
