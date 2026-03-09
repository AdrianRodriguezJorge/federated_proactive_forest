# 📊 Guía de Adaptadores de Datasets

## 1. Análisis del Dataset Iris

### Estructura del archivo `data/iris.csv`
```
sepallength,sepalwidth,petallength,petalwidth,class
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
```

**Características:**
| Aspecto | Valor |
|--------|-------|
| **Muestras totales** | 150 |
| **Características numéricas** | 4 (sepal_length, sepal_width, petal_length, petal_width) |
| **Características categóricas** | 0 (todas numéricas) |
| **Número de clases** | 3 (Iris-setosa, Iris-versicolor, Iris-virginica) |
| **Valores faltantes** | No |
| **Distribución** | Balanceada (50 muestras por clase) |
| **Requires separate train/test files?** | No (split automático 70-30) |

---

## 2. Cambios Realizados

### A. Nuevo archivo: `src/infrastructure/dataset/iris_adapter.py`

**Qué es un adaptador:**
- Clase que implementa la interfaz `IDatasetAdapter`
- Responsable de cargar, procesar, escalar y dividir datos
- Retorna un `DatasetSplit` (X_train, X_test, y_train, y_test, class_names)

**Componentes clave del IrisAdapter:**

```python
class IrisAdapter(IDatasetAdapter):
    # 1. Definir columnas y características
    COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    # 2. Constructor: configurar parámetros
    def __init__(self, data_path, scale=True, scaler_type="standard", 
                 train_test_split_ratio=0.7):
        # data_path: ruta al CSV
        # scale: normalizar o no
        # scaler_type: StandardScaler (media=0, var=1) o MinMaxScaler (0-1)
        # train_test_split_ratio: proporción train (70% default)
        ...
    
    # 3. Propiedades requeridas
    @property
    def name(self) -> str:
        return "iris"  # Identificador único
    
    @property
    def n_classes(self) -> int:
        return len(self._class_names_)
    
    # 4. Método principal: load()
    def load(self) -> DatasetSplit:
        # a) Leer CSV
        df = pd.read_csv(self.data_path)
        
        # b) Separar features y clases
        X = df[FEAT_COLS].values
        y = df["class"].values
        
        # c) Codificar etiquetas (strings → ints)
        #    "Iris-setosa" → 0
        #    "Iris-versicolor" → 1
        #    "Iris-virginica" → 2
        y_encoded = LabelEncoder().fit_transform(y)
        
        # d) Train-test split (70-30)
        X_train, X_test, y_train, y_test = train_test_split(...)
        
        # e) Escalar features (opcional)
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().transform(X_test)
        
        # f) Retornar DatasetSplit
        return DatasetSplit(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=FEAT_COLS,
            class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
            dataset_name="iris"
        )
```

### B. Modificación: `src/interfaces/streamlit/pages/page_config.py`

**Cambio 1: Selector de datasets**
```python
# ANTES:
dataset_type = st.selectbox("Tipo de dataset",
                           ["NSL-KDD", "CSV personalizado"])

# DESPUÉS:
dataset_type = st.selectbox("Tipo de dataset",
                           ["NSL-KDD", "Iris", "CSV personalizado"])
```

**Cambio 2: Lógica condicional para Iris**
```python
if dataset_type == "NSL-KDD":
    # Configuración NSL-KDD
    train_path = "data/NSL-KDD_train.csv"
    test_path = "data/NSL-KDD_test.csv"
    ...
elif dataset_type == "Iris":
    # NUEVO: Configuración automática para Iris
    train_path = "data/iris.csv"
    test_path = None  # Sin split separado
    target_col = "class"
    cat_features = []  # Sin categóricas
    csv_name = "iris"
    st.info("📊 Iris: 150 muestras, 4 features, 3 clases. Split automático 70-30.")
else:
    # CSV personalizado
    ...
```

**Cambio 3: Fábrica de adaptadores**
```python
def _load_dataset(cfg):
    d = cfg["dataset"]
    if d["type"] == "NSL-KDD":
        adapter = NslKddAdapter(...)
    elif d["type"] == "Iris":
        # NUEVO: Instanciar IrisAdapter
        from src.infrastructure.dataset.iris_adapter import IrisAdapter
        adapter = IrisAdapter(
            data_path=d["train_path"],
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
            train_test_split_ratio=0.7
        )
    else:
        adapter = GenericCsvAdapter(...)
    
    return adapter.load()
```

---

## 3. Patrón General para Agregar Nuevos Datasets

### Paso 1: Analizar el dataset
```
✓ Número de muestras
✓ Número y tipos de características (numéricas/categóricas)
✓ Número de clases
✓ Formato del archivo (CSV, JSON, otro)
✓ ¿Tiene separación train/test predefinida?
✓ ¿Column names? ¿Encoding especial?
```

### Paso 2: Crear el adaptador
```python
# Archivo: src/infrastructure/dataset/NOMBRE_adapter.py

from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class NOMBREAdapter(IDatasetAdapter):
    """
    Adaptador para DATASET_NAME.
    
    Características:
    - X muestras
    - Y features (Z numéricas, W categóricas)
    - C clases
    """
    
    COLUMNS = [...]  # Lista de nombres de columnas
    FEAT_COLS = [...]  # Features (sin clase)
    CAT_COLS = [...]  # Categóricas (si las hay)
    
    def __init__(self, train_path, test_path=None, scale=True, scaler_type="standard"):
        self.train_path = train_path
        self.test_path = test_path
        self.scale = scale
        self.scaler_type = scaler_type
        
        if scale:
            self._scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        else:
            self._scaler = None
        
        self._label_encoder = LabelEncoder()
        self._categorical_encoders = {}  # Si hay categóricas
        self._class_names_ = []
    
    @property
    def name(self) -> str:
        return "nombre_dataset"  # ID único
    
    @property
    def n_classes(self) -> int:
        return len(self._class_names_)
    
    def load(self) -> DatasetSplit:
        # 1. Leer datos
        train_df = pd.read_csv(self.train_path)
        if self.test_path:
            test_df = pd.read_csv(self.test_path)
        else:
            train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # 2. Codificar categóricas (si las hay)
        for cat_col in self.CAT_COLS:
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            train_df[cat_col] = encoder.fit_transform(train_df[[cat_col]])
            test_df[cat_col] = encoder.transform(test_df[[cat_col]])
            self._categorical_encoders[cat_col] = encoder
        
        # 3. Extraer X e y
        X_train = train_df[self.FEAT_COLS].values.astype(np.float64)
        X_test = test_df[self.FEAT_COLS].values.astype(np.float64)
        y_train = self._label_encoder.fit_transform(train_df["class"].values)
        y_test = self._label_encoder.transform(test_df["class"].values)
        
        # 4. Escalar
        if self._scaler:
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)
        
        # 5. Retornar
        self._class_names_ = self._label_encoder.classes_.tolist()
        return DatasetSplit(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=self.FEAT_COLS,
            class_names=self._class_names_,
            dataset_name=self.name
        )
```

### Paso 3: Actualizar `page_config.py`

**3a) Agregar a selector:**
```python
dataset_type = st.selectbox(
    "Tipo de dataset",
    ["NSL-KDD", "Iris", "MI_NUEVO_DATASET", "CSV personalizado"]
)
```

**3b) Agregar rama condicional:**
```python
elif dataset_type == "MI_NUEVO_DATASET":
    train_path = "data/ARCHIVO_TRAIN.csv"
    test_path = "data/ARCHIVO_TEST.csv"  # O None
    target_col = "clase"
    cat_features = ["col_cat1", "col_cat2"]  # O []
    csv_name = "mi_nuevo_dataset"
    st.info("📊 Dataset: N muestras, M features, K clases.")
```

**3c) Actualizar `_load_dataset()`:**
```python
elif d["type"] == "MI_NUEVO_DATASET":
    from src.infrastructure.dataset.mi_adapter import MiAdapter
    adapter = MiAdapter(
        train_path=d["train_path"],
        test_path=d.get("test_path"),
        scale=d.get("scale", True),
        scaler_type=d.get("scaler_type", "standard"),
    )
```

### Paso 4: Agregar archivos de datos
```
data/
├── iris.csv              # ← Ya existe
├── NSL-KDD_train.csv     # ← Ya existe
├── NSL-KDD_test.csv      # ← Ya existe
└── mi_nuevo_dataset_train.csv  # ← Agregar
    mi_nuevo_dataset_test.csv   # ← Agregar
```

---

## 4. Checklist para Agregar Datasets

- [ ] Analizar estructura del dataset
- [ ] Crear `src/infrastructure/dataset/NOMBRE_adapter.py`
  - [ ] Definir COLUMNS, FEAT_COLS, CAT_COLS
  - [ ] Implementar `__init__`
  - [ ] Implementar propiedades `name`, `n_classes`
  - [ ] Implementar `load()` retornando `DatasetSplit`
- [ ] Copiar CSV a carpeta `data/`
- [ ] Actualizar `page_config.py`:
  - [ ] Agregar opción al selectbox
  - [ ] Agregar rama `elif` en lógica condicional
  - [ ] Agregar rama `elif` en `_load_dataset()`
- [ ] Probar en Streamlit:
  - [ ] Seleccionar dataset
  - [ ] Guardar configuración
  - [ ] Verificar cargas sin errores
  - [ ] Ejecutar 1 ronda FL

---

## 5. Diferencias Clave: NSL-KDD vs Iris vs CSV Genérico

### NSL-KDD
```python
class NslKddAdapter:
    - Entrada: train_path + test_path (archivos separados)
    - Características: 42 numéricas + 3 categóricas
    - Clases: 5
    - Especificidad: Codificación categórica con OrdinalEncoder
    - Sin split automático
```

### Iris
```python
class IrisAdapter:
    - Entrada: data_path (archivo único)
    - Características: 4 numéricas, 0 categóricas
    - Clases: 3
    - Especificidad: Split automático 70-30 con stratify
    - Sin codificación categórica
```

### CSV Genérico
```python
class GenericCsvAdapter:
    - Entrada: train_path + test_path (o solo train con split)
    - Características: Configurable (usuario especifica)
    - Clases: Configurable
    - Especificidad: Flexible, adaptable a cualquier CSV
    - Codificación categórica dinámica
```

---

## 6. Casos de Uso Reales

### Caso 1: Agregar dataset Wine (similar a Iris)
```
wine.csv: 178 muestras, 13 features numéricos, 3 clases
→ Crear WineAdapter (muy similar a IrisAdapter)
```

### Caso 2: Agregar dataset personalizado con categóricas
```
ventas.csv: 10k muestras,
  - Numéricas: precio, cantidad, descuento
  - Categóricas: región, producto, cliente
  - Clases: rentable/no rentable
→ Crear VentasAdapter (similar a NSL-KDD)
```

### Caso 3: Agregar dataset con archivos separados
```
datos_train.csv: 8k muestras
datos_test.csv: 2k muestras
→ Crear CustomAdapter (configurar en page_config.py)
```

---

## 7. Prueba Rápida del Iris

```bash
# 1. Asegurar que data/iris.csv existe
ls data/iris.csv

# 2. Ejecutar Streamlit
streamlit run src/interfaces/streamlit/app.py

# 3. En UI:
#    - ⚙️ Configuración
#    - Seleccionar: "Iris"
#    - Guardar configuración
#    - ▶️ Ejecutar
#    - Ver métricas
```

**Tiempo esperado (Iris 70 muestras train, CPF 100 árboles, 3 clientes):**
- Entrenamiento: 1-2 min (menos que NSL-KDD)
- Agregación: 10 seg
- Total: ~2 minutos

---

## 8. Troubleshooting

### Error: `FileNotFoundError: No such file 'data/iris.csv'`
```
✓ Copiar iris.csv a carpeta data/
✓ Verificar ruta en page_config.py: train_path = "data/iris.csv"
```

### Error: `KeyError: 'sepal_length'`
```
✓ Verificar nombres exactos de columnas en CSV
✓ Verificar COLUMNS y FEAT_COLS en IrisAdapter
✓ Usar: df.columns.tolist() para ver columnas reales
```

### Error: `ValueError: invalid literal for int()`
```
✓ Problema en codificación de etiquetas
✓ Verificar que y (target) contiene strings o ints, no NaN
✓ Usar LabelEncoder() en lugar de .astype(int)
```

---

## 9. Resumen de Cambios

| Archivo | Cambio |
|--------|--------|
| `src/infrastructure/dataset/iris_adapter.py` | ✨ CREADO |
| `src/interfaces/streamlit/pages/page_config.py` | ✏️ MODIFICADO (3 cambios) |
| `data/iris.csv` | ✓ YA EXISTE |

**Total de líneas agregadas:** ~120 (adapter) + 20 (page_config) = 140

---

## 10. Próximos Pasos

1. **Agregar más datasets:**
   - Wine Dataset (sklearn)
   - Heart Disease (UCI ML)
   - Breast Cancer (sklearn)

2. **Mejorar adaptadores:**
   - Manejar valores faltantes (NaN)
   - Detectar automáticamente tipos de columnas
   - Balancear clases desbalanceadas

3. **Paralelización:**
   - Entrenar clientes en paralelo
   - Reducir tiempo total de FL

---

**Fin de la guía. Para dudas, revisar código del IrisAdapter y comparar con NslKddAdapter.**
