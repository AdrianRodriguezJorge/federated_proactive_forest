"""Constants and presets for the Streamlit interface."""

DATASET_PRESETS = {
    "Iris": {
        "file_path": "data/iris.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": (
            "150 muestras, 4 features, 3 clases (setosa, versicolor, virginica)"
        ),
    },
    "Letter": {
        "file_path": "data/letter.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "20,000 muestras, 16 features, 26 clases (A-Z)",
    },
    "Optdigits": {
        "file_path": "data/optdigits.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "5,620 muestras, 64 features (8x8 píxeles), 10 clases (0-9)",
    },
    "Spambase": {
        "file_path": "data/spambase.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "4,601 muestras, 57 features, 2 clases (spam/ham)",
    },
    "Nursery": {
        "file_path": "data/nursery.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [
            "parents",
            "has_nurs",
            "form",
            "children",
            "housing",
            "finance",
            "social",
            "health",
        ],
        "info": "12,960 muestras, 8 features categóricas, 5 clases",
    },
    "Sonar": {
        "file_path": "data/sonar.csv",
        "target_column": "Class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "208 muestras, 60 features, 2 clases (Rock/Mine)",
    },
    "Vowel": {
        "file_path": "data/vowel.csv",
        "target_column": "Class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "990 muestras, 10 features, 11 clases",
    },
    "Car": {
        "file_path": "data/car.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
        ],
        "info": (
            "1,728 muestras, 6 features categóricas, 4 clases "
            "(unacc, acc, good, vgood)"
        ),
    },
    "Glass": {
        "file_path": "data/glass.csv",
        "target_column": "Type",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "214 muestras, 9 features, 6 clases (tipos de vidrio)",
    },
    "Molecular": {
        "file_path": "data/molecular.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [
            "p-50", "p-49", "p-48", "p-47", "p-46", "p-45", "p-44", "p-43",
            "p-42", "p-41", "p-40", "p-39", "p-38", "p-37", "p-36", "p-35",
            "p-34", "p-33", "p-32", "p-31", "p-30", "p-29", "p-28", "p-27",
            "p-26", "p-25", "p-24", "p-23", "p-22", "p-21", "p-20", "p-19",
            "p-18", "p-17", "p-16", "p-15", "p-14", "p-13", "p-12", "p-11",
            "p-10", "p-9", "p-8", "p-7", "p-6", "p-5", "p-4", "p-3",
            "p-2", "p-1", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
        ],
        "info": "106 muestras, 57 features categóricas (promotores de E. coli), 2 clases",
    },
    "Pendigits": {
        "file_path": "data/pendigits.csv",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "10,992 muestras, 16 features (coordenadas de trazos), 10 clases (0-9)",
    },
    "CSV personalizado": {
        "file_path": "",
        "target_column": "class",
        "test_size": 0.15,
        "scale": True,
        "sep": ",",
        "categorical_columns": [],
        "info": "Carga tu propio archivo CSV",
    },
}

STRATEGY_LABELS = {
    "s1_simple_pool": "S1 — Simple Pool (todos los árboles, sin ordenar)",
    "s2_global_accuracy": "S2 — Global, orden por Accuracy + Progressive",
    "s3_global_f1": "S3 — Global, orden por Macro-F1 + Progressive",
    "s4_global_f1_pcd": (
        "S4 — Global, orden por α·F1 + β·PCD + Progressive"
    ),
    "s5_perclient_accuracy": (
        "S5 — Per-Client, orden por Accuracy + Progressive"
    ),
    "s6_perclient_f1": "S6 — Per-Client, orden por Macro-F1 + Progressive",
    "s7_perclient_f1_pcd": (
        "S7 — Per-Client, orden por α·F1 + β·PCD + Progressive"
    ),
    "s8_roulette": (
        "S8 — Ruleta Global de Atributos (intercambio de vectores "
        "de probabilidad)"
    ),
}

S8_VARIANT_LABELS = {
    "S8_MEAN": "Media Simple (democrática)",
    "S8_WEIGHTED": "Promedio Ponderado (por tamaño de dataset)",
    "S8_MEDIAN": "Mediana (robusta contra outliers)",
    "S8_CONSENSUS": "Consenso (ponderado por F1 local)",
    "S8_PROACTIVE_PCD": "Proactivo PCD (ponderado por diversidad Cepero)",
}
