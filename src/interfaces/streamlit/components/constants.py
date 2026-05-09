"""Constants and presets for the Streamlit interface."""

DATASET_PRESETS = {
    "Iris": {
        "file_path": "data/iris.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "150 muestras, 4 features, 3 clases (setosa, versicolor, virginica)",
    },
    "Letter": {
        "file_path": "data/letter.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "20,000 muestras, 16 features, 26 clases (A-Z)",
    },
    "Optdigits": {
        "file_path": "data/optdigits.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "5,620 muestras, 64 features (8x8 píxeles), 10 clases (0-9)",
    },
    "Spambase": {
        "file_path": "data/spambase.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "4,601 muestras, 57 features, 2 clases (spam/ham)",
    },
    "Nursery": {
        "file_path": "data/nursery.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"],
        "info": "12,960 muestras, 8 features categóricas, 5 clases",
    },
    "Sonar": {
        "file_path": "data/sonar.csv", "target_column": "Class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "208 muestras, 60 features, 2 clases (Rock/Mine)",
    },
    "Vowel": {
        "file_path": "data/vowel.csv", "target_column": "Class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "990 muestras, 10 features, 11 clases",
    },
    "Car": {
        "file_path": "data/car.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": ["buying", "maint", "doors", "persons", "lug_boot", "safety"],
        "info": "1,728 muestras, 6 features categóricas, 4 clases (unacc, acc, good, vgood)",
    },

    "CSV personalizado": {
        "file_path": "", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",",
        "categorical_columns": [],
        "info": "Carga tu propio archivo CSV",
    },
}

STRATEGY_LABELS = {
    "s1_simple_pool":        "S1 — Simple Pool (todos los árboles, sin ordenar)",
    "s2_global_accuracy":    "S2 — Global, orden por Accuracy + Progressive",
    "s3_global_f1":          "S3 — Global, orden por Macro-F1 + Progressive",
    "s4_global_f1_pcd":      "S4 — Global, orden por α·F1 + β·PCD + Progressive",
    "s5_perclient_accuracy": "S5 — Per-Client, orden por Accuracy + Progressive",
    "s6_perclient_f1":       "S6 — Per-Client, orden por Macro-F1 + Progressive",
    "s7_perclient_f1_pcd":   "S7 — Per-Client, orden por α·F1 + β·PCD + Progressive",
    "pw":                    "PW — Progressive Windows (ventanas + score dinámico F1+Diversidad)",
    "s9_roulette":           "S9 — Ruleta Global de Atributos (intercambio de vectores de probabilidad)",
}

S9_VARIANT_LABELS = {
    "S9_MEAN":      "Media Simple (democrática)",
    "S9_WEIGHTED":  "Promedio Ponderado (por tamaño de dataset)",
    "S9_MEDIAN":    "Mediana (robusta contra outliers)",
    "S9_CONSENSUS": "Consenso (ponderado por F1 local)",
    "S9_PROACTIVE_PCD": "Proactivo PCD (ponderado por diversidad Cepero)",
}
