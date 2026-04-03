"""
Test rápido de Inferencia Híbrida - Verificación de configuración.
"""

import numpy as np
import sys
sys.path.append('.')

from src.domain.prediction.hybrid_predictor import HybridPredictor

print("=" * 80)
print("🧪 TEST RÁPIDO DE INFERENCIA HÍBRIDA")
print("=" * 80)

# Crear datos sintéticos simples
np.random.seed(42)
n_samples = 100
n_features = 10
n_classes = 3

X_test = np.random.randn(n_samples, n_features)
y_test = np.random.randint(0, n_classes, n_samples)

class_names = ['Class_0', 'Class_1', 'Class_2']

# Crear árboles mock simples
class MockTree:
    def __init__(self, predictions):
        self.predictions = predictions
    
    def predict(self, x):
        # Para un sample dado, devolver predicción basada en el índice
        idx = np.random.randint(0, len(self.predictions))
        return self.predictions[idx]

# Crear árboles "locales" y "globales" con diferentes patrones de predicción
local_trees = [MockTree(np.random.randint(0, n_classes, 10)) for _ in range(5)]
global_trees = [MockTree(np.random.randint(0, n_classes, 10)) for _ in range(3)]

print(f"\n✅ Datos de test: {n_samples} muestras, {n_classes} clases")
print(f"✅ Árboles locales: {len(local_trees)}")
print(f"✅ Árboles globales: {len(global_trees)}")

# Probar diferentes configuraciones de pesos híbridos
print("\n🔄 Probando diferentes configuraciones de pesos híbridos:")
print("-" * 80)

test_configs = [
    (0.0, 1.0, "100% Global"),
    (0.2, 0.8, "20% Local + 80% Global"),
    (0.4, 0.6, "40% Local + 60% Global (default)"),
    (0.5, 0.5, "50% Local + 50% Global (balanceado)"),
    (0.6, 0.4, "60% Local + 40% Global"),
    (0.8, 0.2, "80% Local + 20% Global"),
    (1.0, 0.0, "100% Local"),
]

results = []
for local_w, global_w, description in test_configs:
    predictor = HybridPredictor(
        local_weight=local_w,
        global_weight=global_w,
        n_classes=n_classes,
        class_names=class_names
    )
    
    np.random.seed(42)  # Para reproducibilidad
    hybrid_preds = predictor.predict(X_test, local_trees, global_trees)
    
    # Calcular accuracy aleatorio (solo para demostración)
    acc = np.mean(hybrid_preds == y_test)
    
    results.append({
        'local_w': local_w,
        'global_w': global_w,
        'description': description,
        'accuracy': acc,
        'unique_predictions': len(np.unique(hybrid_preds))
    })
    
    print(f"   {description:40s} → Accuracy={acc:.4f}, Unique preds={len(np.unique(hybrid_preds))}")

# Verificar que HybridPredictor respeta los pesos
print("\n" + "=" * 80)
print("🔍 VERIFICACIÓN INTERNA DE HYBRIDPREDICTOR")
print("=" * 80)

for local_w, global_w, _ in test_configs[:3]:
    predictor = HybridPredictor(local_weight=local_w, global_weight=global_w, 
                                n_classes=n_classes, class_names=class_names)
    print(f"\n   Configuración: local_weight={local_w}, global_weight={global_w}")
    print(f"   - Atributo lw: {predictor.lw}")
    print(f"   - Atributo gw: {predictor.gw}")
    print(f"   - Suma: {predictor.lw + predictor.gw:.6f}")
    
    if abs(predictor.lw + predictor.gw - 1.0) < 1e-6:
        print(f"   ✅ Los pesos suman 1.0 correctamente")
    else:
        print(f"   ❌ ERROR: Los pesos no suman 1.0")

# Verificar que la configuración se lee correctamente
print("\n" + "=" * 80)
print("📊 VERIFICACIÓN DE CONFIGURACIÓN DESDE DICCIONARIO")
print("=" * 80)

# Simular configuración como vendría de Streamlit
config_examples = [
    {
        'name': 'Config por defecto',
        'config': {'prediction': {'local_weight': 0.4, 'global_weight': 0.6}}
    },
    {
        'name': 'Config balanceada',
        'config': {'prediction': {'local_weight': 0.5, 'global_weight': 0.5}}
    },
    {
        'name': 'Config PW',
        'config': {'prediction': {'local_weight': 0.5, 'global_weight': 0.5}}
    },
    {
        'name': 'Sin configuración (defaults)',
        'config': {}
    }
]

for example in config_examples:
    cfg = example['config']
    local_weight = cfg.get('prediction', {}).get('local_weight', 0.4)
    global_weight = cfg.get('prediction', {}).get('global_weight', 0.6)
    
    print(f"\n   {example['name']}:")
    print(f"   - local_weight leído: {local_weight}")
    print(f"   - global_weight leído: {global_weight}")
    
    # Crear predictor con esta configuración
    predictor = HybridPredictor(
        local_weight=local_weight,
        global_weight=global_weight,
        n_classes=n_classes,
        class_names=class_names
    )
    
    if abs(predictor.lw - local_weight) < 1e-6 and abs(predictor.gw - global_weight) < 1e-6:
        print(f"   ✅ Configuración aplicada correctamente")
    else:
        print(f"   ❌ ERROR: Configuración no aplicada")

print("\n" + "=" * 80)
print("✅ TEST COMPLETADO")
print("=" * 80)
