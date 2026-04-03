"""Test explícito de múltiples rondas Progressive Windows"""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar datos
df = pd.read_csv('data/students_dropout.csv', sep=';')
X = df[[c for c in df.columns if c != 'Target']].values.astype(np.float64)
y = LabelEncoder().fit_transform(df['Target'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from src.domain.aggregation.strategies.progressive_windows import ProgressiveWindowsStrategy

# Simular datos de clientes
np.random.seed(42)
n_clients = 3
n_trees_per_client = 30  # 30 árboles por cliente
window_size = 3
max_rounds = 5

# Crear árboles dummy (usamos accuracy como proxy)
client_trees = {}
client_metadata = {}
for i in range(n_clients):
    client_id = f'client_{i}'
    # Crear árboles con accuracy variable
    trees = [{'id': j, 'accuracy': 0.7 + np.random.random() * 0.2} for j in range(n_trees_per_client)]
    client_trees[client_id] = trees
    client_metadata[client_id] = type('obj', (object,), {
        'client_id': client_id,
        'n_trees': len(trees),
        'accuracy': np.mean([t['accuracy'] for t in trees]),
        'macro_f1': np.mean([t['accuracy'] for t in trees]) * 0.95,
        'pcd': 0.3 + np.random.random() * 0.4,
        'to_dict': lambda self=client_id, trees=trees: {
            'client_id': self,
            'n_trees': len(trees),
            'accuracy': np.mean([t['accuracy'] for t in trees]),
            'macro_f1': np.mean([t['accuracy'] for t in trees]) * 0.95,
            'pcd': 0.3 + np.random.random() * 0.4,
        }
    })()

# Crear datos de validación
X_val = X_test_scaled[:100]
y_val = y_test[:100]

# Ejecutar Progressive Windows
strategy = ProgressiveWindowsStrategy(
    window_size=window_size,
    max_rounds=max_rounds,
    alpha=0.5,
    convergence_threshold=0.002,
    verbose=True
)

print("\n" + "="*100)
print("TEST EXPLÍCITO DE MÚLTIPLES RONDAS PROGRESSIVE WINDOWS")
print("="*100)
print(f"Clientes: {n_clients}")
print(f"Árboles por cliente: {n_trees_per_client}")
print(f"Tamaño de ventana: {window_size}")
print(f"Máximo de rondas: {max_rounds}")
print("="*100)

global_trees, selected_ids, all_entries = strategy.aggregate(
    client_trees,
    client_metadata,
    X_val=X_val,
    y_val=y_val,
    window_size=window_size,
    max_rounds=max_rounds,
    alpha=0.5
)

print("\n" + "="*100)
print("RESULTADOS")
print("="*100)
print(f"Rondas completadas: {strategy.convergence_round}")
print(f"Árboles en bosque global: {len(global_trees)}")
print(f"Árboles seleccionados por cliente:")
for cid, indices in selected_ids.items():
    print(f"  {cid}: {len(indices)} árboles")
print("="*100)
