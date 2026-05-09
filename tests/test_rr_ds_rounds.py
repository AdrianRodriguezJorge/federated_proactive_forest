import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar datos
df = pd.read_csv('data/iris.csv', sep=',')
X = df[[c for c in df.columns if c != 'class']].values.astype(np.float64)
y = LabelEncoder().fit_transform(df['class'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from src.application.orchestrators import FLEXOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit

config = {
    'n_clients': 3,
    'distribution': 'iid',
    'strategy': 'PW',
    'n_estimators': 20,
    'alpha_pf': 0.1,
    'aggregation': {
        'strategy': 'pw',
        'window_size': 3,
        'max_rounds': 5,
        'f1_weight': 0.5,
        'convergence_threshold': 0.002
    },
    'prediction': {'local_weight': 0.5, 'global_weight': 0.5},
    'verbose': True
}

dataset_split = DatasetSplit(
    X_train=X_train_scaled, X_test=X_test_scaled,
    y_train=y_train, y_test=y_test,
    feature_names=list(df.columns[:-1]),
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    dataset_name='Iris'
)

orchestrator = FLEXOrchestrator.from_config(config)
orchestrator.setup_federation(dataset_split, seed=42)
results = orchestrator.run_federated_round()

print('\n' + '='*80)
print('RESULTADOS FINALES')
print('='*80)
print(f'Rondas completadas: {results.num_rounds}')
print(f'Convergencia: {results.convergence_round}')
print(f'Árboles globales: {results.n_trees_global}')
print(f'Accuracy: {results.global_accuracy:.4f}')
