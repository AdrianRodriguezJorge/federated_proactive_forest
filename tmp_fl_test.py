import os, sys

root = r"c:\Users\Adrián Rodríguez\Documents\! Study 📝\📊 KDD 🤖🧠\! Federated learning\federated_proactive_forest"
os.chdir(root)
sys.path.append('src')

from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter
from src.application.fl_orchestrator import FLEXOrchestrator

print('Loading iris...')
adapter = GenericCsvAdapter(name='iris', train_path='data/iris.csv', target_column='class', scale=True, scaler_type='standard')
dataset = adapter.load()
config = {
    'federation': {'n_clients': 3, 'distribution': 'iid'},
    'model': {'n_estimators': 10, 'alpha': 0.1},
    'aggregation': {'strategy': 'S1', 't_max': 10},
    'verbose': False,
    'seed': 42
}

orch = FLEXOrchestrator.from_config(config)
orch.setup_federation(dataset, seed=42)
res = orch.run_federated_round()
print('done', res.global_accuracy, res.global_macro_f1)
