"""FLEX primitive for training Proactive Forest on clients."""
from typing import Dict, Any


def init_server_model_pf():
    """
    Initialize server model for Proactive Forest.
    FLEX primitive function.
    """
    from src.domain.model.proactive_forest import ProactiveForest
    return {
        'model': ProactiveForest(),
        'trees': []
    }


def deploy_server_config_pf(server_flex_model, client_flex_model):
    """
    Deploy server configuration to client.
    """
    # Copy configuration from server to client
    client_flex_model['config'] = server_flex_model.get('config', {})


def deploy_server_model_pf(server_flex_model, client_flex_model):
    """
    Deploy server model to client.
    """
    client_flex_model['global_model'] = server_flex_model['model']
    client_flex_model['global_trees'] = server_flex_model.get('trees', [])


def train_pf(client_flex_model):
    """
    Train Proactive Forest on client data.
    FLEX decorator-compatible function.
    """
    from src.application.commands.train_command import TrainCommand
    from src.domain.model.proactive_forest import ProactiveForest

    train_cmd = TrainCommand(lambda: ProactiveForest(
        n_estimators=client_flex_model.get('config', {}).get('n_estimators', 100),
        alpha=client_flex_model.get('config', {}).get('alpha', 0.5)
    ))

    # Assume data is already in client_flex_model
    updated_data = train_cmd.execute_on_client(client_flex_model)
    client_flex_model.update(updated_data)


def collect_clients_trees_pf(server_flex_model, clients_flex_models):
    """
    Collect trees from all clients.
    """
    all_client_trees = []
    client_metadata = {}

    for client_id, client_model in clients_flex_models.items():
        trees = client_model.get('trees', [])
        all_client_trees.append(trees)
        client_metadata[client_id] = {
            'n_trees': len(trees),
            'accuracy': 0.8  # Placeholder
        }

    server_flex_model['all_client_trees'] = all_client_trees
    server_flex_model['client_metadata'] = client_metadata


def aggregate_trees_from_pf(server_flex_model):
    """
    Aggregate trees using the configured strategy.
    """
    from src.application.commands.aggregate_command import AggregateCommand
    from src.domain.aggregation.aggregation_factory import AggregationFactory

    all_client_trees = server_flex_model.get('all_client_trees', [])
    client_metadata = server_flex_model.get('client_metadata', {})

    # Convert to expected format
    client_trees_dict = {f'client_{i}': trees for i, trees in enumerate(all_client_trees)}

    strategy_name = server_flex_model.get('config', {}).get('strategy', 'S1')
    strategy = AggregationFactory.create_strategy(strategy_name)

    aggregate_cmd = AggregateCommand(strategy)
    global_data = aggregate_cmd.execute(client_trees_dict, client_metadata)

    server_flex_model['global_trees'] = global_data.get('global_trees', [])
    server_flex_model['selected_indices'] = global_data.get('selected_indices', {})


def set_aggregated_trees_pf(server_flex_model):
    """
    Set aggregated trees as the server model.
    """
    from src.domain.model.proactive_forest import ProactiveForest

    global_trees = server_flex_model.get('global_trees', [])
    server_flex_model['model'] = ProactiveForest.from_trees(global_trees)
    server_flex_model['trees'] = global_trees


def evaluate_global_pf_model(server_flex_model, test_data):
    """
    Evaluate global model on test data.
    """
    model = server_flex_model.get('model')
    if model and test_data is not None:
        X_test, y_test = test_data[:, :-1], test_data[:, -1]
        predictions = model.predict(X_test)

        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')

        server_flex_model['global_accuracy'] = accuracy
        server_flex_model['global_f1'] = f1


def evaluate_global_pf_model_at_clients(client_flex_model):
    """
    Evaluate global model at client side.
    """
    global_model = client_flex_model.get('global_model')
    if global_model:
        X_test = client_flex_model.get('X_test')
        y_test = client_flex_model.get('y_test')

        if X_test is not None and y_test is not None:
            predictions = global_model.predict(X_test)

            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')

            client_flex_model['global_accuracy'] = accuracy
            client_flex_model['global_f1'] = f1


def evaluate_local_pf_model_at_clients(client_flex_model):
    """
    Evaluate local model at client side.
    """
    local_model = client_flex_model.get('model')
    if local_model:
        X_test = client_flex_model.get('X_test')
        y_test = client_flex_model.get('y_test')

        if X_test is not None and y_test is not None:
            predictions = local_model.predict(X_test)

            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')

            client_flex_model['local_accuracy'] = accuracy
            client_flex_model['local_f1'] = f1