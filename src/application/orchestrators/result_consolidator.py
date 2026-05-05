from typing import Dict, List, Any, Optional
import numpy as np
from src.domain.metadata.client_metadata import ClientMetadata
from src.domain.metrics.forest_evaluator import ForestEvaluator, ForestReport
from src.domain.prediction.hybrid_predictor import HybridPredictor
from src.domain.services.label_service import SimpleLabelService
from src.application.orchestrators.fl_results import FLResults

class ResultConsolidator:
    """
    Service responsible for consolidating federation results, generating global metrics,
    and applying hybrid predictions.
    """
    def __init__(self, label_service: SimpleLabelService, config: dict):
        self.label_service = label_service
        self.config = config

    def consolidate(self, 
                    strategy_name: str, 
                    flex_pool: Any, 
                    dataset_split: Any, 
                    server_eval: dict = None) -> FLResults:
        server_id = "server"
        server_model = flex_pool._models[server_id]
        
        if server_eval and server_id in server_eval:
            global_acc = server_eval[server_id].get('accuracy', server_model.get('global_accuracy', 0.0))
            global_f1 = server_eval[server_id].get('macro_f1', server_model.get('global_f1', 0.0))
        else:
            global_acc = server_model.get('global_accuracy', 0.0)
            global_f1 = server_model.get('global_f1', 0.0)
            
        global_trees = server_model.get('trees', [])
        
        client_ids = list(flex_pool.clients.actor_ids)
        client_accuracies = {}
        client_f1_scores = {}
        client_metadata = {}
        client_hybrid_predictions = {}
        client_hybrid_forest_sizes = {}

        X_test, y_test = dataset_split.X_test, dataset_split.y_test
        y_test_numeric = self.label_service.transform(y_test)
        class_names = self.label_service.classes
        
        local_weight = self.config.get('prediction', {}).get('local_weight', 0.4)
        predictor = HybridPredictor(
            local_weight=local_weight, 
            global_weight=1.0 - local_weight, 
            n_classes=len(class_names), 
            class_names=class_names,
            label_service=self.label_service
        )

        for cid in client_ids:
            client_model = flex_pool._models[cid]
            client_accuracies[cid] = client_model.get('global_accuracy', 0.0)
            client_f1_scores[cid] = client_model.get('global_f1', 0.0)
            
            meta_val = client_model.get('metadata')
            if isinstance(meta_val, ClientMetadata):
                meta = meta_val
            elif isinstance(meta_val, dict):
                meta = ClientMetadata(**meta_val)
            else:
                meta = ClientMetadata(
                    client_id=cid, 
                    n_trees=len(client_model.get('trees', [])), 
                    accuracy=client_model.get('local_accuracy', 0.0), 
                    macro_f1=client_model.get('local_f1', 0.0), 
                    pcd=0.0
                )
            
            selected_ids_raw = server_model.get('selected_ids', {})
            selected_ids_dict = {str(k): v for k, v in selected_ids_raw.items()}
            meta.selected_local_tree_ids = selected_ids_dict.get(str(cid), [])
            client_metadata[cid] = meta

            local_trees = client_model.get('trees', [])
            external_global_trees = [t for t in global_trees if not any(t is lt for lt in local_trees)]
            
            hybrid_preds = predictor.predict(X_test, local_trees, external_global_trees)
            client_hybrid_predictions[cid] = hybrid_preds
            client_hybrid_forest_sizes[cid] = len(local_trees) + len(external_global_trees)

        global_model = server_model.get('model')
        if global_model:
            global_preds = global_model.predict(X_test)
            try:
                global_pcd = float(global_model.diversity_measure(X_test, y_test, diversity='pcd'))
            except:
                global_pcd = 0.0
                
            global_report = ForestEvaluator.evaluate_from_predictions(
                global_preds, y_test_numeric, class_names, len(global_trees), pcd=global_pcd
            )
        else:
            global_report = ForestReport(
                accuracy=global_acc,
                macro_f1=global_f1,
                macro_precision=0.0,
                macro_recall=0.0,
                per_class_f1={cn: 0.0 for cn in class_names},
                per_class_prec={cn: 0.0 for cn in class_names},
                per_class_recall={cn: 0.0 for cn in class_names},
                confusion_matrix=np.zeros((len(class_names), len(class_names))),
                pcd=0.0,
                forest_size=len(global_trees),
                class_names=class_names,
                accuracy_ci=(0.0, 0.0),
                macro_f1_ci=(0.0, 0.0)
            )

        return FLResults(
            strategy_id=strategy_name,
            global_accuracy=global_report.accuracy,
            global_macro_f1=global_report.macro_f1,
            n_trees_global=len(global_trees),
            client_ids=client_ids,
            client_accuracies=client_accuracies,
            client_f1_scores=client_f1_scores,
            client_metadata=client_metadata,
            global_report=global_report,
            client_hybrid_predictions=client_hybrid_predictions,
            y_test=y_test_numeric,
            class_names=class_names,
            client_hybrid_forest_sizes=client_hybrid_forest_sizes,
            convergence_round=server_model.get('convergence_round'),
            round_logs=server_model.get('round_logs', []),
            selected_ids=selected_ids_dict,
            all_tree_entries=server_model.get('all_tree_entries', [])
        )
