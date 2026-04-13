"""
Módulo para selección de características y medidas de diversidad.
"""
from abc import ABC, abstractmethod
import numpy as np


class DiversityMeasure(ABC):
    @abstractmethod
    def get_measure(self, predictors, X, y):
        pass


class PercentageCorrectDiversity(DiversityMeasure):
    """
    PCD: Porcentaje de instancias donde entre 10% y 90% de los predictores aciertan.
    """
    def get_measure(self, predictors, X, y):
        if not predictors:
            return 0.0
        
        n_instances = X.shape[0]
        n_predictors = len(predictors)
        
        # Vectorized prediction: Get all predictions for all instances at once
        # Shape: (n_predictors, n_instances)
        all_preds = np.array([p.predict(X) for p in predictors])
        
        # Compare with true labels (broadcasting)
        # Shape: (n_predictors, n_instances)
        correct_mask = (all_preds == y)
        
        # Count correct predictors per instance
        # Shape: (n_instances,)
        n_corrects = np.sum(correct_mask, axis=0)
        
        # Apply PCD thresholds
        lower_bound = 0.1 * n_predictors
        upper_bound = 0.9 * n_predictors
        
        tally = np.sum((n_corrects >= lower_bound) & (n_corrects <= upper_bound))
        
        return tally / n_instances


class QStatisticDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        if not predictors or len(predictors) < 2:
            return 0.0
            
        n_instances = X.shape[0]
        n_predictors = len(predictors)
        
        # Matrix of successes: S[i, k] = 1 if predictor i is correct for instance k
        # Shape: (n_predictors, n_instances)
        successes = np.array([(p.predict(X) == y).astype(int) for p in predictors])
        failures = 1 - successes
        
        q_total = 0
        for i in range(0, n_predictors - 1):
            for j in range(i + 1, n_predictors):
                # Calculate counts for the 2x2 table using dot products (vectorized over instances)
                n11 = np.sum(successes[i] * successes[j])
                n00 = np.sum(failures[i] * failures[j])
                n10 = np.sum(successes[i] * failures[j])
                n01 = np.sum(failures[i] * successes[j])
                
                # Laplace smoothing (from original code: if 0, add 1)
                # It's better to add a small epsilon or only if zero, 
                # but we follow the original logic for consistency.
                n11 = n11 if n11 > 0 else 1
                n00 = n00 if n00 > 0 else 1
                n10 = n10 if n10 > 0 else 1
                n01 = n01 if n01 > 0 else 1
                
                same = n11 * n00
                diff = n10 * n01
                q_total += (same - diff) / (same + diff)
                
        return 2 * q_total / (n_predictors * (n_predictors - 1))


class FeatureSelection(ABC):
    @abstractmethod
    def get_features(self, n_features, prob):
        pass


class AllFeatureSelection(FeatureSelection):
    @property
    def name(self):
        return 'all'

    def get_features(self, n_features, prob=None):
        return list(range(n_features))


class LogFeatureSelection(FeatureSelection):
    @property
    def name(self):
        return 'log'

    def get_features(self, n_features, prob=None):
        import math
        sample_size = int(math.floor(math.log2(n_features)) + 1)
        population = list(range(n_features))
        selected = np.random.choice(population, replace=False, size=sample_size, p=prob)
        return selected


class ProbFeatureSelection(FeatureSelection):
    @property
    def name(self):
        return 'prob'

    def get_features(self, n_features, prob=None):
        sample_size = n_features
        population = list(range(n_features))
        selected = np.random.choice(population, replace=True, size=sample_size, p=prob)
        return np.unique(selected)


def resolve_feature_selection(name):
    if name == 'all':
        return AllFeatureSelection()
    elif name == 'log':
        return LogFeatureSelection()
    elif name == 'prob':
        return ProbFeatureSelection()
    else:
        raise ValueError('Unknown feature selection criterion {}'.format(name))
