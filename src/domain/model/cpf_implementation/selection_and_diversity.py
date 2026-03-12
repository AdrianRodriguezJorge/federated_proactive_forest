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
        tally = 0
        n_instances = X.shape[0]
        for i in range(n_instances):
            instance, target = X[i], y[i]
            n_corrects = sum(1 for p in predictors if p.predict(instance) == target)
            if 0.1 * len(predictors) <= n_corrects <= 0.9 * len(predictors):
                tally += 1
        return tally / n_instances


class QStatisticDiversity(DiversityMeasure):
    def get_measure(self, predictors, X, y):
        n_instances = X.shape[0]
        n_predictors = len(predictors)
        q_total = 0
        for i in range(0, n_predictors - 1):
            for j in range(i + 1, n_predictors):
                n = np.zeros((2, 2))
                for k in range(n_instances):
                    i_pred = predictors[i].predict(X[k])
                    j_pred = predictors[j].predict(X[k])
                    true_y = y[k]
                    if i_pred == true_y:
                        n[1][1] += 1 if j_pred == true_y else 0
                        n[1][0] += 1 if j_pred != true_y else 0
                    else:
                        n[0][1] += 1 if j_pred == true_y else 0
                        n[0][0] += 1 if j_pred != true_y else 0
                for k in range(2):
                    for l in range(2):
                        if n[k][l] == 0:
                            n[k][l] += 1
                same = n[1][1] * n[0][0]
                diff = n[1][0] * n[0][1]
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
