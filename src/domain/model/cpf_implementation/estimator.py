"""
Módulo de estimadores para Proactive Forest.

Modificaciones mínimas respecto al original para soporte FL:
  - Se añaden get_trees() y set_trees() a DecisionForestClassifier
    para permitir extraer y reemplazar árboles durante la agregación federada.
  - Se eliminan los print() de buildEpisode (se usa callback opcional en su lugar).
  - Todo lo demás es idéntico al código original de Cepero (2023).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from . import utils
from .selection_and_diversity import PercentageCorrectDiversity, QStatisticDiversity
from .tree_builder import TreeBuilder
from .sampling_and_voting import PerformanceWeightingVoter
from .sampling_and_voting import SimpleSet, BaggingSet, ProbabilitySet
from .probabilites import FIProbabilityLedger
from .criteria_and_splits import resolve_split_selection, resolve_split_criterion
from .selection_and_diversity import resolve_feature_selection


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Árbol de decisión individual con selección de características personalizables."""

    def __init__(self, max_depth=None, split_chooser='best', split_criterion='gini',
                 min_samples_leaf=1, min_samples_split=2, feature_selection='all',
                 feature_prob=None, min_gain_split=0):
        self._tree = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._encoder = None
        self._n_classes = None

        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError('The depth of the tree must be greater than 0.')
        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError('The minimum number of instances to place in a leaf must be greater than 0.')
        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError('The minimum number of instances to make a split must be greater than 1')
        if feature_prob is None or (utils.check_array_sum_one(feature_prob) and
                                    utils.check_positive_array(feature_prob)):
            self._feature_prob = feature_prob
        else:
            raise ValueError('The features probabilities must be positive values and the sum must be one')
        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError('The minimum value of gain to make a split must be greater or equal to 0')
        if split_chooser is not None:
            self._split_chooser = resolve_split_selection(split_chooser)
        else:
            raise ValueError('The split chooser can not be None.')
        if split_criterion is not None:
            self._split_criterion = resolve_split_criterion(split_criterion)
        else:
            raise ValueError('The split criterion can not be None.')
        if feature_selection is not None:
            self._feature_selection = resolve_feature_selection(feature_selection)
        else:
            raise ValueError('The feature selection criteria can not be None.')

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y)
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=self._feature_prob,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)
        self._tree = self._tree_builder.build_tree(X, y, self._n_classes)
        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)
        sample_size, _ = X.shape
        result = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            result[i] = self._tree.predict(X[i])
        return self._encoder.inverse_transform(result)

    def predict_proba(self, X, check_input=True):
        if check_input:
            X = self._validate_predict(X, check_input=check_input)
        sample_size, _ = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            result[i] = self._tree.predict_proba(X[i])
        return result

    def _validate_predict(self, X, check_input):
        if self._tree is None:
            raise NotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        if check_input:
            X = check_array(X, dtype=None)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must match the input. "
                             "Model n_features is %s and input n_features is %s "
                             % (self._n_features, n_features))
        return X


class DecisionForestClassifier(BaseEstimator, ClassifierMixin):
    """Bosque de árboles con bagging y votación por rendimiento."""

    def __init__(self, n_estimators=100, bootstrap=True, max_depth=None,
                 split_chooser='best', split_criterion='gini', min_samples_leaf=1,
                 feature_selection='log', feature_prob=None, min_gain_split=0,
                 min_samples_split=2, EPISODE=5):
        self._trees = []
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._n_classes = None
        self._encoder = None
        self._bootstrap = bootstrap
        self.EPISODE = EPISODE

        if n_estimators is None or n_estimators > 0:
            self._n_estimators = n_estimators
        else:
            raise ValueError('The number of trees must be greater than 0.')
        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError('The depth of the tree must be greater than 0.')
        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError('The minimum number of instances to place in a leaf must be greater than 0.')
        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError('The minimum number of instances to make a split must be greater than 1')
        if feature_prob is None or (utils.check_array_sum_one(feature_prob) and
                                    utils.check_positive_array(feature_prob)):
            self._feature_prob = feature_prob
        else:
            raise ValueError('The features probabilities must be positive values and the sum must be one')
        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError('The minimum value of gain to make a split must be greater or equal to 0')
        if split_chooser is not None:
            self._split_chooser = resolve_split_selection(split_chooser)
        else:
            raise ValueError('The split chooser can not be None.')
        if split_criterion is not None:
            self._split_criterion = resolve_split_criterion(split_criterion)
        else:
            raise ValueError('The split criterion can not be None.')
        if feature_selection is not None:
            self._feature_selection = resolve_feature_selection(feature_selection)
        else:
            raise ValueError('The feature selection criteria can not be None.')

    @property
    def n_estimators(self):
        return self._n_estimators

    @property
    def bootstrap(self):
        return self._bootstrap

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y)
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._trees = []

        set_generator = BaggingSet(self._n_instances) if self._bootstrap else SimpleSet(self._n_instances)
        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=self._feature_prob,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)
        for _ in range(self._n_estimators):
            ids = set_generator.training_ids()
            new_tree = self._tree_builder.build_tree(X[ids], y[ids], self._n_classes)
            if self._bootstrap:
                validation_ids = set_generator.oob_ids()
                if validation_ids:
                    new_tree.weight = accuracy_score(y[validation_ids],
                                                     self._predict_on_tree(X[validation_ids], new_tree))
            self._trees.append(new_tree)
            set_generator.clear()
        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate(X, check_input=check_input)
        voter = PerformanceWeightingVoter(self._trees, self._n_classes)
        sample_size, _ = X.shape
        result = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            result[i] = voter.predict(X[i])
        return self._encoder.inverse_transform(result)

    def predict_proba(self, X, indexs, check_input=True):
        if check_input:
            X = self._validate(X, check_input=check_input)
        voter = PerformanceWeightingVoter(self._trees, self._n_classes)
        sample_size, _ = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            result[i] = voter.predict_proba(X[i], indexs)
        return result

    def feature_importances(self):
        importances = np.zeros(self._n_features)
        for tree in self._trees:
            importances += tree.feature_importances()
        importances /= len(self._trees)
        return importances

    def trees_mean_weight(self):
        return np.mean([tree.weight for tree in self._trees])

    def diversity_measure(self, X, y, diversity='pcd'):
        X, y = check_X_y(X, y, dtype=None)
        y = self._encoder.transform(y)
        if diversity == 'pcd':
            metric = PercentageCorrectDiversity()
        elif diversity == 'qstat':
            metric = QStatisticDiversity()
        else:
            raise ValueError("It was not possible to recognize the diversity measure.")
        return metric.get_measure(self._trees, X, y)

    def _validate(self, X, check_input):
        if self._trees is None:
            raise NotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        if check_input:
            X = check_array(X, dtype=None)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must match the input. "
                             "Model n_features is %s and input n_features is %s "
                             % (self._n_features, n_features))
        return X

    def _predict_on_tree(self, X, tree, check_input=True):
        if check_input:
            X = self._validate(X, check_input=check_input)
        sample_size, _ = X.shape
        result = np.zeros(sample_size, dtype=int)
        for i in range(sample_size):
            result[i] = tree.predict(X[i])
        return result

    def clean_trees(self):
        self._trees = []

    # ── FL integration: mínimas adiciones para el sistema federado ─────────────
    def get_trees(self):
        """Retorna la lista interna de DecisionTree para transferencia FL."""
        return list(self._trees)

    def set_trees(self, trees):
        """
        Reemplaza los árboles internos (usado por ClientUpdater en la ronda FL).
        Mantiene encoder y metadata del fit original.
        """
        self._trees = list(trees)

    def get_fl_metadata(self):
        """Retorna metadatos necesarios para reconstruir el bosque en el servidor."""
        return {
            'n_features': self._n_features,
            'n_classes': self._n_classes,
            'encoder_classes': self._encoder.classes_.tolist() if self._encoder else None,
        }


class ProactiveForestClassifier(DecisionForestClassifier):
    """
    Clasificador de bosque proactivo.
    Extiende DecisionForestClassifier ajustando dinámicamente las probabilidades
    de selección de características basado en importancia acumulada (Cepero, 2023).
    """

    def __init__(self, n_estimators=100, bootstrap=True, max_depth=None,
                 split_chooser='best', split_criterion='entropy', min_samples_leaf=1,
                 feature_selection='prob', feature_prob=None, min_gain_split=0,
                 min_samples_split=2, alpha=0.1):
        if 0 < alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError("The diversity rate can only take values from (0, 1].")
        super().__init__(n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth,
                         split_chooser=split_chooser, split_criterion=split_criterion,
                         min_samples_leaf=min_samples_leaf, feature_selection=feature_selection,
                         feature_prob=feature_prob, min_gain_split=min_gain_split,
                         min_samples_split=min_samples_split)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        if self._encoder is None:
            self._encoder = LabelEncoder()
            y = self._encoder.fit_transform(y)
        else:
            y = self._encoder.transform(y)
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._trees = []

        set_generator = BaggingSet(self._n_instances) if self._bootstrap else SimpleSet(self._n_instances)
        ledger = FIProbabilityLedger(probabilities=self._feature_prob,
                                     n_features=self._n_features, alpha=self.alpha)
        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=ledger.probabilities,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)
        for i in range(1, self._n_estimators + 1):
            ids = set_generator.training_ids()
            new_tree = self._tree_builder.build_tree(X[ids], y[ids], self._n_classes)
            if self._bootstrap:
                validation_ids = set_generator.oob_ids()
                if validation_ids:
                    new_tree.weight = accuracy_score(y[validation_ids],
                                                     self._predict_on_tree(X[validation_ids], new_tree))
            self._trees.append(new_tree)
            set_generator.clear()
            rate = i / self._n_estimators
            ledger.update_probabilities(new_tree, rate=rate)
            self._tree_builder.feature_prob = ledger.probabilities
        return self

    def buildEpisode(self, X, y, Xt, yt, EPISODE: int, verbose: bool = False):
        """
        Construye un episodio de EPISODE árboles para CPF.
        verbose=False suprime los print() originales (se usa en producción FL).
        """
        self._n_instances, self._n_features = X.shape
        self._encoder = LabelEncoder()
        y = self._encoder.fit_transform(y)
        yt = self._encoder.fit_transform(yt)
        self._n_classes = utils.count_classes(y)
        self.set_generator = ProbabilitySet(self._n_instances)
        self._m_progressive_accuracy = []

        if len(self._trees) >= self.n_estimators:
            self._trees = []

        self.ledger = FIProbabilityLedger(probabilities=self._feature_prob,
                                          n_features=self._n_features, alpha=self.alpha)
        self._tree_builder = TreeBuilder(split_criterion=self._split_criterion,
                                         feature_prob=self.ledger.probabilities,
                                         feature_selection=self._feature_selection,
                                         max_depth=self._max_depth,
                                         min_samples_leaf=self._min_samples_leaf,
                                         min_gain_split=self._min_gain_split,
                                         min_samples_split=self._min_samples_split,
                                         split_chooser=self._split_chooser)
        prob = [1 / len(y) for _ in range(len(y))]

        for i in range(EPISODE):
            ids = self.set_generator.training_ids(prob)
            new_tree = self._tree_builder.build_tree(X[ids], y[ids], self._n_classes)
            self._trees.append(new_tree)
            acc = accuracy_score(yt, self._predict_on_tree(Xt, new_tree))
            self._m_progressive_accuracy.append(acc)
            rate = i / self._n_estimators
            self.ledger.update_probabilities(new_tree, rate=rate)
            self._tree_builder.feature_prob = self.ledger.probabilities
            self.set_generator.clear()
            if verbose:
                print(f"  Árbol {len(self._trees)} | acc={acc:.4f}")
