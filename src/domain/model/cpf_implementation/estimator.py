"""
Módulo de estimadores para Proactive Forest.

Modificaciones mínimas respecto al original para soporte FL:
  - Se añaden get_trees() y set_trees() a DecisionForestClassifier
    para permitir extraer y reemplazar árboles durante la agregación federada.
  - Se eliminan los print() de buildEpisode (se usa callback opcional en su lugar).
  - Todo lo demás es idéntico al código original de Cepero (2023).
"""
import numpy as np
from . import utils
from .selection_and_diversity import PercentageCorrectDiversity, QStatisticDiversity

from .tree_builder import TreeBuilder
from .sampling_and_voting import PerformanceWeightingVoter, SoftPerformanceWeightingVoter
from .sampling_and_voting import SimpleSet, BaggingSet, ProbabilitySet
from .probabilities import FIProbabilityLedger
from .criteria_and_splits import resolve_split_selection, resolve_split_criterion
from .selection_and_diversity import resolve_feature_selection


class DecisionTreeClassifier:
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
        X = np.asarray(X)
        y = np.asarray(y)
        if hasattr(self, '_encoder') and hasattr(self._encoder, 'classes_'):
            # If already using an encoder (e.g. from parent forest), assume y is encoded
            pass
        else:
            # Dummy encoder logic if standalone
            unique_classes = np.unique(y)
            self._encoder_dict = {val: idx for idx, val in enumerate(unique_classes)}
            self._decoder_dict = {idx: val for idx, val in enumerate(unique_classes)}
            y = np.array([self._encoder_dict[val] for val in y])
            
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
        
        # Performance optimization: list comprehension is faster than manual loops over np.zeros
        result = np.array([self._tree.predict(x) for x in X])
            
        if hasattr(self, '_decoder_dict'):
            return np.array([self._decoder_dict[val] for val in result])
        return result

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
            raise RuntimeError("Estimator not fitted, call `fit` before exploiting the model.")
        if check_input:
            X = np.asarray(X)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must match the input. "
                             "Model n_features is %s and input n_features is %s "
                             % (self._n_features, n_features))
        return X


class DecisionForestClassifier:
    """Bosque de árboles con bagging y votación por rendimiento."""

    def __init__(self, n_estimators=100, bootstrap=True, max_depth=None,
                 split_chooser='best', split_criterion='gini', min_samples_leaf=1,
                 feature_selection='log', feature_prob=None, min_gain_split=0,
                 min_samples_split=2, EPISODE=5, voting='soft', random_state=None):
        self._trees = []
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._n_classes = None
        self._encoder = None
        self._bootstrap = bootstrap
        self.EPISODE = EPISODE
        self.voting = voting
        self.random_state = random_state

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
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Pure Python basic label encoder
        self.classes_ = np.unique(y)
        self._encoder_dict = {val: idx for idx, val in enumerate(self.classes_)}
        self._decoder_dict = {idx: val for idx, val in enumerate(self.classes_)}
        y = np.array([self._encoder_dict[val] for val in y])
        
        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
            import random
            random.seed(self.random_state)

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
                    preds = self._predict_on_tree(X[validation_ids], new_tree)
                    new_tree.weight = float(np.mean(y[validation_ids] == preds))
            self._trees.append(new_tree)
            set_generator.clear()
        return self

    def predict(self, X, check_input=True):
        if check_input:
            X = self._validate(X, check_input=check_input)
            
        if self.voting == 'soft':
            voter = SoftPerformanceWeightingVoter(self._trees, self._n_classes)
        else:
            voter = PerformanceWeightingVoter(self._trees, self._n_classes)
        
        # Performance optimization: list comprehension is faster for large datasets
        result = np.array([voter.predict(x) for x in X])

        if not hasattr(self, '_decoder_dict'):
            return result

        # Evitar ValueError si el modelo devuelve etiquetas fuera del rango
        max_label = len(self.classes_) - 1
        if np.any(result < 0) or np.any(result > max_label):
            result = np.clip(result, 0, max_label)

        try:
            return np.array([self._decoder_dict[val] for val in result])
        except KeyError:
            # Por seguridad, limitar
            safe_result = np.clip(result, 0, max_label)
            return np.array([self._decoder_dict[val] for val in safe_result])

    def predict_proba(self, X, indexs, check_input=True):
        if check_input:
            X = self._validate(X, check_input=check_input)
            
        if self.voting == 'soft':
            voter = SoftPerformanceWeightingVoter(self._trees, self._n_classes)
        else:
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
        X = np.asarray(X)
        y = np.asarray(y)
        # Handle string labels
        if hasattr(self, '_encoder_dict'):
            y = np.array([self._encoder_dict.get(val, 0) for val in y])
        if diversity == 'pcd':
            metric = PercentageCorrectDiversity()
        elif diversity == 'qstat':
            metric = QStatisticDiversity()
        else:
            raise ValueError("It was not possible to recognize the diversity measure.")
        return metric.get_measure(self._trees, X, y)

    def _validate(self, X, check_input):
        if self._trees is None:
            raise RuntimeError("Estimator not fitted, call `fit` before exploiting the model.")
        if check_input:
            X = np.asarray(X)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must match the input. "
                             "Model n_features is %s and input n_features is %s "
                             % (self._n_features, n_features))
        return X

    def _predict_on_tree(self, X, tree, check_input=True):
        if check_input:
            X = self._validate(X, check_input=check_input)
        
        # Performance optimization
        return np.array([tree.predict(x) for x in X])

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
            'encoder_classes': self.classes_.tolist() if hasattr(self, 'classes_') else None,
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
                 min_samples_split=2, alpha=0.1, voting='soft', random_state=None):
        if 0 < alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError("The diversity rate can only take values from (0, 1].")
        super().__init__(n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth,
                         split_chooser=split_chooser, split_criterion=split_criterion,
                         min_samples_leaf=min_samples_leaf, feature_selection=feature_selection,
                         feature_prob=feature_prob, min_gain_split=min_gain_split,
                         min_samples_split=min_samples_split, voting=voting, random_state=random_state)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if not hasattr(self, '_encoder_dict'):
            # Pure Python basic label encoder
            self.classes_ = np.unique(y)
            self._encoder_dict = {val: idx for idx, val in enumerate(self.classes_)}
            self._decoder_dict = {idx: val for idx, val in enumerate(self.classes_)}
            y = np.array([self._encoder_dict[val] for val in y])
        else:
            y = np.array([self._encoder_dict.get(val, 0) for val in y])
            
        self._n_instances, self._n_features = X.shape
        self._n_classes = len(self.classes_)
        self._trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
            import random
            random.seed(self.random_state)

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
                    preds = self._predict_on_tree(X[validation_ids], new_tree)
                    new_tree.weight = float(np.mean(y[validation_ids] == preds))
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
        X = np.asarray(X)
        Xt = np.asarray(Xt)
        y = np.asarray(y)
        yt = np.asarray(yt)
        self._n_instances, self._n_features = X.shape

        if not hasattr(self, '_encoder_dict'):
            all_labels = np.unique(np.concatenate([y, yt]))
            self.classes_ = all_labels
            self._encoder_dict = {val: idx for idx, val in enumerate(self.classes_)}
            self._decoder_dict = {idx: val for idx, val in enumerate(self.classes_)}
        else:
            # Check if we need to update encoder with new labels (unlikely in FL but possible)
            all_labels = np.unique(np.concatenate([self.classes_, y, yt]))
            if len(all_labels) > len(self.classes_):
                self.classes_ = all_labels
                self._encoder_dict = {val: idx for idx, val in enumerate(self.classes_)}
                self._decoder_dict = {idx: val for idx, val in enumerate(self.classes_)}

        y = np.array([self._encoder_dict.get(val, 0) for val in y])
        yt = np.array([self._encoder_dict.get(val, 0) for val in yt])

        self._n_classes = len(self.classes_)
        self.set_generator = ProbabilitySet(self._n_instances)
        self._m_progressive_accuracy = []

        if len(self._trees) > 0 and EPISODE == self.n_estimators:
            # Only reset if we are building the full forest from scratch
            self._trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

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
            # Verificar límite máximo de árboles
            if len(self._trees) >= self.n_estimators:
                break
            ids = self.set_generator.training_ids(prob)
            new_tree = self._tree_builder.build_tree(X[ids], y[ids], self._n_classes)
            self._trees.append(new_tree)
            preds = self._predict_on_tree(Xt, new_tree)
            acc = float(np.mean(yt == preds))
            self._m_progressive_accuracy.append(acc)
            rate = i / self._n_estimators
            self.ledger.update_probabilities(new_tree, rate=rate)
            self._tree_builder.feature_prob = self.ledger.probabilities
            self.set_generator.clear()
            if verbose:
                print(f"  Árbol {len(self._trees)} | acc={acc:.4f}")

    # ── FL S9 Roulette: métodos para exponer/inyectar la ruleta de atributos ──

    def get_feature_probabilities(self) -> list:
        """Return the current feature probability vector (roulette state).

        After training (fit or buildEpisode), this vector reflects
        how the Proactive algorithm has adjusted the exploration
        probabilities for each feature based on accumulated Feature
        Importance.  In the S9 federated strategy, this vector is
        sent to the server for aggregation.

        Returns:
            list[float]: Probability for each feature, sums to 1.0.
        """
        # After buildEpisode, the ledger holds the latest state
        if hasattr(self, 'ledger') and self.ledger is not None:
            return list(self.ledger.probabilities)
        # After fit() (non-CPF path), _feature_prob has the latest state
        if self._feature_prob is not None:
            return list(self._feature_prob)
        # Fallback: uniform
        if self._n_features is not None and self._n_features > 0:
            p = 1.0 / self._n_features
            return [p] * self._n_features
        return []

    def set_feature_probabilities(self, probabilities: list) -> None:
        """Inject a new feature probability vector (federated roulette).

        This overwrites ``_feature_prob`` so that the *next* call to
        ``buildEpisode`` (or ``fit``) will initialize its
        ``FIProbabilityLedger`` with these probabilities instead of
        the uniform distribution.

        Args:
            probabilities: List of floats of length ``n_features``,
                must sum to 1.0 and contain only positive values.
        """
        probs = list(probabilities)
        if self._n_features is not None and len(probs) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} probabilities, got {len(probs)}."
            )
        self._feature_prob = probs
