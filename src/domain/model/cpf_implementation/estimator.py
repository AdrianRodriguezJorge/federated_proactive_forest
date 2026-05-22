"""Estimator classes for Proactive Forest classifiers.

Implements decision tree classifiers, decision forest classifiers (bagging with
OOB performance weighting), and proactive forest classifiers (incorporating
dynamic feature probability ledger tracking). Minimal FL extensions allow tree
extraction, injection, and metadata transfers.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .criteria_and_splits import (
    resolve_split_criterion,
    resolve_split_selection,
)
from .probabilities import FIProbabilityLedger
from .sampling_and_voting import (
    BaggingSet,
    PerformanceWeightingVoter,
    ProbabilitySet,
    SimpleSet,
    SoftPerformanceWeightingVoter,
)
from .selection_and_diversity import (
    PercentageCorrectDiversity,
    QStatisticDiversity,
    resolve_feature_selection,
)
from .tree_builder import TreeBuilder
from . import utils


class DecisionTreeClassifier:
    """Individual decision tree classifier with custom splitters."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        split_chooser: str = "best",
        split_criterion: str = "gini",
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        feature_selection: str = "all",
        feature_prob: Optional[List[float]] = None,
        min_gain_split: float = 0.0,
    ):
        """Initializes decision tree classifier parameters.

        Args:
            max_depth (Optional[int]): Maximum tree depth. Defaults to None.
            split_chooser (str): Choose best or random split.
                Defaults to "best".
            split_criterion (str): Splitting criterion ("gini" or "entropy").
                Defaults to "gini".
            min_samples_leaf (int): Minimum leaf samples. Defaults to 1.
            min_samples_split (int): Minimum split samples. Defaults to 2.
            feature_selection (str): Feature subset selector strategy.
                Defaults to "all".
            feature_prob (Optional[List[float]]): Array of selection probabilities.
                Defaults to None.
            min_gain_split (float): Minimum information gain to split a node.
                Defaults to 0.0.

        Raises:
            ValueError: If parameters violate boundary constraints.
        """
        self._tree = None
        self._n_features = None
        self._n_instances = None
        self._tree_builder = None
        self._encoder = None
        self._n_classes = None

        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError("The depth of the tree must be greater than 0.")

        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError(
                "The minimum number of instances to place in a leaf "
                "must be greater than 0."
            )

        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError(
                "The minimum number of instances to make a split "
                "must be greater than 1"
            )

        if feature_prob is None or (
            utils.check_array_sum_one(feature_prob)
            and utils.check_positive_array(feature_prob)
        ):
            self._feature_prob = feature_prob
        else:
            raise ValueError(
                "The features probabilities must be positive values "
                "and the sum must be one"
            )

        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError(
                "The minimum value of gain to make a split "
                "must be greater or equal to 0"
            )

        if split_chooser is not None:
            self._split_chooser = resolve_split_selection(split_chooser)
        else:
            raise ValueError("The split chooser can not be None.")

        if split_criterion is not None:
            self._split_criterion = resolve_split_criterion(split_criterion)
        else:
            raise ValueError("The split criterion can not be None.")

        if feature_selection is not None:
            self._feature_selection = resolve_feature_selection(
                feature_selection
            )
        else:
            raise ValueError("The feature selection criteria can not be None.")

    def fit(self, X: Any, y: Any) -> "DecisionTreeClassifier":
        """Fits the decision tree classifier on a dataset.

        Args:
            X (Any): 2D array of shape (n_samples, n_features).
            y (Any): 1D array of target labels.

        Returns:
            DecisionTreeClassifier: Fitted classifier instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if hasattr(self, "_encoder") and hasattr(self._encoder, "classes_"):
            # If already using an encoder, assume y is encoded
            pass
        else:
            unique_classes = np.unique(y)
            self._encoder_dict = {
                val: idx for idx, val in enumerate(unique_classes)
            }
            self._decoder_dict = {
                idx: val for idx, val in enumerate(unique_classes)
            }
            y = np.array([self._encoder_dict[val] for val in y])

        self._n_instances, self._n_features = X.shape
        self._n_classes = utils.count_classes(y)
        self._tree_builder = TreeBuilder(
            split_criterion=self._split_criterion,
            feature_prob=self._feature_prob,
            feature_selection=self._feature_selection,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            min_gain_split=self._min_gain_split,
            min_samples_split=self._min_samples_split,
            split_chooser=self._split_chooser,
        )
        self._tree = self._tree_builder.build_tree(X, y, self._n_classes)
        return self

    def predict(self, X: Any, check_input: bool = True) -> np.ndarray:
        """Predict target labels for a dataset.

        Args:
            X (Any): Features matrix.
            check_input (bool): Validate input shape. Defaults to True.

        Returns:
            np.ndarray: Predicted labels.
        """
        if check_input:
            X = self._validate_predict(X, check_input=check_input)

        # Vectorized tree-level prediction
        result = self._tree.predict(X)

        if hasattr(self, "_decoder_dict"):
            return np.array([self._decoder_dict[val] for val in result])
        return result

    def predict_proba(self, X: Any, check_input: bool = True) -> List[Any]:
        """Predict class probability distributions for a dataset.

        Args:
            X (Any): Feature matrix.
            check_input (bool): Validate input shape. Defaults to True.

        Returns:
            List[Any]: List of probability arrays per sample.
        """
        if check_input:
            X = self._validate_predict(X, check_input=check_input)
        sample_size, _ = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            result[i] = self._tree.predict_proba(X[i])
        return result

    def _validate_predict(self, X: Any, check_input: bool) -> np.ndarray:
        """Validate feature matrix dimensions prior to prediction.

        Args:
            X (Any): Input.
            check_input (bool): Perform array conversion.

        Returns:
            np.ndarray: Verified array.
        """
        if self._tree is None:
            raise RuntimeError(
                "Estimator not fitted, call `fit` before exploiting the model."
            )
        if check_input:
            X = np.asarray(X)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError(
                f"Number of features of the model must match the input. "
                f"Model n_features is {self._n_features} and input "
                f"n_features is {n_features}"
            )
        return X


class DecisionForestClassifier:
    """Forest classifier supporting bagging and performance-weighted voting."""

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap: bool = True,
        max_depth: Optional[int] = None,
        split_chooser: str = "best",
        split_criterion: str = "gini",
        min_samples_leaf: int = 1,
        feature_selection: str = "log",
        feature_prob: Optional[List[float]] = None,
        min_gain_split: float = 0.0,
        min_samples_split: int = 2,
        EPISODE: int = 5,
        voting: str = "soft",
        random_state: Optional[int] = None,
    ):
        """Initializes the forest classifier parameters."""
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
            raise ValueError("The number of trees must be greater than 0.")
        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError("The depth of the tree must be greater than 0.")
        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError(
                "The minimum number of instances to place in a leaf "
                "must be greater than 0."
            )
        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError(
                "The minimum number of instances to make a split "
                "must be greater than 1"
            )
        if feature_prob is None or (
            utils.check_array_sum_one(feature_prob)
            and utils.check_positive_array(feature_prob)
        ):
            self._feature_prob = feature_prob
        else:
            raise ValueError(
                "The features probabilities must be positive values "
                "and the sum must be one"
            )
        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError(
                "The minimum value of gain to make a split "
                "must be greater or equal to 0"
            )
        if split_chooser is not None:
            self._split_chooser = resolve_split_selection(split_chooser)
        else:
            raise ValueError("The split chooser can not be None.")
        if split_criterion is not None:
            self._split_criterion = resolve_split_criterion(split_criterion)
        else:
            raise ValueError("The split criterion can not be None.")
        if feature_selection is not None:
            self._feature_selection = resolve_feature_selection(
                feature_selection
            )
        else:
            raise ValueError("The feature selection criteria can not be None.")

    @property
    def n_estimators(self) -> int:
        """Return the estimator limit.

        Returns:
            int: Tree count target.
        """
        return self._n_estimators

    @property
    def bootstrap(self) -> bool:
        """Return bootstrap state.

        Returns:
            bool: True if bootstrap is enabled.
        """
        return self._bootstrap

    def fit(self, X: Any, y: Any) -> "DecisionForestClassifier":
        """Fit forest classifiers on a dataset.

        Args:
            X (Any): Input features.
            y (Any): Target labels.

        Returns:
            DecisionForestClassifier: Fitted forest.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Pure Python basic label encoder
        if not hasattr(self, "_encoder_dict") or self._encoder_dict is None:
            self.classes_ = np.unique(y)
            self._encoder_dict = {
                val: idx for idx, val in enumerate(self.classes_)
            }
            self._decoder_dict = {
                idx: val for idx, val in enumerate(self.classes_)
            }

        # Safe transform to internal indices
        y = np.array([self._encoder_dict[val] for val in y])

        self._n_instances, self._n_features = X.shape
        self._n_classes = len(self._encoder_dict)
        self._trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
            import random

            random.seed(self.random_state)

        set_generator = (
            BaggingSet(self._n_instances)
            if self._bootstrap
            else SimpleSet(self._n_instances)
        )
        self._tree_builder = TreeBuilder(
            split_criterion=self._split_criterion,
            feature_prob=self._feature_prob,
            feature_selection=self._feature_selection,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            min_gain_split=self._min_gain_split,
            min_samples_split=self._min_samples_split,
            split_chooser=self._split_chooser,
        )
        for _ in range(self._n_estimators):
            ids = set_generator.training_ids()
            new_tree = self._tree_builder.build_tree(
                X[ids], y[ids], self._n_classes
            )
            if self._bootstrap:
                validation_ids = set_generator.oob_ids()
                if validation_ids:
                    preds = self._predict_on_tree(X[validation_ids], new_tree)
                    new_tree.weight = float(
                        np.mean(y[validation_ids] == preds)
                    )
            self._trees.append(new_tree)
            set_generator.clear()
        return self

    def predict(self, X: Any, check_input: bool = True) -> np.ndarray:
        """Predict classes using performance-weighted voting.

        Args:
            X (Any): Input features.
            check_input (bool): Verify shape. Defaults to True.

        Returns:
            np.ndarray: Predicted labels.
        """
        if check_input:
            X = self._validate(X, check_input=check_input)

        if self.voting == "soft":
            voter = SoftPerformanceWeightingVoter(self._trees, self._n_classes)
        else:
            voter = PerformanceWeightingVoter(self._trees, self._n_classes)

        # Vectorized batch prediction
        result = voter.predict(X)

        if not hasattr(self, "_decoder_dict"):
            return result

        # Avoid label overflow crashes
        max_label = len(self.classes_) - 1
        if np.any(result < 0) or np.any(result > max_label):
            result = np.clip(result, 0, max_label)

        try:
            return np.array([self._decoder_dict[val] for val in result])
        except KeyError:
            safe_result = np.clip(result, 0, max_label)
            return np.array([self._decoder_dict[val] for val in safe_result])

    def predict_proba(
        self, X: Any, indexs: Any, check_input: bool = True
    ) -> List[Any]:
        """Predict probabilities per sample.

        Args:
            X (Any): Feature matrix.
            indexs (Any): Output indices.
            check_input (bool): Verify shape. Defaults to True.

        Returns:
            List[Any]: Predicted probabilities.
        """
        if check_input:
            X = self._validate(X, check_input=check_input)

        if self.voting == "soft":
            voter = SoftPerformanceWeightingVoter(self._trees, self._n_classes)
        else:
            voter = PerformanceWeightingVoter(self._trees, self._n_classes)

        sample_size, _ = X.shape
        result = list(range(sample_size))
        for i in range(sample_size):
            result[i] = voter.predict_proba(X[i], indexs)
        return result

    def feature_importances(self) -> np.ndarray:
        """Return average feature importances across the forest.

        Returns:
            np.ndarray: Calculated feature importances.
        """
        importances = np.zeros(self._n_features)
        for tree in self._trees:
            importances += tree.feature_importances()
        importances /= len(self._trees)
        return importances

    def trees_mean_weight(self) -> float:
        """Return average tree weighting.

        Returns:
            float: Mean tree weight.
        """
        return np.mean([tree.weight for tree in self._trees])

    def diversity_measure(
        self, X: Any, y: Any, diversity: str = "pcd"
    ) -> float:
        """Return diversity measure score.

        Args:
            X (Any): Features.
            y (Any): Ground truth targets.
            diversity (str): Metric type ('pcd' or 'qstat'). Defaults to "pcd".

        Returns:
            float: Diversity score.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # Handle label types robustly when an encoder mapping exists.
        # Cases to support:
        # - `y` already contains integer indices -> use as-is (fast path)
        # - `y` contains original label names (strings) -> map via _encoder_dict
        # - `y` contains numeric strings -> try string lookup or numeric index
        if hasattr(self, "_encoder_dict"):
            n_classes = len(self._encoder_dict)
            # Fast path: already integer indices in valid range
            if np.issubdtype(y.dtype, np.integer) and y.size > 0:
                if np.all((y >= 0) & (y < n_classes)):
                    y = y.astype(np.int64)
                else:
                    # Some integers out of range: map per-value defensively
                    mapped = []
                    for val in y:
                        if val in self._encoder_dict:
                            mapped.append(self._encoder_dict[val])
                        else:
                            sval = str(val)
                            mapped.append(self._encoder_dict.get(sval, 0))
                    y = np.array(mapped, dtype=np.int64)
            else:
                # Non-integer inputs: try mapping by label, by string, or numeric fallback
                mapped = []
                for val in y:
                    if val in self._encoder_dict:
                        mapped.append(self._encoder_dict[val])
                        continue
                    sval = str(val)
                    if sval in self._encoder_dict:
                        mapped.append(self._encoder_dict[sval])
                        continue
                    # Last attempt: numeric string -> integer index if valid
                    try:
                        ival = int(val)
                        if 0 <= ival < n_classes:
                            mapped.append(ival)
                        else:
                            mapped.append(0)
                    except Exception:
                        mapped.append(0)
                y = np.array(mapped, dtype=np.int64)
        if diversity == "pcd":
            metric = PercentageCorrectDiversity()
        elif diversity == "qstat":
            metric = QStatisticDiversity()
        else:
            raise ValueError(
                "It was not possible to recognize the diversity measure."
            )
        return metric.get_measure(self._trees, X, y)

    def _validate(self, X: Any, check_input: bool) -> np.ndarray:
        """Validate input.

        Args:
            X (Any): Feature matrix.
            check_input (bool): Whether to perform validation.

        Returns:
            np.ndarray: Validated array.
        """
        if self._trees is None:
            raise RuntimeError(
                "Estimator not fitted, call `fit` before exploiting the model."
            )
        if check_input:
            X = np.asarray(X)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError(
                f"Number of features of the model must match the input. "
                f"Model n_features is {self._n_features} and input "
                f"n_features is {n_features}"
            )
        return X

    def _predict_on_tree(
        self, X: Any, tree: Any, check_input: bool = True
    ) -> np.ndarray:
        """Predict labels on a single tree.

        Args:
            X (Any): Features matrix.
            tree (Any): Tree object.
            check_input (bool): Validate shape. Defaults to True.

        Returns:
            np.ndarray: Tree prediction output.
        """
        if check_input:
            X = self._validate(X, check_input=check_input)
        return tree.predict(X)

    def clean_trees(self) -> None:
        """Discard internal trees in the forest."""
        self._trees = []

    # -- FL integration: federated additions --
    def get_trees(self) -> List[Any]:
        """Retorna la lista interna de DecisionTree para transferencia FL.

        Returns:
            List[Any]: List of estimators.
        """
        return list(self._trees)

    def set_trees(self, trees: List[Any]) -> None:
        """Reemplaza los árboles internos (usado en la ronda FL).

        Mantiene encoder y metadata del fit original.

        Args:
            trees (List[Any]): New list of tree objects.
        """
        self._trees = list(trees)

    def get_fl_metadata(self) -> Dict[str, Any]:
        """Retorna metadatos necesarios para reconstruir el bosque.

        Returns:
            Dict[str, Any]: Key-value metadata.
        """
        return {
            "n_features": self._n_features,
            "n_classes": self._n_classes,
            "encoder_classes": (
                self.classes_.tolist() if hasattr(self, "classes_") else None
            ),
        }


class ProactiveForestClassifier(DecisionForestClassifier):
    """Proactive Forest classifier using dynamic feature exploration tracking."""

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap: bool = True,
        max_depth: Optional[int] = None,
        split_chooser: str = "best",
        split_criterion: str = "entropy",
        min_samples_leaf: int = 1,
        feature_selection: str = "prob",
        feature_prob: Optional[List[float]] = None,
        min_gain_split: float = 0.0,
        min_samples_split: int = 2,
        alpha: float = 0.1,
        voting: str = "soft",
        random_state: Optional[int] = None,
    ):
        """Initializes proactive forest parameters."""
        if 0 < alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError(
                "The diversity rate can only take values from (0, 1]."
            )
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_depth=max_depth,
            split_chooser=split_chooser,
            split_criterion=split_criterion,
            min_samples_leaf=min_samples_leaf,
            feature_selection=feature_selection,
            feature_prob=feature_prob,
            min_gain_split=min_gain_split,
            min_samples_split=min_samples_split,
            voting=voting,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any) -> "ProactiveForestClassifier":
        """Fit proactive forest on a dataset.

        Args:
            X (Any): Features matrix.
            y (Any): Targets.

        Returns:
            ProactiveForestClassifier: Fitted instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if not hasattr(self, "_encoder_dict"):
            self.classes_ = np.unique(y)
            self._encoder_dict = {
                val: idx for idx, val in enumerate(self.classes_)
            }
            self._decoder_dict = {
                idx: val for idx, val in enumerate(self.classes_)
            }
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

        set_generator = (
            BaggingSet(self._n_instances)
            if self._bootstrap
            else SimpleSet(self._n_instances)
        )
        ledger = FIProbabilityLedger(
            probabilities=self._feature_prob,
            n_features=self._n_features,
            alpha=self.alpha,
        )
        self._tree_builder = TreeBuilder(
            split_criterion=self._split_criterion,
            feature_prob=ledger.probabilities,
            feature_selection=self._feature_selection,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            min_gain_split=self._min_gain_split,
            min_samples_split=self._min_samples_split,
            split_chooser=self._split_chooser,
        )
        for i in range(1, self._n_estimators + 1):
            ids = set_generator.training_ids()
            new_tree = self._tree_builder.build_tree(
                X[ids], y[ids], self._n_classes
            )
            if self._bootstrap:
                validation_ids = set_generator.oob_ids()
                if validation_ids:
                    preds = self._predict_on_tree(X[validation_ids], new_tree)
                    new_tree.weight = float(
                        np.mean(y[validation_ids] == preds)
                    )
            self._trees.append(new_tree)
            set_generator.clear()
            rate = i / self._n_estimators
            ledger.update_probabilities(new_tree, rate=rate)
            self._tree_builder.feature_prob = ledger.probabilities
        return self

    def buildEpisode(
        self,
        X: Any,
        y: Any,
        Xt: Any,
        yt: Any,
        EPISODE: int,
        verbose: bool = False,
    ) -> None:
        """Construye un episodio de EPISODE árboles para CPF.

        verbose=False suprime los print() originales.

        Args:
            X (Any): Train features.
            y (Any): Train targets.
            Xt (Any): Test features.
            yt (Any): Test targets.
            EPISODE (int): Size of the episode.
            verbose (bool): Whether to print progress. Defaults to False.
        """
        X = np.asarray(X)
        Xt = np.asarray(Xt)
        y = np.asarray(y)
        yt = np.asarray(yt)
        self._n_instances, self._n_features = X.shape

        if not hasattr(self, "_encoder_dict"):
            all_labels = np.unique(np.concatenate([y, yt]))
            self.classes_ = all_labels
            self._encoder_dict = {
                val: idx for idx, val in enumerate(self.classes_)
            }
            self._decoder_dict = {
                idx: val for idx, val in enumerate(self.classes_)
            }
        else:
            # Check if encoder classes need updating
            all_labels = np.unique(np.concatenate([self.classes_, y, yt]))
            if len(all_labels) > len(self.classes_):
                self.classes_ = all_labels
                self._encoder_dict = {
                    val: idx for idx, val in enumerate(self.classes_)
                }
                self._decoder_dict = {
                    idx: val for idx, val in enumerate(self.classes_)
                }

        y = np.array([self._encoder_dict.get(val, 0) for val in y])
        yt = np.array([self._encoder_dict.get(val, 0) for val in yt])

        self._n_classes = len(self.classes_)
        self.set_generator = ProbabilitySet(self._n_instances)
        if (
            not hasattr(self, "_m_progressive_accuracy")
            or self._m_progressive_accuracy is None
        ):
            self._m_progressive_accuracy = []

        if len(self._trees) > 0 and EPISODE == self.n_estimators:
            # Only reset if we are building the full forest from scratch
            self._trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.ledger = FIProbabilityLedger(
            probabilities=self._feature_prob,
            n_features=self._n_features,
            alpha=self.alpha,
        )
        self._tree_builder = TreeBuilder(
            split_criterion=self._split_criterion,
            feature_prob=self.ledger.probabilities,
            feature_selection=self._feature_selection,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            min_gain_split=self._min_gain_split,
            min_samples_split=self._min_samples_split,
            split_chooser=self._split_chooser,
        )
        prob = [1 / len(y) for _ in range(len(y))]

        for i in range(EPISODE):
            if len(self._trees) >= self.n_estimators:
                break
            ids = self.set_generator.training_ids(prob)
            new_tree = self._tree_builder.build_tree(
                X[ids], y[ids], self._n_classes
            )
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

    # -- FL S9 Roulette: methods to expose/inject the feature roulette --

    def get_feature_probabilities(self) -> List[float]:
        """Return the current feature probability vector (roulette state).

        After training (fit or buildEpisode), this vector reflects
        how the Proactive algorithm has adjusted the exploration
        probabilities for each feature based on accumulated Feature
        Importance. In the S9 federated strategy, this vector is
        sent to the server for aggregation.

        Returns:
            List[float]: Probability for each feature, sums to 1.0.
        """
        # After buildEpisode, the ledger holds the latest state
        if hasattr(self, "ledger") and self.ledger is not None:
            return list(self.ledger.probabilities)
        # After fit() (non-CPF path), _feature_prob has the latest state
        if self._feature_prob is not None:
            return list(self._feature_prob)
        # Fallback: uniform
        if self._n_features is not None and self._n_features > 0:
            p = 1.0 / self._n_features
            return [p] * self._n_features
        return []

    def set_feature_probabilities(self, probabilities: List[float]) -> None:
        """Inject a new feature probability vector (federated roulette).

        This overwrites ``_feature_prob`` so that the *next* call to
        ``buildEpisode`` (or ``fit``) will initialize its
        ``FIProbabilityLedger`` with these probabilities instead of
        the uniform distribution.

        Args:
            probabilities (List[float]): List of floats of length
                ``n_features``, must sum to 1.0 and contain positive values.
        """
        probs = list(probabilities)
        if self._n_features is not None and len(probs) != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} probabilities, got {len(probs)}."
            )
        self._feature_prob = probs
