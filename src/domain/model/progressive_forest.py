"""Progressive Forest wrapper using Comparative Progressive Forest (CPF).

Provides automated episodic training and convergence-based early stopping.
"""

from typing import Any, Optional
import numpy as np


class ComparativeProgressiveForest:
    """Comparative Progressive Forest (CPF) implementation.

    Trains a ProactiveForest in episodes, applying early stopping based on
    the convergence of validation accuracy across trees.
    """

    def __init__(
        self,
        classifier: Any,
        verbose: bool = False,
        convergence_threshold: float = 0.002,
    ):
        """Initializes CPF tracking.

        Args:
            classifier (Any): Instancia de ProactiveForestClassifier.
            verbose (bool): Si True, imprime logs de progreso.
            convergence_threshold (float): Umbral de convergencia.
        """
        self._classifier = classifier
        self.CONVERGENCE = convergence_threshold
        self.EPISODE = 5
        self.verbose = verbose

    def fit(
        self, X: np.ndarray, y: np.ndarray, Xt: np.ndarray, yt: np.ndarray
    ) -> "ComparativeProgressiveForest":
        """Train CPF with early stopping.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            Xt (np.ndarray): Validation features.
            yt (np.ndarray): Validation labels.

        Returns:
            ComparativeProgressiveForest: Self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        Xt = np.asarray(Xt)
        yt = np.asarray(yt)
        models_built = 0
        stop_counter = 0
        previous_episode_accuracy = None
        episode_accuracy_dif = 0.002
        self.EPISODE = 5
        self._classifier.clean_trees()

        while models_built < self._classifier.n_estimators:
            self._classifier.buildEpisode(
                X, y, Xt, yt, self.EPISODE, verbose=self.verbose
            )
            # Count trees actually constructed
            new_trees = len(self._classifier._trees) - models_built
            models_built = len(self._classifier._trees)

            current_episode_accs = self._classifier._m_progressive_accuracy[-new_trees:]
            min_accuracy = min(current_episode_accs) if current_episode_accs else 0.0
            max_accuracy = max(current_episode_accs) if current_episode_accs else 0.0
            episode_accuracy = max_accuracy - min_accuracy

            # Convergence check based strictly on Thesis (Algoritmo 5):
            # Spread (Max - Min) <= Valor_Convergencia
            if episode_accuracy <= self.CONVERGENCE:
                stop_counter += 1
                if self.verbose:
                    print(
                        f"  [CPF] Condición de convergencia alcanzada "
                        f"(spread={episode_accuracy:.5f}, "
                        f"counter={stop_counter})"
                    )
                if stop_counter == 2:
                    # Truncate ensemble to point of maximum validation efficiency
                    best_idx_in_episode = np.argmax(
                        self._classifier._m_progressive_accuracy
                    )
                    trees_to_keep = (models_built - new_trees) + (
                        best_idx_in_episode + 1
                    )
                    self._classifier._trees = self._classifier._trees[
                        :trees_to_keep
                    ]
                    if self.verbose:
                        print(
                            f"  [CPF] Convergencia final alcanzada. "
                            f"Truncando bosque a {trees_to_keep} árboles."
                        )
                    break
            else:
                stop_counter = 0
                self.EPISODE += 1  # Standard progressive behavior
                if self.verbose:
                    print(
                        f"  [CPF] No converge (spread={episode_accuracy:.5f}), "
                        f"aumentando episodio a {self.EPISODE}"
                    )

            if models_built + self.EPISODE > self._classifier.n_estimators:
                self.EPISODE = self._classifier.n_estimators - models_built

            previous_episode_accuracy = episode_accuracy
            if self.verbose:
                print(
                    f"  [CPF] Árboles construidos={models_built}, "
                    f"próximo episodio={self.EPISODE}"
                )
                print("  " + "-" * 40)

        return self

    def return_forest(self) -> Any:
        """Retorna el clasificador interno con los árboles construidos.

        Returns:
            Any: Inner classifier.
        """
        return self._classifier

    def forest_tree_size(self) -> int:
        """Retorna el número de árboles construidos.

        Returns:
            int: Tree count.
        """
        return len(self._classifier._trees)


class ProgressiveForest:
    """Wrapper for ComparativeProgressiveForest with simplified interface.

    Implements episodic training with early stopping based on convergence.
    """

    CONVERGENCE_THRESHOLD = 0.002
    INITIAL_EPISODE_SIZE = 5

    def __init__(
        self,
        forest: Any,
        verbose: bool = False,
        convergence_threshold: float = 0.002,
    ):
        """Initializes Progressive Forest.

        Args:
            forest (Any): Classifier or forest instance.
            verbose (bool): Print debugging info.
            convergence_threshold (float): Convergence threshold.
        """
        self.forest = forest
        self.verbose = verbose
        self.convergence_threshold = convergence_threshold
        self._cpf = None

    def fit_with_early_stopping(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "ProgressiveForest":
        """Fit forest with early stopping based on convergence.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.

        Returns:
            ProgressiveForest: Self.
        """
        if hasattr(self.forest, "_classifier"):
            classifier = self.forest._classifier
        else:
            classifier = self.forest

        self._cpf = ComparativeProgressiveForest(
            classifier,
            verbose=self.verbose,
            convergence_threshold=self.convergence_threshold,
        )
        self._cpf.fit(X_train, y_train, X_val, y_val)

        return self

    def get_forest(self) -> Any:
        """Retorna el bosque entrenado.

        Returns:
            Any: The trained forest classifier.
        """
        if self._cpf is not None:
            return self._cpf.return_forest()
        return self.forest

    def forest_tree_size(self) -> int:
        """Retorna el número de árboles construidos.

        Returns:
            int: Tree count.
        """
        if self._cpf is not None:
            return self._cpf.forest_tree_size()
        if hasattr(self.forest, "_trees"):
            return len(self.forest._trees)
        return 0
