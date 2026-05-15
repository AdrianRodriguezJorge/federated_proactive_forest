"""
Progressive Forest wrapper usando ComparativeProgressiveForest (CPF).

Este módulo proporciona una interfaz simplificada para entrenar Proactive Forest
con early stopping basado en convergencia.

Uso:
    from src.domain.model.progressive_forest import ProgressiveForest

    pf = ProgressiveForest(proactive_forest_instance, verbose=False)
    pf.fit_with_early_stopping(X_train, y_train, X_val, y_val)
    forest = pf.get_forest()
"""
from typing import Any
import numpy as np



class ComparativeProgressiveForest:
    """Comparative Progressive Forest (CPF) implementation.
    
    Trains a ProactiveForest in episodes, applying early stopping based on 
    the convergence of validation accuracy across trees.
    """

    def __init__(self, classifier, verbose: bool = False, convergence_threshold: float = 0.002):
        """
        :param classifier: Instancia de ProactiveForestClassifier.
        :param verbose: Si True, imprime logs de progreso (útil para debug/notebooks).
        :param convergence_threshold: Umbral de convergencia para early stopping.
        """
        self._classifier = classifier
        self.CONVERGENCE = convergence_threshold
        self.EPISODE = 5
        self.verbose = verbose

    def fit(self, X, y, Xt, yt):
        """
        Entrena CPF con early stopping.

        :param X: Datos de entrenamiento.
        :param y: Etiquetas de entrenamiento.
        :param Xt: Datos de validación (usados para medir convergencia).
        :param yt: Etiquetas de validación.
        :return: self
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
            self._classifier.buildEpisode(X, y, Xt, yt, self.EPISODE,
                                          verbose=self.verbose)
            # Contar árboles realmente construidos (puede ser menos que self.EPISODE si se alcanza el límite)
            new_trees = len(self._classifier._trees) - models_built
            models_built = len(self._classifier._trees)

            min_accuracy = min(self._classifier._m_progressive_accuracy)
            max_accuracy = max(self._classifier._m_progressive_accuracy)
            episode_accuracy = max_accuracy - min_accuracy

            # Convergence check based strictly on Thesis (Algoritmo 5): 
            # Spread (Max - Min) <= Valor_Convergencia (0.002)
            if episode_accuracy <= self.CONVERGENCE:
                stop_counter += 1
                if self.verbose:
                    print(f"  [CPF] Condición de convergencia alcanzada (spread={episode_accuracy:.5f}, counter={stop_counter})")
                if stop_counter == 2:
                    # Truncate ensemble to the point of maximum efficiency in the last episode
                    best_idx_in_episode = np.argmax(self._classifier._m_progressive_accuracy)
                    # Number of trees to keep: trees before this episode + (best_idx + 1)
                    trees_to_keep = (models_built - new_trees) + (best_idx_in_episode + 1)
                    self._classifier._trees = self._classifier._trees[:trees_to_keep]
                    if self.verbose:
                        print(f"  [CPF] Convergencia final alcanzada. Truncando bosque a {trees_to_keep} árboles.")
                    break
            else:
                stop_counter = 0
                self.EPISODE += 1  # Standard progressive behavior
                if self.verbose:
                    print(f"  [CPF] No converge (spread={episode_accuracy:.5f}), aumentando episodio a {self.EPISODE}")

            if models_built + self.EPISODE > self._classifier.n_estimators:
                self.EPISODE = self._classifier.n_estimators - models_built

            previous_episode_accuracy = episode_accuracy
            if self.verbose:
                print(f"  [CPF] Árboles construidos={models_built}, próximo episodio={self.EPISODE}")
                print("  " + "-" * 40)

        return self

    def return_forest(self):
        """Retorna el clasificador interno con los árboles construidos."""
        return self._classifier

    def forest_tree_size(self):
        """Retorna el número de árboles construidos."""
        return len(self._classifier._trees)


class ProgressiveForest:
    """
    Wrapper para ComparativeProgressiveForest con interfaz simplificada.

    Implementa entrenamiento por episodios con early stopping basado en
    convergencia de accuracy en validación.

    Parámetros:
    - CONVERGENCE: Umbral de convergencia (0.002 por defecto)
    - EPISODE: Tamaño inicial del episodio (5 por defecto)
    """

    CONVERGENCE_THRESHOLD = 0.002
    INITIAL_EPISODE_SIZE = 5

    def __init__(self, forest: Any, verbose: bool = False, convergence_threshold: float = 0.002):
        """
        Inicializa Progressive Forest.
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
        y_val: np.ndarray
    ) -> 'ProgressiveForest':
        # ...
        if hasattr(self.forest, '_classifier'):
            classifier = self.forest._classifier
        else:
            classifier = self.forest

        self._cpf = ComparativeProgressiveForest(
            classifier, 
            verbose=self.verbose,
            convergence_threshold=self.convergence_threshold
        )
        self._cpf.fit(X_train, y_train, X_val, y_val)

        return self

    def get_forest(self) -> Any:
        """
        Retorna el bosque entrenado.

        Returns:
            El clasificador con los árboles construidos por CPF
        """
        if self._cpf is not None:
            return self._cpf.return_forest()
        return self.forest

    def forest_tree_size(self) -> int:
        """
        Retorna el número de árboles construidos.

        Returns:
            Número de árboles en el bosque
        """
        if self._cpf is not None:
            return self._cpf.forest_tree_size()
        return len(self.forest._trees) if hasattr(self.forest, '_trees') else 0
