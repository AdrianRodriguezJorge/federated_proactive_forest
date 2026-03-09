"""
Módulo para Comparative Progressive Forest (CPF).

Modificación mínima respecto al original: se añade parámetro verbose (default=False)
para suprimir los print() en producción FL. La lógica es idéntica.
"""
from sklearn.utils import check_X_y
import numpy as np


class ComparativeProgressiveForest:
    """
    Comparative Progressive Forest (CPF).
    Entrena un ProactiveForest en episodios con early stopping por convergencia.

    Parámetros:
    - CONVERGENCE: Umbral de convergencia (0.002).
    - EPISODE: Tamaño inicial del episodio (5).
    """

    def __init__(self, classifier, verbose: bool = False):
        """
        :param classifier: Instancia de ProactiveForestClassifier.
        :param verbose: Si True, imprime logs de progreso (útil para debug/notebooks).
        """
        self._classifier = classifier
        self.CONVERGENCE = 0.002
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
        X, y = check_X_y(X, y, dtype=None)
        Xt, yt = check_X_y(Xt, yt, dtype=None)
        models_built = 0
        stop_counter = 0
        previous_episode_accuracy = None
        episode_accuracy_dif = 0.002
        self.EPISODE = 5
        self._classifier.clean_trees()

        while models_built < self._classifier.n_estimators:
            self._classifier.buildEpisode(X, y, Xt, yt, self.EPISODE,
                                          verbose=self.verbose)
            models_built += self._classifier.EPISODE

            min_accuracy = min(self._classifier._m_progressive_accuracy)
            max_accuracy = max(self._classifier._m_progressive_accuracy)
            episode_accuracy = max_accuracy - min_accuracy

            if previous_episode_accuracy is not None:
                episode_accuracy_dif = episode_accuracy - previous_episode_accuracy

            if episode_accuracy_dif < self.CONVERGENCE or episode_accuracy < self.CONVERGENCE:
                stop_counter += 1
                self.EPISODE = max(1, self.EPISODE - 1)
                if self.verbose:
                    print(f"  [CPF] Condición de parada cumplida (stop_counter={stop_counter})")
                if stop_counter == 2:
                    break
            else:
                stop_counter = 0
                self.EPISODE += 1
                if self.verbose:
                    print(f"  [CPF] No converge, siguiente episodio={self.EPISODE}")

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
