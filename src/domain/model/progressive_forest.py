"""
Progressive Forest wrapper usando ComparativeProgressiveForest (CPF).

Este módulo proporciona una interfaz simplificada para entrenar Proactive Forest
con early stopping basado en convergencia, utilizando la implementación original
de CPF en cpf_implementation/newalg.py.

Uso:
    from src.domain.model.progressive_forest import ProgressiveForest
    
    pf = ProgressiveForest(proactive_forest_instance, verbose=False)
    pf.fit_with_early_stopping(X_train, y_train, X_val, y_val)
    forest = pf.get_forest()
"""
from typing import Any
import numpy as np
from .cpf_implementation.newalg import ComparativeProgressiveForest


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

    def __init__(self, forest: Any, verbose: bool = False):
        """
        Inicializa Progressive Forest.
        
        Args:
            forest: Instancia de ProactiveForest o ProactiveForestClassifier
            verbose: Si True, imprime logs de progreso durante el entrenamiento
        """
        self.forest = forest
        self.verbose = verbose
        self._cpf = None

    def fit_with_early_stopping(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> 'ProgressiveForest':
        """
        Entrena el bosque con early stopping basado en convergencia.
        
        El algoritmo construye árboles en episodios y se detiene cuando
        la diferencia de accuracy entre episodios consecutivos es menor
        al umbral de convergencia durante 2 episodios consecutivos.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación (para medir convergencia)
            y_val: Etiquetas de validación
            
        Returns:
            Self para encadenamiento de métodos
        """
        # Obtener el clasificador subyacente del forest
        if hasattr(self.forest, '_classifier'):
            classifier = self.forest._classifier
        else:
            classifier = self.forest
        
        # Crear y ejecutar CPF con la implementación original
        self._cpf = ComparativeProgressiveForest(classifier, verbose=self.verbose)
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
