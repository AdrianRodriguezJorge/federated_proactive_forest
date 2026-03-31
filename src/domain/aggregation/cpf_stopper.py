"""
Adaptador ligero de ComparativeProgressiveForest para usarlo como criterio
de parada durante la agregación FL (sin necesidad de entrenar árboles nuevos).

El CPF original necesita un clasificador para buildEpisode; aquí sólo
necesitamos la lógica de convergencia sobre una secuencia de accuracy,
así que reimplementamos el criterio de parada de forma autocontenida.

CRITERIO ORIGINAL (newalg.py - Cepero 2023, líneas 43-47):
    min_accuracy = min(self._classifier._m_progressive_accuracy)
    max_accuracy = max(self._classifier._m_progressive_accuracy)
    episode_accuracy = max_accuracy - min_accuracy
    
    if episode_accuracy_dif < CONVERGENCE or episode_accuracy < CONVERGENCE:
        stop_counter += 1
        if stop_counter == 2:
            break
"""


class ProgressiveStopper:
    """
    Replica el criterio de parada de CPF aplicado sobre una lista creciente de árboles.

    Uso:
        stopper = ProgressiveStopper()
        for tree in ranked_trees:
            selected.append(tree)
            acc = evaluate_tree(tree, X_val, y_val)
            accuracies.append(acc)
            if stopper.should_stop(accuracies):
                break

    El criterio original (Cepero 2023):
        - episode_accuracy = max(TODOS los accuracies) - min(TODOS los accuracies)
        - episode_accuracy_dif = episode_accuracy_actual - episode_accuracy_anterior
        - Parar si: episode_accuracy_dif < CONVERGENCE OR episode_accuracy < CONVERGENCE
        - Durante 2 episodios consecutivos
    """

    def __init__(self, convergence: float = 0.002, episode_size: int = 5,
                 consecutive_episodes: int = 2):
        """
        Inicializa el stopper progresivo.
        
        Args:
            convergence: Umbral de convergencia (0.002 por defecto, igual que CPF original)
            episode_size: Tamaño del episodio (k en la tesis)
            consecutive_episodes: Episodios consecutivos para confirmar convergencia
        """
        self.convergence = convergence
        self.episode_size = episode_size
        self.consecutive_episodes = consecutive_episodes

    def should_stop(self, accuracies: list) -> bool:
        """
        Determina si debe detenerse basado en el criterio de convergencia de CPF.
        
        Args:
            accuracies: Lista completa de accuracies de todos los árboles añadidos
            
        Returns:
            True si se cumplen las condiciones de convergencia, False en otro caso
            
        Criterio idéntico al CPF original (newalg.py):
            min_accuracy = min(all_accuracies)
            max_accuracy = max(all_accuracies)
            episode_accuracy = max_accuracy - min_accuracy
            episode_accuracy_dif = episode_accuracy - previous_episode_accuracy
            
            if episode_accuracy_dif < CONVERGENCE or episode_accuracy < CONVERGENCE:
                stop_counter += 1
                if stop_counter == 2:
                    return True  # Detener agregación
        """
        # Necesita al menos un episodio completo para evaluar
        if len(accuracies) < self.episode_size:
            return False

        # CRITERIO ORIGINAL: max - min de TODOS los accuracies acumulados
        # (no solo del último episodio como estaba antes)
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)
        episode_accuracy = max_accuracy - min_accuracy

        if not hasattr(self, '_stop_counter'):
            self._stop_counter = 0
            self._prev_accuracy = None

        # Calcular diferencia con el episodio anterior
        if self._prev_accuracy is not None:
            episode_accuracy_dif = episode_accuracy - self._prev_accuracy
        else:
            episode_accuracy_dif = episode_accuracy

        # Criterio de convergencia ORIGINAL (newalg.py líneas 46-47)
        if episode_accuracy_dif < self.convergence or episode_accuracy < self.convergence:
            self._stop_counter += 1
            if self._stop_counter >= self.consecutive_episodes:
                return True
        else:
            self._stop_counter = 0

        self._prev_accuracy = episode_accuracy
        return False

    def reset(self):
        """Reinicia el contador para una nueva secuencia de agregación."""
        self._stop_counter = 0
        self._prev_accuracy = None
