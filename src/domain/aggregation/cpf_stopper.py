"""
Adaptador ligero de ComparativeProgressiveForest para usarlo como criterio
de parada durante la agregación FL (sin necesidad de entrenar árboles nuevos).

El CPF original necesita un clasificador para buildEpisode; aquí sólo
necesitamos la lógica de convergencia sobre una secuencia de accuracy,
así que reimplementamos el criterio de parada de forma autocontenida.
"""


class ProgressiveStopper:
    """
    Replica el criterio de parada de CPF aplicado sobre una lista creciente de árboles.

    Uso:
        stopper = ProgressiveStopper()
        for tree in ranked_trees:
            selected.append(tree)
            if stopper.should_stop(episode_accuracies_so_far):
                break

    El criterio original:
        - Se evalúa por episodios de tamaño EPISODE.
        - Si max-min accuracies del episodio < CONVERGENCE por 2 episodios consecutivos → parar.
    """

    def __init__(self, convergence: float = 0.002, episode_size: int = 5,
                 consecutive_episodes: int = 2):
        self.convergence = convergence
        self.episode_size = episode_size
        self.consecutive_episodes = consecutive_episodes

    def should_stop(self, accuracies: list) -> bool:
        """
        Recibe la lista completa de accuracy de los árboles añadidos hasta ahora.
        Retorna True si se cumplen las condiciones de convergencia.
        """
        if len(accuracies) < self.episode_size:
            return False

        # Analizar último episodio completo
        last_ep = accuracies[-self.episode_size:]
        ep_range = max(last_ep) - min(last_ep)

        if not hasattr(self, '_stop_counter'):
            self._stop_counter = 0
            self._prev_range = None

        diff = ep_range - self._prev_range if self._prev_range is not None else ep_range

        if diff < self.convergence or ep_range < self.convergence:
            self._stop_counter += 1
        else:
            self._stop_counter = 0

        self._prev_range = ep_range
        return self._stop_counter >= self.consecutive_episodes

    def reset(self):
        self._stop_counter = 0
        self._prev_range = None
