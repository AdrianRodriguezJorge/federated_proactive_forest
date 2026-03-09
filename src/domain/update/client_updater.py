"""
No-Repeat Merge: actualiza el bosque del cliente con el global
excluyendo los árboles locales ya seleccionados durante la agregación.

bosque_extendido = local_no_seleccionados + todos_los_árboles_globales
"""
from typing import Any, List


class ClientUpdater:
    @staticmethod
    def merge(local_trees: List[Any], global_trees: List[Any],
              selected_local_ids: List[int]) -> List[Any]:
        """
        :param local_trees: Árboles del bosque local antes de la ronda.
        :param global_trees: Árboles del bosque global recibido del servidor.
        :param selected_local_ids: Índices locales que ya están en el bosque global.
        :return: Lista combinada sin duplicados.
        """
        selected_set = set(selected_local_ids)
        surviving_local = [t for i, t in enumerate(local_trees) if i not in selected_set]
        return surviving_local + global_trees
