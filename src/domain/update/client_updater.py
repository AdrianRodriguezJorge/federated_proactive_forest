"""No-Repeat Merge for Client Forest Updates.

Updates the client's local forest with the globally aggregated server forest
by excluding any local trees that were already selected during the server
aggregation round.
"""

from typing import Any, List


class ClientUpdater:
    """Handles updating client local forests using global forests."""

    @staticmethod
    def merge(
        local_trees: List[Any],
        global_trees: List[Any],
        selected_local_ids: List[int],
    ) -> List[Any]:
        """Merges global and local forests excluding selected local trees.

        Args:
            local_trees (List[Any]): Client's local trees before the round.
            global_trees (List[Any]): Global trees received from the server.
            selected_local_ids (List[int]): Local indices already in the
                global forest.

        Returns:
            List[Any]: Merged forest containing surviving local and all global.
        """
        selected_set = set(selected_local_ids)
        surviving_local = [
            t for i, t in enumerate(local_trees) if i not in selected_set
        ]
        return surviving_local + global_trees
