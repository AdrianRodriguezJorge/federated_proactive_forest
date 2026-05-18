"""Mandatory metadata sent from each client to the server with their trees."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClientMetadata:
    """Metadata representing the state of a client during federation.

    Attributes:
        client_id (str): Unique identifier of the client. Defaults to "unknown".
        n_trees (int): Number of trees currently in client's local forest.
            Defaults to 0.
        selected_local_tree_ids (List[int]): IDs of the client's local trees
            that were chosen during aggregation. Used to implement the
            No-Repeat Merge mechanism. Defaults to an empty list.
        has_converged (bool): Whether the client has met local convergence
            criteria. Defaults to False.
        stop_counter (int): Number of consecutive rounds the client's local
            forest evaluation has failed to improve. Defaults to 0.
        prev_episode_acc (float): Accuracy obtained in the previous round
            or training episode. Defaults to 0.0.
    """

    client_id: str = "unknown"
    n_trees: int = 0
    selected_local_tree_ids: List[int] = field(default_factory=list)
    has_converged: bool = False
    stop_counter: int = 0
    prev_episode_acc: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientMetadata":
        """Create a ClientMetadata instance from a dictionary.

        Filters out any keys not defined as attributes in the dataclass
        to prevent initialization crashes.

        Args:
            data (Dict[str, Any]): Dictionary containing configuration fields.

        Returns:
            ClientMetadata: A newly created metadata instance.
        """
        import dataclasses

        fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata properties into a standard dictionary.

        Returns:
            Dict[str, Any]: Key-value representation of the metadata.
        """
        return {
            "client_id": self.client_id,
            "n_trees": self.n_trees,
            "has_converged": self.has_converged,
            "stop_counter": self.stop_counter,
            "prev_episode_acc": self.prev_episode_acc,
        }
