from abc import ABC, abstractmethod


class BehaviorStrategy(ABC):
    """Base class for agent behaviors.

    Agents are tracked by coordinates only (not marked on grid).
    Each behavior has a color for visualization.
    """

    # Class-level color (gray default, override in subclasses)
    _color: tuple[int, int, int] = (128, 128, 128)

    @property
    def color(self) -> tuple[int, int, int]:
        """RGB color for visualization."""
        return self._color
    
    @property
    @abstractmethod
    def requirements(self) -> list[str]:
        """Returns a list of requirements this behavior needs from the environment.
        Example: ['ground_truth'] or ['proxy_metric']
        """
        pass

    @abstractmethod
    def decide_action(self, agent, view) -> tuple[int, int]:
        """
        Determine the next action for the agent based on its local view.

        Args:
            agent: The AgentWrapper instance (provides properties like sight_radius)
            view: Tensor of shape (C, H, W) containing the agent's local observation

        Returns:
            (dx, dy) tuple representing the movement direction
        """
        pass
