from abc import ABC, abstractmethod


# Role default colors (prey=cyan family, predator=red family)
ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    'prey': (0, 255, 255),      # Cyan
    'predator': (255, 0, 0),    # Red
}


class BehaviorStrategy(ABC):
    """Base class for agent behaviors.
    
    Attributes:
        role: What type of cell this agent appears as ('prey' or 'predator')
        color: RGB tuple for visualization (defaults to role color if not overridden)
    """
    
    # Class-level color override (None = use role default)
    _color: tuple[int, int, int] | None = None
    
    @property
    def role(self) -> str:
        """Agent role determines its cell type on the grid.
        
        Default is 'prey'. Override to 'predator' for hunter agents.
        """
        return 'prey'
    
    @property
    def color(self) -> tuple[int, int, int]:
        """RGB color for visualization. Defaults to role color if not overridden."""
        if self._color is not None:
            return self._color
        return ROLE_COLORS.get(self.role, (128, 128, 128))
    
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
