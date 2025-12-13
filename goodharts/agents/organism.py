"""
Agent (Organism) implementation with flexible observation system.

Organisms navigate the world, consume resources, and can be controlled
by various behavior strategies (hardcoded or learned).
"""
import numpy as np
from goodharts.environments.world import World
from goodharts.behaviors import BehaviorStrategy
from goodharts.utils.logging_config import get_logger
from dataclasses import dataclass

logger = get_logger("agent")


@dataclass
class Observation:
    """
    Flexible observation container for agent perception.
    
    Supports:
    - Multiple grid channels (proxy, ground_truth, density, etc.)
    - Scalar values (energy, step count, population, etc.)
    
    The CNN/behavior can select which channels to use.
    """
    grids: dict[str, np.ndarray]  # Named grids, each (H, W)
    scalars: dict[str, float]     # Named scalar values
    
    def get_stacked_grids(self, channel_names: list[str]) -> np.ndarray:
        """
        Stack selected grids into a multi-channel array.
        
        Args:
            channel_names: List of grid names to stack in order
            
        Returns:
            Array of shape (num_channels, H, W)
        """
        channels = [self.grids[name] for name in channel_names if name in self.grids]
        if not channels:
            raise ValueError(f"No valid channels found. Available: {list(self.grids.keys())}")
        return np.stack(channels, axis=0)
    
    def get_default_grid(self, mode: str = 'ground_truth') -> np.ndarray:
        """Get single grid for backward compatibility."""
        if mode == 'proxy' and 'proxy' in self.grids:
            return self.grids['proxy']
        return self.grids.get('ground_truth', next(iter(self.grids.values())))


class Organism:
    """
    An agent that navigates the world, consumes resources, and learns.
    
    Attributes:
        x, y: Position in world
        energy: Current energy level (dies when <= 0)
        alive: Whether agent is still active
        sight_radius: How far agent can see
        behavior: Strategy that decides actions
    """
    
    def __init__(self, x: int, y: int, energy: float, sight_radius: int, 
                 world_ref: 'World', behavior: 'BehaviorStrategy', config: dict):
        self.x: int = x
        self.y: int = y
        self.world: 'World' = world_ref
        self.energy: float = energy
        self.initial_energy: float = energy  # For normalization
        self.alive: bool = True
        self.sight_radius: int = sight_radius
        self.behavior: 'BehaviorStrategy' = behavior
        self.config: dict = config
        
        self.id: int = id(self)
        self.death_reason: str | None = None
        self.suspicion_score: int = 0
        self.steps_alive: int = 0
        
        self.check_compatibility()

    def check_compatibility(self):
        """Verify behavior requirements match world capabilities."""
        reqs = self.behavior.requirements
        caps = self.world.capabilities
        for req in reqs:
            if req not in caps:
                raise ValueError(f"Incompatible: Behavior requires '{req}' but World only supports {caps}")

    def move(self, dx: int | float, dy: int | float):
        """
        Move by (dx, dy), respecting speed limits.
        
        Energy cost is applied for the ATTEMPTED move distance,
        so agents still pay for trying to move into walls.
        World handles coordinate normalization (wrap or clamp).
        """
        # Handle continuous actions
        dx = int(round(dx))
        dy = int(round(dy))
        
        # Apply speed cap
        max_dist = self.config.get('MAX_MOVE_DISTANCE', 3)
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > max_dist:
            scale = max_dist / distance
            dx = int(round(dx * scale))
            dy = int(round(dy * scale))
            distance = np.sqrt(dx ** 2 + dy ** 2)
        
        # Update position (world handles topology - wrap or clamp)
        self.x, self.y = self.world.wrap_position(self.x + dx, self.y + dy)
        
        # Nonlinear energy cost (applied for attempted distance)
        if not self.config.get('FREEZE_ENERGY', False):
            exponent = self.config.get('MOVE_COST_EXPONENT', 1.0)
            base_cost = self.config['ENERGY_MOVE_COST']
            self.energy -= (distance ** exponent) * base_cost

    def eat(self):
        """Consume whatever is at current position and trigger respawning."""
        cell_value = self.world.grid[self.y, self.x]
        CellType = self.config['CellType']
        cell_info = CellType.by_value(cell_value)
        
        if cell_info and (cell_info.energy_reward > 0 or cell_info.energy_penalty > 0):
            # Suspicion tracking
            proxy_signal = self.world.proxy_grid[self.y, self.x]
            if proxy_signal > 0.5 and cell_info.energy_penalty > 0:
                self.suspicion_score += 1
            
            # Apply energy changes (unless frozen)
            if not self.config.get('FREEZE_ENERGY', False):
                self.energy += cell_info.energy_reward
                self.energy -= cell_info.energy_penalty
            
            # DEBUG: Confirm eating
            # print(f"DEBUG EAT: Found {cell_info.name} (Val={cell_value}) Reward={cell_info.energy_reward}", flush=True)
            
            # Clear consumed cell
            self.world.grid[self.y, self.x] = CellType.EMPTY
            self.world.proxy_grid[self.y, self.x] = 0.0
            
            # Respawn consumed resource elsewhere (keeps simulation running indefinitely)
            # Respawn consumed resource elsewhere
            if self.config.get('RESPAWN_RESOURCES', True):
                if cell_info.energy_reward > 0:
                    self.world.place_food(1)
                elif cell_info.energy_penalty > 0:
                    self.world.place_poison(1)
            
            
            return (cell_info.name.upper(), cell_info.energy_reward if cell_info.energy_reward > 0 else cell_info.energy_penalty)
        
        # Check for death
        if self.energy <= 0:
            self.alive = False
            if cell_info and cell_info.energy_penalty > 0 and self.energy + cell_info.energy_penalty > 0:
                self.death_reason = "Poison"
            else:
                self.death_reason = "Starvation"
            logger.debug(f"Agent {self.id} died: {self.death_reason}")

    def _extract_grid_view(self, grid: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """Extract local view from a grid - delegates to world."""
        return self.world.get_view(grid, self.x, self.y, self.sight_radius, fill_value)

    def _build_onehot_channels(self, grid_view: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert integer cell grid to one-hot encoded channels.
        
        Creates one binary channel per cell type:
        - 'cell_empty': 1.0 where cell is empty, 0 elsewhere
        - 'cell_wall': 1.0 where cell is wall, etc.
        
        Center cell is blanked so agent can't "see itself".
        Extensible: automatically handles any number of cell types.
        """
        CellType = self.config['CellType']
        channels = {}
        
        for cell_info in CellType.all_types():
            channel_name = f"cell_{cell_info.name.lower()}"
            channels[channel_name] = (grid_view == cell_info.value).astype(np.float32)
        
        # Blank center cell - agent can't see itself
        center = self.sight_radius
        for channel in channels.values():
            channel[center, center] = 0.0
        
        return channels

    def _build_property_channels(self, grid_view: np.ndarray) -> dict[str, np.ndarray]:
        """
        Build channels for cell properties (interestingness, etc.).
        
        Each property becomes a channel where each cell's value is that
        property for that cell type.
        
        Extensible: automatically handles any CellTypeInfo properties.
        """
        CellType = self.config['CellType']
        properties = ['interestingness', 'energy_reward', 'energy_penalty']
        channels = {}
        
        for prop in properties:
            channel = np.zeros_like(grid_view, dtype=np.float32)
            for cell_info in CellType.all_types():
                mask = (grid_view == cell_info.value)
                channel[mask] = cell_info.get_property(prop)
            channels[prop] = channel
        
        return channels

    def get_observation(self) -> Observation:
        """
        Get full observation with all available channels.
        
        Channels include:
        - One-hot per cell type (cell_empty, cell_wall, cell_food, cell_poison)
        - Property channels (interestingness, energy_reward, energy_penalty)
        - Raw grids (ground_truth_raw, proxy_raw) for backward compatibility
        
        Returns:
            Observation with grids and scalars
        """
        CellType = self.config['CellType']
        
        # Extract raw grid views
        raw_grid = self._extract_grid_view(
            self.world.grid, 
            fill_value=CellType.WALL.value
        )
        proxy_grid = self._extract_grid_view(
            self.world.proxy_grid, 
            fill_value=0.0
        )
        
        # Build all channel types
        grids = {}
        
        # One-hot encoded cell types (for ground truth mode)
        grids.update(self._build_onehot_channels(raw_grid))
        
        # Property-based channels (interestingness is what proxy mode sees)
        grids.update(self._build_property_channels(raw_grid))
        
        # Raw values for backward compatibility
        grids['ground_truth_raw'] = raw_grid.astype(np.float32)
        grids['proxy_raw'] = proxy_grid
        
        # Scalars (agent's internal state)
        scalars = {
            'energy': self.energy,
            'energy_normalized': np.clip(self.energy / self.initial_energy, 0.0, 2.0),
            'steps_alive': float(self.steps_alive),
            'x_normalized': self.x / self.world.width,
            'y_normalized': self.y / self.world.height,
        }
        
        return Observation(grids=grids, scalars=scalars)

    def get_local_view(self, mode: str | None = None) -> np.ndarray:
        """
        Get appropriately formatted view for the current behavior mode.
        
        Uses ObservationSpec to determine which channels to include,
        making this method extensible without hardcoding mode logic.
        
        Args:
            mode: Observation mode (e.g., 'ground_truth', 'proxy').
                  If None, derived from behavior.requirements.
                  
        Returns:
            Stacked observation array of shape (num_channels, H, W)
        """
        from goodharts.configs.observation_spec import ObservationSpec, get_mode_for_requirement
        
        obs = self.get_observation()
        
        # Determine mode from behavior if not specified
        if mode is None:
            # Use first requirement to determine mode
            reqs = self.behavior.requirements
            req = reqs[0] if reqs else 'ground_truth'
            mode = get_mode_for_requirement(req, self.config)
        
        # Get channel names from the spec
        spec = ObservationSpec.for_mode(mode, self.config)
        channel_names = spec.channel_names
        
        # Build observation by stacking requested channels
        channels = []
        for name in channel_names:
            if name in obs.grids:
                channels.append(obs.grids[name])
            else:
                # Unknown channel - log warning and use zeros
                import warnings
                warnings.warn(f"Unknown channel '{name}' for mode '{mode}', using zeros")
                example = next(iter(obs.grids.values()))
                channels.append(np.zeros_like(example))
        
        return np.stack(channels, axis=0)

    def update(self):
        """One simulation step: observe, decide, act."""
        self.steps_alive += 1
        view = self.get_local_view()
        dx, dy = self.behavior.decide_action(self, view)
        self.move(dx, dy)
        self.eat()

