import numpy as np
from environments.world import World
from behaviors import BehaviorStrategy


class Organism:
    def __init__(self, x: int, y: int, energy: float, sight_radius: int, world_ref: World, behavior: BehaviorStrategy, config: dict):
        self.x = x
        self.y = y
        self.world = world_ref
        self.energy = energy
        self.alive = True
        self.sight_radius = sight_radius
        self.behavior = behavior
        self.config = config
        
        self.check_compatibility()

    def check_compatibility(self):
        reqs = self.behavior.requirements
        caps = self.world.capabilities
        for req in reqs:
            if req not in caps:
                raise ValueError(f"Incompatible: Behavior requires '{req}' but World only supports {caps}")

    def move(self, dx: int, dy: int):
        # Apply speed cap - organisms can't move more than MAX_MOVE_DISTANCE per step
        max_dist = self.config.get('MAX_MOVE_DISTANCE', 3)
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > max_dist:
            # Scale down the movement to stay within speed limit
            scale = max_dist / distance
            dx = int(round(dx * scale))
            dy = int(round(dy * scale))
            distance = np.sqrt(dx ** 2 + dy ** 2)
        
        # Update position (clamped to world bounds)
        self.x = max(0, min(self.x + dx, self.world.width - 1))
        self.y = max(0, min(self.y + dy, self.world.height - 1))
        
        # Nonlinear energy cost: distance^exponent * base_cost
        # Exponent > 1 means farther = disproportionately more expensive
        exponent = self.config.get('MOVE_COST_EXPONENT', 1.0)
        base_cost = self.config['ENERGY_MOVE_COST']
        self.energy -= (distance ** exponent) * base_cost

    def eat(self):
        cell_value = self.world.grid[self.y, self.x]
        CellType = self.config['CellType']
        cell_info = CellType.by_value(cell_value)
        
        if cell_info and (cell_info.energy_reward > 0 or cell_info.energy_penalty > 0):
            self.energy += cell_info.energy_reward
            self.energy -= cell_info.energy_penalty
            self.world.grid[self.y, self.x] = CellType.EMPTY
            self.world.proxy_grid[self.y, self.x] = 0.0  # Clear signal
        
        if self.energy <= 0:
            self.alive = False

    def get_local_view(self) -> np.ndarray:
        # Determine what view to fetch based on behavior requirements
        # Default to ground truth if not specified or mixed (simpler for now)
        reqs = self.behavior.requirements
        
        target_grid = self.world.grid
        constant_val = self.config['CellType'].WALL
        
        if 'proxy_metric' in reqs:
            target_grid = self.world.proxy_grid
            constant_val = 0.0 # Outside world is 0 signal
        
        x_min_world = self.x - self.sight_radius
        x_max_world = self.x + self.sight_radius
        y_min_world = self.y - self.sight_radius
        y_max_world = self.y + self.sight_radius

        x_slice_start = max(0, x_min_world)
        x_slice_end = min(self.world.width, x_max_world + 1)
        y_slice_start = max(0, y_min_world)
        y_slice_end = min(self.world.height, y_max_world + 1)

        world_view = target_grid[y_slice_start:y_slice_end, x_slice_start:x_slice_end]

        pad_top = y_slice_start - y_min_world
        pad_bottom = (y_max_world + 1) - y_slice_end
        pad_left = x_slice_start - x_min_world
        pad_right = (x_max_world + 1) - x_slice_end

        view = np.pad(
            world_view,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=constant_val
        )
        return view

    def update(self):
        view = self.get_local_view()
        dx, dy = self.behavior.decide_action(self, view)
        self.move(dx, dy)
        self.eat()
