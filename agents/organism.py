import numpy as np
from environment import World
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

    def move(self, dx: int, dy: int):
        self.x = max(0, min(self.x + dx, self.world.width - 1))
        self.y = max(0, min(self.y + dy, self.world.height - 1))
        self.energy -= np.sqrt(dx ** 2 + dy ** 2) * self.config['ENERGY_MOVE_COST']

    def eat(self):
        cell_type = self.world.grid[self.y, self.x]
        CellType = self.config['CellType']
        if cell_type in self.config['CELL_PROPERTIES']:
            properties = self.config['CELL_PROPERTIES'][cell_type]
            if 'energy_reward' in properties:
                self.energy += properties['energy_reward']
            if 'energy_penalty' in properties:
                self.energy -= properties['energy_penalty']
            self.world.grid[self.y, self.x] = CellType.EMPTY
        
        if self.energy <= 0:
            self.alive = False

    def get_local_view(self) -> np.ndarray:
        x_min_world = self.x - self.sight_radius
        x_max_world = self.x + self.sight_radius
        y_min_world = self.y - self.sight_radius
        y_max_world = self.y + self.sight_radius

        x_slice_start = max(0, x_min_world)
        x_slice_end = min(self.world.width, x_max_world + 1)
        y_slice_start = max(0, y_min_world)
        y_slice_end = min(self.world.height, y_max_world + 1)

        world_view = self.world.grid[y_slice_start:y_slice_end, x_slice_start:x_slice_end]

        pad_top = y_slice_start - y_min_world
        pad_bottom = (y_max_world + 1) - y_slice_end
        pad_left = x_slice_start - x_min_world
        pad_right = (x_max_world + 1) - x_slice_end

        view = np.pad(
            world_view,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=self.config['CellType'].WALL
        )
        return view

    def update(self):
        view = self.get_local_view()
        dx, dy = self.behavior.decide_action(self, view)
        self.move(dx, dy)
        self.eat()
