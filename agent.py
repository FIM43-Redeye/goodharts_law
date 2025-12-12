import numpy as np

from constants import *
from environment import World


class Organism:
    def __init__(self, x: int, y: int, energy: float, sight_radius: int, world_ref: World):
        self.x = x
        self.y = y
        self.world = world_ref
        self.energy = energy
        self.alive = True
        self.sight_radius = sight_radius

    def move(self, dx: int, dy: int):
        self.x = max(0, min(self.x + dx, self.world.width - 1))
        self.y = max(0, min(self.y + dy, self.world.height - 1))
        # Calculate the cost of moving based on distance - no diagonal cheating!
        self.energy -= np.sqrt(dx ** 2 + dy ** 2) * ENERGY_MOVE_COST


    def eat(self):
        match self.world.grid[self.y, self.x]:
            case CellType.FOOD:
                self.energy += ENERGY_FOOD_REWARD
                self.world.grid[self.y, self.x] = CellType.EMPTY
            case CellType.POISON:
                self.energy -= ENERGY_POISON_PENALTY
                self.world.grid[self.y, self.x] = CellType.EMPTY
            case _:
                pass
        if self.energy <= 0:
            self.alive = False # I am dead. not big surprise

    def get_local_view(self) -> np.ndarray:
        """
        Extracts a square view of the world centered on the agent, padded with walls where the view goes off-map.
        """
        # Define the bounds of the view in world coordinates
        x_min_world = self.x - self.sight_radius
        x_max_world = self.x + self.sight_radius
        y_min_world = self.y - self.sight_radius
        y_max_world = self.y + self.sight_radius

        # Determine the slice to extract from the world grid, clamping to boundaries
        x_slice_start = max(0, x_min_world)
        x_slice_end = min(self.world.width, x_max_world + 1)
        y_slice_start = max(0, y_min_world)
        y_slice_end = min(self.world.height, y_max_world + 1)

        world_view = self.world.grid[y_slice_start:y_slice_end, x_slice_start:x_slice_end]

        # Calculate padding required for each side based on how much was clipped
        pad_top = y_slice_start - y_min_world
        pad_bottom = (y_max_world + 1) - y_slice_end
        pad_left = x_slice_start - x_min_world
        pad_right = (x_max_world + 1) - x_slice_end

        # Apply padding to create the final view.
        view = np.pad(
            world_view,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=CellType.WALL
        )

        return view

    def think(self) -> tuple[int, int]:
        view = self.get_local_view()
        center = self.sight_radius  # The agent is always at the center of the local view

        best_moves = []
        safe_moves = []

        # Iterate over the local view
        # We iterate relative coordinates (e.g., -2 to +2)
        for dy in range(-self.sight_radius, self.sight_radius + 1):
            for dx in range(-self.sight_radius, self.sight_radius + 1):

                # OPTIONAL: Circular Mask
                # If distance > radius, we can't "see" it. Skip.
                if np.sqrt(dx ** 2 + dy ** 2) > self.sight_radius:
                    continue

                # Don't check self (0,0)
                if dx == 0 and dy == 0:
                    continue

                # Look at the cell in our local view
                cell = view[center + dy, center + dx]

                # Logic: If I see something, how do I get there?
                # We need to normalize the direction to a single step (-1, 0, 1)
                step_x = np.sign(dx)
                step_y = np.sign(dy)

                # Check if the IMMEDIATE step is valid (not a wall)
                # We peek at the center neighbors of our view
                immediate_cell = view[center + step_y, center + step_x]
                if immediate_cell == CellType.WALL:
                    continue

                if cell == CellType.FOOD:
                    # If we see food, suggest moving 1 step towards it
                    best_moves.append((step_x, step_y))

                elif cell == CellType.EMPTY and immediate_cell == CellType.EMPTY:
                    # If we see empty space and the path is clear
                    safe_moves.append((step_x, step_y))

        # Decision Logic (Same as before)
        if best_moves:
            # Pick the move that minimizes distance to the food
            min_dist = float('inf')
            best_move = (0, 0)
            for dx, dy in best_moves:
                # Find the actual food location in the view that this move leads towards
                # We need to iterate through the view again to find the food
                for food_dy in range(-self.sight_radius, self.sight_radius + 1):
                    for food_dx in range(-self.sight_radius, self.sight_radius + 1):
                        if view[center + food_dy, center + food_dx] == CellType.FOOD:
                            dist = np.sqrt((food_dx - dx)**2 + (food_dy - dy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_move = (dx, dy)
            return best_move
        elif safe_moves:
            index = np.random.randint(0, len(safe_moves))
            return safe_moves[index]

        return 0, 0