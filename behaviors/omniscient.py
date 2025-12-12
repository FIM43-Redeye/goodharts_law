import numpy as np
from .base import BehaviorStrategy

# 8-directional random walk moves (cardinals + diagonals)
RANDOM_WALK_MOVES = [
    (0, 1), (0, -1), (1, 0), (-1, 0),  # Cardinals
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
]


class OmniscientSeeker(BehaviorStrategy):
    @property
    def requirements(self) -> list[str]:
        return ['ground_truth']

    def decide_action(self, agent, view):
        # 'view' is expected to be the ground truth grid
        center = agent.sight_radius
        CellType = agent.config['CellType']
        
        # Find all visible food and pick the closest one
        food_positions = []
        for dy in range(-agent.sight_radius, agent.sight_radius + 1):
            for dx in range(-agent.sight_radius, agent.sight_radius + 1):
                if np.sqrt(dx ** 2 + dy ** 2) > agent.sight_radius:
                    continue
                if view[center + dy, center + dx] == CellType.FOOD:
                    food_positions.append((dx, dy))
        
        if food_positions:
            # Move towards closest food
            min_dist = float('inf')
            closest_food = None
            for fx, fy in food_positions:
                dist = np.sqrt(fx ** 2 + fy ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_food = (fx, fy)
            
            dx, dy = closest_food
            step_x = int(np.sign(dx))
            step_y = int(np.sign(dy))
            
            # Avoid stepping on poison
            immediate_cell = view[center + step_y, center + step_x]
            if immediate_cell == CellType.POISON or immediate_cell == CellType.WALL:
                # Try cardinal directions only
                for alt_dx, alt_dy in [(step_x, 0), (0, step_y)]:
                    if alt_dx == 0 and alt_dy == 0:
                        continue
                    alt_cell = view[center + alt_dy, center + alt_dx]
                    if alt_cell != CellType.POISON and alt_cell != CellType.WALL:
                        return alt_dx, alt_dy
                return 0, 0  # Stuck - don't move
            
            return step_x, step_y
        
        # Random walk if no food visible (8 directions, avoid poison/walls)
        safe_moves = []
        for dx, dy in RANDOM_WALK_MOVES:
            cell = view[center + dy, center + dx]
            if cell != CellType.POISON and cell != CellType.WALL:
                safe_moves.append((dx, dy))
        
        if safe_moves:
            idx = np.random.randint(0, len(safe_moves))
            return safe_moves[idx]
        
        return 0, 0  # Completely stuck

