"""
Omniscient behavior that sees ground truth cell types.

Uses one-hot encoded channels where:
- Channel 0: is_empty
- Channel 1: is_wall  
- Channel 2: is_food
- Channel 3: is_poison
"""
import numpy as np
from .base import BehaviorStrategy


# 8-directional random walk moves (cardinals + diagonals)
RANDOM_WALK_MOVES = [
    (0, 1), (0, -1), (1, 0), (-1, 0),   # Cardinals
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
]

# Channel indices for one-hot encoding
CHANNEL_EMPTY = 0
CHANNEL_WALL = 1
CHANNEL_FOOD = 2
CHANNEL_POISON = 3


class OmniscientSeeker(BehaviorStrategy):
    """
    Behavior that can see exactly what each cell contains.
    
    Uses one-hot encoded view with separate channels for each cell type.
    This agent can distinguish food from poison perfectly.
    """
    
    @property
    def requirements(self) -> list[str]:
        return ['ground_truth']

    def decide_action(self, agent, view: np.ndarray) -> tuple[int, int]:
        """
        Decide action based on full ground-truth observation.
        
        Args:
            agent: The organism instance
            view: Shape (num_channels, H, W) with one-hot encoded cells
            
        Returns:
            (dx, dy) movement vector
        """
        center = agent.sight_radius
        
        # Extract relevant channels
        food_channel = view[CHANNEL_FOOD]      # (H, W) - 1.0 where food exists
        poison_channel = view[CHANNEL_POISON]  # (H, W) - 1.0 where poison exists
        wall_channel = view[CHANNEL_WALL]      # (H, W) - 1.0 where walls exist
        
        # Find all visible food positions
        food_positions = []
        for dy in range(-agent.sight_radius, agent.sight_radius + 1):
            for dx in range(-agent.sight_radius, agent.sight_radius + 1):
                if np.sqrt(dx ** 2 + dy ** 2) > agent.sight_radius:
                    continue
                if food_channel[center + dy, center + dx] > 0.5:  # Using 0.5 threshold
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
            
            # Check if immediate step would hit poison or wall
            is_poison = poison_channel[center + step_y, center + step_x] > 0.5
            is_wall = wall_channel[center + step_y, center + step_x] > 0.5
            
            if is_poison or is_wall:
                # Try cardinal directions only
                for alt_dx, alt_dy in [(step_x, 0), (0, step_y)]:
                    if alt_dx == 0 and alt_dy == 0:
                        continue
                    alt_poison = poison_channel[center + alt_dy, center + alt_dx] > 0.5
                    alt_wall = wall_channel[center + alt_dy, center + alt_dx] > 0.5
                    if not alt_poison and not alt_wall:
                        return alt_dx, alt_dy
                return 0, 0  # Stuck, don't move
            
            return step_x, step_y
        
        # Random walk if no food visible, avoiding poison and walls
        safe_moves = []
        for dx, dy in RANDOM_WALK_MOVES:
            is_poison = poison_channel[center + dy, center + dx] > 0.5
            is_wall = wall_channel[center + dy, center + dx] > 0.5
            if not is_poison and not is_wall:
                safe_moves.append((dx, dy))
        
        if safe_moves:
            idx = np.random.randint(0, len(safe_moves))
            return safe_moves[idx]
        
        return 0, 0  # Completely stuck
