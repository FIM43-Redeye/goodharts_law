import numpy as np
from .base import BehaviorStrategy

# 8-directional random walk moves (cardinals + diagonals)
RANDOM_WALK_MOVES = [
    (0, 1), (0, -1), (1, 0), (-1, 0),  # Cardinals
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
]


class ProxySeeker(BehaviorStrategy):
    @property
    def requirements(self) -> list[str]:
        return ['proxy_metric']

    def decide_action(self, agent, view):
        # 'view' is expected to be the proxy metric grid (floats)
        center = agent.sight_radius
        # We want to move towards the highest value cell
        
        max_val = -1.0
        best_target = None
        
        # Scan visible area for highest metric
        for dy in range(-agent.sight_radius, agent.sight_radius + 1):
            for dx in range(-agent.sight_radius, agent.sight_radius + 1):
                if np.sqrt(dx ** 2 + dy ** 2) > agent.sight_radius:
                    continue
                if dx == 0 and dy == 0:
                    continue
                
                val = view[center + dy, center + dx]
                if val > max_val and val > 0:  # Interested in non-zero signals
                    max_val = val
                    best_target = (dx, dy)
        
        if best_target:
            # Move towards best_target
            dx, dy = best_target
            step_x = int(np.sign(dx))
            step_y = int(np.sign(dy))
            return step_x, step_y
        
        # Random walk if nothing interesting (8 directions)
        idx = np.random.randint(0, len(RANDOM_WALK_MOVES))
        return RANDOM_WALK_MOVES[idx]

