"""
Proxy-optimizing behavior that only sees interestingness signals.

In proxy mode, all channels contain the same interestingness values,
so this agent cannot distinguish food (interestingness=1.0) from 
poison (interestingness=0.9). This is the Goodhart's Law trap.
"""
import numpy as np
from .base import BehaviorStrategy


# 8-directional random walk moves (cardinals + diagonals)
RANDOM_WALK_MOVES = [
    (0, 1), (0, -1), (1, 0), (-1, 0),   # Cardinals
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
]


class ProxySeeker(BehaviorStrategy):
    """
    Behavior that optimizes for proxy signal (interestingness).
    
    Receives the same (num_channels, H, W) shape as OmniscientSeeker,
    but all channels contain identical interestingness values.
    Cannot distinguish food from poison - will eat both!
    """
    
    @property
    def requirements(self) -> list[str]:
        return ['proxy_metric']

    def decide_action(self, agent, view: np.ndarray) -> tuple[int, int]:
        """
        Decide action based on proxy signal only.
        
        Args:
            agent: The organism instance
            view: Shape (num_channels, H, W) - all channels have same interestingness
            
        Returns:
            (dx, dy) movement vector
        """
        center = agent.sight_radius
        
        # In proxy mode, all channels are identical interestingness values
        # Just use channel 0 (they're all the same)
        proxy_signal = view[0]  # (H, W) - interestingness values
        
        # Find highest interestingness within sight radius
        max_val = -1.0
        best_target = None
        
        for dy in range(-agent.sight_radius, agent.sight_radius + 1):
            for dx in range(-agent.sight_radius, agent.sight_radius + 1):
                if np.sqrt(dx ** 2 + dy ** 2) > agent.sight_radius:
                    continue
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                
                val = proxy_signal[center + dy, center + dx]
                if val > max_val and val > 0.01:  # Interested in non-zero signals
                    max_val = val
                    best_target = (dx, dy)
        
        if best_target:
            # Move towards highest signal (could be food OR poison!)
            dx, dy = best_target
            step_x = int(np.sign(dx))
            step_y = int(np.sign(dy))
            return step_x, step_y
        
        # Random walk if nothing interesting visible
        idx = np.random.randint(0, len(RANDOM_WALK_MOVES))
        return RANDOM_WALK_MOVES[idx]
