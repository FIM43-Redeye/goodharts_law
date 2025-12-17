"""
Proxy-optimizing behavior that only sees interestingness signals.

In proxy mode, all channels contain the same interestingness values,
so this agent cannot distinguish food (interestingness=1.0) from 
poison (interestingness=0.9). This is the Goodhart's Law trap.
"""
import torch
from goodharts.behaviors.base import BehaviorStrategy


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
    
    # Distinct color for visualization (magenta)
    _color = (255, 0, 255)
    
    @property
    def requirements(self) -> list[str]:
        return ['proxy_metric']

    def decide_action(self, agent, view: torch.Tensor) -> tuple[int, int]:
        """
        Decide action based on proxy signal only.
        
        Args:
            agent: The organism instance
            view: Shape (num_channels, H, W) tensor - all channels have same interestingness
            
        Returns:
            (dx, dy) movement vector
        """
        # Ensure view is on correct device (it should be, but good to be safe if moved)
        # view is (C, H, W)
        device = view.device
        center = agent.sight_radius
        
        # In proxy mode, all channels are identical interestingness values
        # Just use channel 0 (they're all the same)
        proxy_signal = view[0]  # (H, W)
        
        # We need to find the specific (dx, dy) with the max value.
        # Mask out the center (agent itself) to avoid staying put if current tile is high?
        # Original logic: "if dx==0 and dy==0: continue" -> Yes.
        
        input_view = proxy_signal.clone()
        input_view[center, center] = -1.0 # Mask center
        
        # Mask out area outside sight radius (circle)
        # Create a grid of coordinates
        H, W = input_view.shape
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist_sq = (x_grid - center)**2 + (y_grid - center)**2
        radius_sq = agent.sight_radius**2
        
        # Mask outside radius
        input_view[dist_sq > radius_sq] = -1.0
        
        # Find max value
        max_val = torch.max(input_view)
        
        if max_val > 0.01:
            # Get indices of max value
            # (If multiple, argmax finds first flattened index)
            # To handle ties randomly or consistently? Original used loop order (top-left first).
            # We can just use argmax for speed.
            
            flat_idx = torch.argmax(input_view)
            best_y = (flat_idx // W).item()
            best_x = (flat_idx % W).item()
            
            dx = best_x - center
            dy = best_y - center
            
            step_x = int(torch.sign(torch.tensor(dx)).item())
            step_y = int(torch.sign(torch.tensor(dy)).item())
            return step_x, step_y
        
        # Random walk if nothing interesting visible
        idx = torch.randint(0, len(RANDOM_WALK_MOVES), (1,), device=device).item()
        return RANDOM_WALK_MOVES[idx]
