"""
Omniscient behavior that sees ground truth cell types.

Uses one-hot encoded channels derived from CellType for full observability.
This agent can distinguish food from poison perfectly.
"""
import torch
from goodharts.behaviors.base import BehaviorStrategy
from goodharts.behaviors.utils import RANDOM_WALK_MOVES, sign_scalar, create_circular_mask
from goodharts.configs.default_config import CellType


class OmniscientSeeker(BehaviorStrategy):
    """
    Behavior that can see exactly what each cell contains.
    
    Uses one-hot encoded view with separate channels for each cell type.
    This agent can distinguish food from poison perfectly.
    """
    
    # Distinct color for visualization (bright green)
    _color = (0, 255, 128)
    
    @property
    def requirements(self) -> list[str]:
        return ['ground_truth']

    def decide_action(self, agent, view: torch.Tensor) -> tuple[int, int]:
        """
        Decide action based on full ground-truth observation.
        
        Args:
            agent: The organism instance
            view: Shape (num_channels, H, W) tensor copy (will not be modified in place ideally, but we can clone)
            
        Returns:
            (dx, dy) movement vector
        """
        device = view.device
        center = agent.sight_radius
        
        # Extract relevant channels using CellType for indices
        food_channel = view[CellType.FOOD.channel_index]      # (H, W)
        poison_channel = view[CellType.POISON.channel_index]  # (H, W)
        wall_channel = view[CellType.WALL.channel_index]      # (H, W)
        
        # Find all visible food positions using circular mask
        H, W = food_channel.shape
        y_grid, x_grid, visible_mask, dist_sq = create_circular_mask(
            H, center, agent.sight_radius, device
        )
        
        # Identify food locations > 0.5
        food_mask = (food_channel > 0.5) & visible_mask
        
        food_indices = food_mask.nonzero(as_tuple=False) # (N_food, 2) -> [y, x]
        
        if food_indices.shape[0] > 0:
            # Calculate actual distances for these points
            food_y = food_indices[:, 0]
            food_x = food_indices[:, 1]
            
            # Distance from center
            dists = dist_sq[food_y, food_x]
            
            # Find min distance
            min_idx = torch.argmin(dists)
            
            target_y = food_y[min_idx].item()
            target_x = food_x[min_idx].item()
            
            dx = target_x - center
            dy = target_y - center
            
            step_x = sign_scalar(dx)
            step_y = sign_scalar(dy)
            
            # Check if immediate step would hit poison or wall
            # indices for check:
            check_y = center + step_y
            check_x = center + step_x
            
            is_poison = poison_channel[check_y, check_x] > 0.5
            is_wall = wall_channel[check_y, check_x] > 0.5
            
            if is_poison or is_wall:
                # Try cardinal directions only
                for alt_dx, alt_dy in [(step_x, 0), (0, step_y)]:
                    if alt_dx == 0 and alt_dy == 0:
                        continue
                    
                    chk_y = center + alt_dy
                    chk_x = center + alt_dx
                    
                    alt_poison = poison_channel[chk_y, chk_x] > 0.5
                    alt_wall = wall_channel[chk_y, chk_x] > 0.5
                    
                    if not alt_poison and not alt_wall:
                        return alt_dx, alt_dy
                return 0, 0  # Stuck
            
            return step_x, step_y
        
        # Random walk if no food visible, avoiding poison and walls
        # Filter safe moves efficiently?
        # Since purely random walk is small (8 moves), checking them is cheap.
        
        moves_tensor = torch.tensor(RANDOM_WALK_MOVES, device=device, dtype=torch.long)
        # moves is (8, 2) -> (dx, dy)
        
        # Calculate target indices for all moves
        tgt_x = center + moves_tensor[:, 0]
        tgt_y = center + moves_tensor[:, 1]
        
        # Check constraints
        # Ensure indices inside view (view includes padding so usually safe, but good to be careful)
        # Assuming view is large enough (2*r+1), and moves are 1 step.
        
        p_vals = poison_channel[tgt_y, tgt_x] > 0.5
        w_vals = wall_channel[tgt_y, tgt_x] > 0.5
        unsafe = p_vals | w_vals
        
        safe_indices = (~unsafe).nonzero(as_tuple=True)[0]
        
        if len(safe_indices) > 0:
            idx = safe_indices[torch.randint(0, len(safe_indices), (1,), device=device)].item()
            move = moves_tensor[idx]
            return move[0].item(), move[1].item()
        
        return 0, 0  # Completely stuck
