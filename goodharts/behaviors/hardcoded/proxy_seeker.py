"""
Proxy-optimizing behavior that only sees interestingness signals.

In proxy mode, both channels contain the same interestingness values,
so this agent cannot distinguish food from poison based on cell type.
With poison being MORE interesting (1.0) than food (0.5), this agent
will actively PREFER poison. This is the Goodhart's Law trap.

Goodhart's Law in Action:
    "When a measure becomes a target, it ceases to be a good measure."

    THE PROXY METRIC: "Interestingness" is a measurable signal that correlates
    with valuable resources - food is interesting (0.5), poison is interesting
    (1.0). But interestingness is NOT the true objective (survival/energy).
    It's a proxy: easy to measure, but fundamentally disconnected from what
    actually matters.

    INFORMATION BLINDNESS: The proxy observation encoding places the same
    interestingness value in both channels. Food appears as (0.5, 0.5), poison
    as (1.0, 1.0). The agent perceives different magnitudes but has no channel
    that distinguishes cell TYPES. The encoding itself destroys the information
    needed to survive.

    WHY OPTIMIZATION FAILS: This agent greedily maximizes interestingness.
    Since poison (1.0) > food (0.5), the agent systematically PREFERS poison.
    It achieves perfect performance on the proxy metric while dying. This is
    the core Goodhart failure: optimizing the measure, not the goal.

    REAL-WORLD PARALLELS:
    - Recommendation algorithms maximizing engagement (clicks, watch time)
      instead of user wellbeing, leading to addictive or harmful content
    - Companies optimizing metrics that look good on reports but don't
      reflect actual value creation
    - ML models learning spurious correlations that ace benchmarks but
      fail catastrophically in deployment

    WHY ANTI-CORRELATION SHARPENS THE DEMONSTRATION: If poison had equal
    interestingness to food, failures would be random (50% chance). By making
    poison MORE interesting, the failure becomes systematic and dramatic -
    the better the agent optimizes, the faster it dies. This makes the
    Goodhart trap unmistakable rather than attributable to noise.
"""
import torch
from goodharts.behaviors.base import BehaviorStrategy
from goodharts.behaviors.utils import RANDOM_WALK_MOVES, sign_scalar, create_circular_mask


class ProxySeeker(BehaviorStrategy):
    """
    Behavior that optimizes for proxy signal (interestingness).

    Uses 2-channel encoding where both channels contain identical
    interestingness values. Cannot distinguish food from poison.
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
            view: Shape (2, H, W) tensor - both channels have same interestingness

        Returns:
            (dx, dy) movement vector
        """
        device = view.device
        center = agent.sight_radius

        # In proxy mode, both channels have identical interestingness values
        # Just use channel 0 (they're the same)
        proxy_signal = view[0]  # (H, W)
        
        # We need to find the specific (dx, dy) with the max value.
        # Mask out the center (agent itself) to avoid staying put if current tile is high?
        # Original logic: "if dx==0 and dy==0: continue" -> Yes.
        
        input_view = proxy_signal.clone()
        input_view[center, center] = -1.0  # Mask center

        # Mask out area outside sight radius (circle)
        H, W = input_view.shape
        _, _, visible_mask, _ = create_circular_mask(
            H, center, agent.sight_radius, device
        )
        input_view[~visible_mask] = -1.0
        
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

            step_x = sign_scalar(dx)
            step_y = sign_scalar(dy)
            return step_x, step_y
        
        # Random walk if nothing interesting visible
        idx = torch.randint(0, len(RANDOM_WALK_MOVES), (1,), device=device).item()
        return RANDOM_WALK_MOVES[idx]
