"""
Proxy-optimizing behavior that only sees interestingness signals.

In proxy mode, both channels contain the same interestingness values,
so this agent cannot distinguish food from poison based on cell type.
Even though food is MORE interesting (1.0) than poison (0.5), this agent
still consumes poison because the metric doesn't encode harm.

Goodhart's Law in Action:
    "When a measure becomes a target, it ceases to be a good measure."

    THE PROXY METRIC: "Interestingness" is a measurable signal that correlates
    with resources - food is interesting (1.0), poison is also interesting
    (0.5). But interestingness is NOT the true objective (survival/energy).
    It's a proxy: easy to measure, but incomplete - it doesn't encode harm.

    INFORMATION BLINDNESS: The proxy observation encoding places the same
    interestingness value in both channels. Food appears as (1.0, 1.0), poison
    as (0.5, 0.5). The agent perceives different magnitudes but has no channel
    that distinguishes HARMFUL from SAFE. The encoding itself destroys the
    information needed to survive.

    WHY OPTIMIZATION FAILS: This agent greedily maximizes interestingness.
    Food is more interesting, so the agent prefers food when choosing. But
    poison is ALSO interesting (0.5 > 0), so the agent consumes it too.
    The metric doesn't signal "this will kill you" - it just says "interesting!"
    This is the core Goodhart failure: the proxy is incomplete, not adversarial.

    REAL-WORLD PARALLELS:
    - Recommendation algorithms maximizing engagement (clicks, watch time)
      without encoding "is this harmful?" - engagement doesn't mean wellbeing
    - Companies optimizing metrics that look good on reports but don't
      capture negative externalities
    - ML models learning correlations that ace benchmarks but miss crucial
      safety-relevant features

    WHY INCOMPLETE > ANTI-CORRELATED: Real proxy metrics aren't designed to
    cause harm - they just fail to capture everything that matters. Food being
    MORE interesting than poison makes this demonstration more honest: the
    metric isn't rigged against the agent, it's just missing a dimension.
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
