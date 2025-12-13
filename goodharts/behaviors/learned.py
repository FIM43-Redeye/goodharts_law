"""
Learned behavior strategies using neural networks.

Supports training on either ground-truth or proxy signals,
enabling empirical demonstration of Goodhart's Law.
"""
import torch
import numpy as np
from goodharts.behaviors import BehaviorStrategy
from goodharts.behaviors.brains.tiny_cnn import TinyCNN
from goodharts.behaviors.action_space import build_action_space, action_to_index, index_to_action, num_actions


class LearnedBehavior(BehaviorStrategy):
    """
    A behavior strategy that uses a neural network (TinyCNN) to decide actions.
    
    Supports two modes:
    - 'ground_truth': Agent sees real cell types (like OmniscientSeeker)
    - 'proxy': Agent sees only the proxy/interestingness signal (like ProxySeeker)
    
    The action space is defined in behaviors.action_space (single source of truth).
    """
    
    def __init__(self, mode: str = 'ground_truth', model_path: str | None = None, 
                 epsilon: float = 0.0, max_move_distance: int = 1,
                 temperature: float = 1.0):
        """
        Args:
            mode: 'ground_truth' or 'proxy' - determines what the agent can see
            model_path: Path to saved model weights (optional)
            epsilon: Exploration rate for epsilon-greedy (0.0 = pure exploitation)
            max_move_distance: Maximum movement distance per step (affects action space)
            temperature: Softmax temperature for action sampling.
                         Low (0.1) = nearly deterministic (like argmax)
                         1.0 = sample from softmax probs
                         High (2.0+) = more random
        """
        if mode not in ('ground_truth', 'proxy'):
            raise ValueError(f"mode must be 'ground_truth' or 'proxy', got '{mode}'")
        
        self._mode = mode
        self.model_path = model_path
        self.epsilon = epsilon
        self.max_move_distance = max_move_distance
        self.temperature = temperature
        
        self.brain: TinyCNN | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get action space from centralized module
        self._actions = build_action_space(max_move_distance)

    # Use centralized functions - keep these as class methods for compatibility
    @staticmethod
    def _build_action_space(max_dist: int) -> list[tuple[int, int]]:
        """Delegate to centralized action_space module."""
        return build_action_space(max_dist)

    @property
    def requirements(self) -> list[str]:
        """Declare what information this behavior needs from the world."""
        if self._mode == 'ground_truth':
            return ['ground_truth']
        else:
            return ['proxy_metric']

    @property
    def num_actions(self) -> int:
        """Number of possible actions in the action space."""
        return len(self._actions)

    def _init_brain(self, input_shape: tuple[int, int, int]):
        """
        Lazy initialization of the brain based on actual view shape.
        
        Supports loading both TinyCNN and ActorCritic (PPO) models.
        
        Args:
            input_shape: (num_channels, height, width) from observation
        """
        num_channels, height, width = input_shape
        self.brain = TinyCNN(
            input_shape=(height, width), 
            input_channels=num_channels, 
            output_size=self.num_actions
        )
        
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                
                # Detect if this is an ActorCritic model (has 'actor' keys)
                if any('actor' in k for k in state_dict.keys()):
                    # Map ActorCritic keys to TinyCNN keys
                    # ActorCritic: conv1, conv2, fc_shared (64), actor, critic
                    # TinyCNN: conv1, conv2, fc1 (64), fc_out (output)
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('conv1') or k.startswith('conv2'):
                            new_state_dict[k] = v
                        elif k.startswith('fc_shared'):
                            # fc_shared -> fc1
                            new_key = k.replace('fc_shared', 'fc1')
                            new_state_dict[new_key] = v
                        elif k.startswith('actor'):
                            # actor -> fc_out (output layer)
                            new_key = k.replace('actor', 'fc_out')
                            new_state_dict[new_key] = v
                        # Skip 'critic' - not needed for inference
                    
                    self.brain.load_state_dict(new_state_dict)
                    print(f"[LearnedBehavior] Loaded PPO ActorCritic model from {self.model_path}")
                else:
                    # Standard TinyCNN model
                    self.brain.load_state_dict(state_dict)
                    print(f"[LearnedBehavior] Loaded TinyCNN model from {self.model_path}")
                    
            except FileNotFoundError:
                print(f"[LearnedBehavior] Model not found at {self.model_path}, using random weights")
            except Exception as e:
                print(f"[LearnedBehavior] Error loading model: {e}, using random weights")
        
        self.brain.to(self.device)
        self.brain.eval()

    def get_action_logits(self, view: np.ndarray) -> torch.Tensor:
        """
        Get raw logits from the neural network.
        
        Useful for training and saliency visualization.
        
        Args:
            view: 3D numpy array (channels, H, W) or 2D (H, W) for backward compat
            
        Returns:
            Tensor of shape (1, num_actions) with raw scores
        """
        if self.brain is None:
            self._init_brain(view.shape)
        
        tensor_view = torch.from_numpy(view).float().to(self.device)
        
        # Handle both 2D (H, W) and 3D (C, H, W) inputs
        if tensor_view.dim() == 2:
            # Old format: add both batch and channel dims -> (1, 1, H, W)
            input_tensor = tensor_view.unsqueeze(0).unsqueeze(0)
        else:
            # New format: already (C, H, W), just add batch dim -> (1, C, H, W)
            input_tensor = tensor_view.unsqueeze(0)
        
        return self.brain(input_tensor)

    def decide_action(self, agent, view: np.ndarray) -> tuple[int, int]:
        """
        Decides the next action using the neural network.
        
        Uses temperature-based sampling from softmax probabilities,
        which provides natural exploration when uncertain.
        
        Args:
            agent: The organism instance (unused but required by interface)
            view: The local view of the environment
        
        Returns:
            (dx, dy) movement vector
        """
        # Epsilon-greedy exploration (for training)
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            idx = np.random.randint(0, self.num_actions)
            return self._actions[idx]
        
        # Neural network inference
        with torch.no_grad():
            logits = self.get_action_logits(view)
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)
            
            # Sample from the distribution (not argmax!)
            # This gives natural exploration when probabilities are uniform
            action_idx = torch.multinomial(probs, num_samples=1).item()
        
        if 0 <= action_idx < len(self._actions):
            return self._actions[action_idx]
        
        return (0, 1)  # Fallback: move down

    def set_training_mode(self, training: bool):
        """Switch between training and evaluation mode."""
        if self.brain is not None:
            if training:
                self.brain.train()
            else:
                self.brain.eval()

    def get_brain(self) -> TinyCNN | None:
        """Access the underlying neural network for training."""
        return self.brain
    
    def action_to_index(self, dx: int, dy: int) -> int:
        """Convert a (dx, dy) action to its index in the action space."""
        try:
            return self._actions.index((dx, dy))
        except ValueError:
            # If exact action not found, find closest
            min_dist = float('inf')
            closest_idx = 0
            for i, (ax, ay) in enumerate(self._actions):
                dist = abs(ax - dx) + abs(ay - dy)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            return closest_idx

    def index_to_action(self, idx: int) -> tuple[int, int]:
        """Convert an action index to (dx, dy)."""
        if 0 <= idx < len(self._actions):
            return self._actions[idx]
        return (0, 1)


# Convenience subclasses for cleaner config
class LearnedGroundTruth(LearnedBehavior):
    """Learned behavior that sees true cell types."""
    def __init__(self, model_path: str | None = None, epsilon: float = 0.0, 
                 max_move_distance: int = 1, temperature: float = 0.5):
        super().__init__(
            mode='ground_truth', 
            model_path=model_path, 
            epsilon=epsilon,
            max_move_distance=max_move_distance,
            temperature=temperature  # Lower = more deterministic when confident
        )


class LearnedProxy(LearnedBehavior):
    """Learned behavior that only sees proxy/interestingness signals."""
    def __init__(self, model_path: str | None = None, epsilon: float = 0.0,
                 max_move_distance: int = 1, temperature: float = 0.5):
        super().__init__(
            mode='proxy', 
            model_path=model_path, 
            epsilon=epsilon,
            max_move_distance=max_move_distance,
            temperature=temperature
        )


class LearnedProxyIllAdjusted(LearnedBehavior):
    """
    Learned behavior trained with ill-adjusted proxy rewards.
    
    During training, this agent was rewarded for touching interesting cells
    (regardless of whether they're food or poison). It sees the same proxy
    observations as LearnedProxy but learned a different policy due to
    the reward misalignment - a demonstration of Goodhart's Law.
    """
    def __init__(self, model_path: str | None = None, epsilon: float = 0.0,
                 max_move_distance: int = 1, temperature: float = 0.5):
        if model_path is None:
            model_path = 'models/ppo_proxy_ill_adjusted.pth'
        super().__init__(
            mode='proxy',  # Same observation as proxy
            model_path=model_path, 
            epsilon=epsilon,
            max_move_distance=max_move_distance,
            temperature=temperature
        )

