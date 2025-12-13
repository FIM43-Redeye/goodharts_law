"""
Learned behavior strategies using neural networks.

Supports training on either ground-truth or proxy signals,
enabling empirical demonstration of Goodhart's Law.
"""
import torch
import numpy as np
from behaviors import BehaviorStrategy
from behaviors.brains.tiny_cnn import TinyCNN
from behaviors.action_space import build_action_space, action_to_index, index_to_action, num_actions


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
                self.brain.load_state_dict(
                    torch.load(self.model_path, map_location=self.device, weights_only=True)
                )
                print(f"[LearnedBehavior] Loaded model from {self.model_path}")
            except FileNotFoundError:
                print(f"[LearnedBehavior] Model not found at {self.model_path}, using random weights")
        
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


# =============================================================================
# RL STUBS - To be implemented for full reinforcement learning support
# =============================================================================

class RLTrainer:
    """
    Stub for reinforcement learning trainer.
    
    TODO: Implement policy gradient (REINFORCE) or Q-learning.
    
    Current plan: Start with reward-weighted behavior cloning,
    then upgrade to this for true RL.
    """
    
    def __init__(self, behavior: LearnedBehavior, lr: float = 1e-3):
        self.behavior = behavior
        self.lr = lr
        self.optimizer = None  # Will be torch.optim.Adam
        
    def compute_policy_gradient_loss(self, states, actions, rewards):
        """
        REINFORCE-style policy gradient.
        
        loss = -sum(log_prob(action) * reward)
        
        TODO: Implement with proper baseline subtraction
        """
        raise NotImplementedError("Policy gradient not yet implemented")
    
    def compute_q_learning_loss(self, states, actions, rewards, next_states, dones):
        """
        DQN-style Q-learning.
        
        TODO: Implement with target network and experience replay
        """
        raise NotImplementedError("Q-learning not yet implemented")
    
    def update(self, batch):
        """Single training step."""
        raise NotImplementedError("Training step not yet implemented")
