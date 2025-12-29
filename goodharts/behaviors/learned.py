"""
Learned behavior strategies using neural networks.

Supports training on either ground-truth or proxy signals,
enabling empirical demonstration of Goodhart's Law.
"""
import torch
from goodharts.behaviors import BehaviorStrategy
from goodharts.behaviors.brains.base_cnn import BaseCNN
from goodharts.behaviors.brains import load_brain
from goodharts.behaviors.action_space import (
    ActionSpace,
    DiscreteGridActionSpace,
    create_action_space,
    load_action_space,
)
from goodharts.utils.device import get_device
from goodharts.utils.logging_config import get_logger

logger = get_logger("learned_behavior")


class LearnedBehavior(BehaviorStrategy):
    """
    A behavior strategy that uses a neural network (BaseCNN) to decide actions.
    
    Supports two modes:
    - 'ground_truth': Agent sees real cell types (like OmniscientSeeker)
    - 'proxy': Agent sees only the proxy/interestingness signal (like ProxySeeker)
    
    The action space is defined in behaviors.action_space (single source of truth).
    """
    
    def __init__(
        self,
        mode: str = 'ground_truth',
        model_path: str | None = None,
        epsilon: float = 0.0,
        max_move_distance: int = 1,
        temperature: float = 1.0,
        name: str = "LearnedBehavior",
        action_space: ActionSpace | None = None,
    ):
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
            action_space: ActionSpace instance (optional, defaults to DiscreteGridActionSpace)
        """
        if mode not in ('ground_truth', 'ground_truth_handhold', 'proxy', 'ground_truth_blinded'):
            raise ValueError(f"mode must be 'ground_truth', 'ground_truth_handhold', 'proxy', or 'ground_truth_blinded', got '{mode}'")

        self.name = name
        self._mode = mode
        self.model_path = model_path
        self.epsilon = epsilon
        self.max_move_distance = max_move_distance
        self.temperature = temperature

        self.brain: BaseCNN | None = None
        self.device = get_device(verbose=False)

        # Action space (pluggable - defaults to discrete grid)
        if action_space is not None:
            self.action_space = action_space
        else:
            self.action_space = create_action_space('discrete_grid', max_move_distance)

    @property
    def requirements(self) -> list[str]:
        """Declare what information this behavior needs from the world."""
        if self._mode in ('ground_truth', 'ground_truth_handhold'):
            return ['ground_truth']
        else:
            return ['proxy_metric']

    @property
    def num_actions(self) -> int:
        """Number of possible actions in the action space."""
        return self.action_space.n_outputs

    def _init_brain(self, input_shape: tuple[int, int, int]):
        """
        Lazy initialization of the brain based on actual view shape.

        Uses load_brain() which handles both new checkpoint format (with metadata)
        and legacy format (raw state_dict).

        Args:
            input_shape: (num_channels, height, width) from observation
        """
        num_channels, height, width = input_shape

        if self.model_path:
            try:
                # Use load_brain() which handles both new and legacy formats
                self.brain, metadata = load_brain(self.model_path, device=self.device)

                steps = metadata.get('training_steps', metadata.get('total_steps', '?'))
                mode = metadata.get('mode', '?')
                logger.info(f"Loaded model from {self.model_path} (mode={mode}, steps={steps})")

                # Update action space from checkpoint if available
                # This ensures we use the same action space the model was trained with
                if 'action_space' in metadata:
                    self.action_space = load_action_space(metadata['action_space'])
                    logger.debug(f"Loaded action space from checkpoint: {self.action_space}")

            except FileNotFoundError:
                logger.warning(f"Model not found at {self.model_path}, using random weights")
                self.brain = None
            except Exception as e:
                logger.warning(f"Error loading model: {e}, using random weights")
                self.brain = None

        # Fallback: create fresh brain if loading failed or no path provided
        if self.brain is None:
            self.brain = BaseCNN(
                input_shape=(height, width),
                input_channels=num_channels,
                output_size=self.num_actions
            )
            self.brain.to(self.device)

        # Set to inference mode (disables dropout, batchnorm training, etc.)
        self.brain.train(False)

    def get_action_logits(self, view: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits from the neural network.
        
        Useful for training and saliency visualization.
        
        Args:
            view: Tensor (channels, H, W) on device
            
        Returns:
            Tensor of shape (1, num_actions) with raw scores
        """
        if self.brain is None:
            self._init_brain(view.shape)
        
        # Ensure view is tensor
        if not isinstance(view, torch.Tensor):
            # Fallback (should prevent this but just in case)
            tensor_view = torch.from_numpy(view).float().to(self.device)
        else:
            tensor_view = view.float()
            
        if tensor_view.device != self.device:
            tensor_view = tensor_view.to(self.device)
        
        # Handle both 2D (H, W) and 3D (C, H, W) inputs
        if tensor_view.dim() == 2:
            # Old format: add both batch and channel dims -> (1, 1, H, W)
            input_tensor = tensor_view.unsqueeze(0).unsqueeze(0)
        else:
            # New format: already (C, H, W), just add batch dim -> (1, C, H, W)
            input_tensor = tensor_view.unsqueeze(0)
        
        return self.brain(input_tensor)

    def decide_action(self, agent, view: torch.Tensor) -> tuple[int, int]:
        """
        Decides the next action using the neural network.

        Uses temperature-based sampling from softmax probabilities,
        which provides natural exploration when uncertain.

        Args:
            agent: The organism instance (unused but required by interface)
            view: The local view of the environment (Tensor)

        Returns:
            (dx, dy) movement vector
        """
        # Epsilon-greedy exploration (for training)
        if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
            # Random action - sample uniformly from action space
            idx = torch.randint(0, self.action_space.n_outputs, (1,)).item()
            if isinstance(self.action_space, DiscreteGridActionSpace):
                return self.action_space.index_to_action(idx)
            # For other action spaces, generate random logits
            random_logits = torch.randn(1, self.action_space.n_outputs, device=self.device)
            return self.action_space.decode(random_logits, sample=True, temperature=1.0)

        # Neural network inference
        with torch.no_grad():
            logits = self.get_action_logits(view)
            # Delegate decoding to action space
            return self.action_space.decode(logits, sample=True, temperature=self.temperature)

    def set_training_mode(self, training: bool):
        """Switch between training and evaluation mode."""
        if self.brain is not None:
            if training:
                self.brain.train()
            else:
                self.brain.eval()

    def get_brain(self) -> BaseCNN | None:
        """Access the underlying neural network for training."""
        return self.brain
    
    def action_to_index(self, dx: int, dy: int) -> int:
        """Convert a (dx, dy) action to its index in the action space.

        Only valid for discrete action spaces.
        """
        if isinstance(self.action_space, DiscreteGridActionSpace):
            return self.action_space.action_to_index(dx, dy)
        raise TypeError(f"action_to_index not supported for {type(self.action_space).__name__}")

    def index_to_action(self, idx: int) -> tuple[int, int]:
        """Convert an action index to (dx, dy).

        Only valid for discrete action spaces.
        """
        if isinstance(self.action_space, DiscreteGridActionSpace):
            return self.action_space.index_to_action(idx)
        raise TypeError(f"index_to_action not supported for {type(self.action_space).__name__}")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.name} mode={self._mode}>"


# =============================================================================
# PRESET FACTORY (preferred way to create learned behaviors)
# =============================================================================

# Preset configurations for common learned behavior types
LEARNED_PRESETS: dict[str, dict] = {
    'ground_truth': {
        'mode': 'ground_truth',
        'model_path': 'models/ppo_ground_truth.pth',
        'color': (0, 200, 255),  # Light cyan
    },
    'ground_truth_handhold': {
        'mode': 'ground_truth_handhold',
        'model_path': 'models/ppo_ground_truth_handhold.pth',
        'color': (100, 255, 100),  # Light green
    },
    'proxy': {
        'mode': 'proxy',
        'model_path': 'models/ppo_proxy.pth',
        'color': (255, 100, 255),  # Light magenta
    },
    'ground_truth_blinded': {
        'mode': 'ground_truth_blinded',
        'model_path': 'models/ppo_ground_truth_blinded.pth',
        'color': (138, 43, 226),  # Blue-violet
    },
}


def create_learned_behavior(preset: str = 'ground_truth', **kwargs) -> LearnedBehavior:
    """
    Create a learned behavior from a preset.
    
    This is the preferred way to create learned behaviors. Presets define
    sensible defaults that can be overridden via kwargs.
    
    Args:
        preset: One of 'ground_truth', 'proxy', 'ground_truth_blinded'
        **kwargs: Override preset values (model_path, epsilon, temperature, etc.)
        
    Returns:
        Configured LearnedBehavior instance
        
    Example:
        behavior = create_learned_behavior('ground_truth', model_path='models/my_model.pth')
    """
    if preset not in LEARNED_PRESETS:
        available = ', '.join(sorted(LEARNED_PRESETS.keys()))
        raise ValueError(f"Unknown preset: '{preset}'. Available: [{available}]")
    
    config = LEARNED_PRESETS[preset].copy()
    color = config.pop('color', None)
    config.update(kwargs)
    
    # Pass preset name as name
    behavior = LearnedBehavior(name=preset, **config)
    if color:
        behavior._color = color
    return behavior
