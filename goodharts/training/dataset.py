"""
Replay buffer and dataset utilities for training learned behaviors.

Supports both supervised learning (behavior cloning) and 
reinforcement learning (experience replay) workflows.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import NamedTuple
from collections import deque
from goodharts.utils.device import get_device


class Experience(NamedTuple):
    """A single experience tuple from the simulation."""
    state: torch.Tensor | np.ndarray    # The agent's view
    action: int                         # Action index taken
    reward: float                       # Energy delta or shaped reward
    next_state: torch.Tensor | np.ndarray | None # View after taking action
    done: bool                          # Whether episode ended (agent died)


class ReplayBuffer:
    """
    Experience replay buffer for reinforcement learning.
    
    Stores experiences and supports random sampling for training.
    Uses a deque for efficient FIFO eviction when capacity is reached.
    """
    
    def __init__(self, capacity: int = 100_000):
        """
        Args:
            capacity: Maximum number of experiences to store.
                     Oldest experiences are evicted when full.
        """
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state: torch.Tensor | np.ndarray, action: int, reward: float, 
            next_state: torch.Tensor | np.ndarray | None = None, done: bool = False):
        """Add a single experience to the buffer."""
        # Convert to tensor if numpy, or clone if tensor
        # We store on CPU to save VRAM usually, but user wants full Torch.
        # We'll store as is. If incoming is GPU tensor, it stays on GPU.
        # If this causes OOM, we can add .cpu() here.
        
        def process_state(s):
            if s is None: return None
            if isinstance(s, torch.Tensor):
                return s.clone()
            return torch.from_numpy(s).float() # Convert to tensor immediately
        
        exp = Experience(
            state=process_state(state),
            action=action,
            reward=reward,
            next_state=process_state(next_state),
            done=done
        )
        self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of Experience tuples
        """
        # Using numpy for random choice of indices is fine
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_tensors(self, batch_size: int, device: torch.device = None) -> dict[str, torch.Tensor]:
        """
        Sample a batch and convert to tensors ready for training.
        
        Returns:
            dict with keys: 'states', 'actions', 'rewards', 'dones'
            (next_states omitted if not needed for standard PPO, added if needed)
        """
        if device is None:
            device = get_device(verbose=False)
            
        batch = self.sample(batch_size)
        
        # Helper to stack
        def stack_states(states_list):
            if isinstance(states_list[0], torch.Tensor):
                return torch.stack(states_list).to(device)
            else:
                return torch.from_numpy(np.stack(states_list)).float().to(device)

        states = stack_states([e.state for e in batch])
        
        # Ensure correct shape (B, C, H, W)
        if states.dim() == 3: # (B, H, W)
             states = states.unsqueeze(1)
        
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float, device=device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool, device=device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Empty the buffer."""
        self.buffer.clear()


class SimulationDataset(Dataset):
    """
    PyTorch Dataset wrapper around experiences.
    
    Useful for supervised learning (behavior cloning) where we want
    to iterate through all data in epochs rather than random sampling.
    """
    
    def __init__(self, experiences: list[Experience] | None = None):
        """
        Args:
            experiences: Optional initial list of experiences
        """
        self.experiences: list[Experience] = experiences or []
    
    def __len__(self) -> int:
        return len(self.experiences)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            (state_tensor, action_tensor, reward_tensor)
        """
        exp = self.experiences[idx]
        
        if isinstance(exp.state, torch.Tensor):
            state_tensor = exp.state.float()
        else:
            state_tensor = torch.from_numpy(exp.state).float()
        
        # Handle both 2D (H, W) and 3D (C, H, W) states
        if state_tensor.dim() == 2:
            # Old format: add channel dimension (H, W) -> (1, H, W)
            state_tensor = state_tensor.unsqueeze(0)
        # else: already (C, H, W), no change needed
        
        action_tensor = torch.tensor(exp.action, dtype=torch.long)
        reward_tensor = torch.tensor(exp.reward, dtype=torch.float)
        
        return state_tensor, action_tensor, reward_tensor
    
    def add(self, experience: Experience):
        """Add a single experience."""
        self.experiences.append(experience)
    
    def add_batch(self, experiences: list[Experience]):
        """Add multiple experiences."""
        self.experiences.extend(experiences)
    
    def from_replay_buffer(self, buffer: ReplayBuffer):
        """Load all experiences from a replay buffer."""
        self.experiences = list(buffer.buffer)
    
    def filter_by_reward(self, min_reward: float = 0.0) -> 'SimulationDataset':
        """
        Create a new dataset with only positive-reward experiences.
        
        Useful for behavior cloning where we only want to imitate good actions.
        """
        filtered = [e for e in self.experiences if e.reward >= min_reward]
        return SimulationDataset(filtered)
    
    def compute_reward_weights(self) -> np.ndarray:
        """
        Compute per-sample weights based on rewards.
        
        For reward-weighted behavior cloning:
        - Positive rewards get higher weight
        - Negative rewards get lower weight (or zero)
        
        Returns normalized weights that sum to len(dataset).
        """
        # Convert to numpy for calculation
        rewards = np.array([e.reward for e in self.experiences])
        
        # Shift rewards to be non-negative
        min_r = rewards.min()
        shifted = rewards - min_r + 1e-6  # Small epsilon for zero rewards
        
        # Normalize
        weights = shifted / shifted.sum() * len(self.experiences)
        
        return weights
    
    def compute_visibility_weights(
        self, 
        food_channel: int = 2, 
        poison_channel: int = 3,
        visible_multiplier: float = 10.0,
        base_weight: float = 1.0,
    ) -> np.ndarray:
        """
        Compute per-sample weights based on whether targets are visible.
        
        Samples where food or poison is visible get higher weight.
        """
        weights = []
        
        for exp in self.experiences:
            state = exp.state
            
            # Helper to check visibility on tensor or numpy
            is_tensor = isinstance(state, torch.Tensor)
            shape = state.shape
            
            if len(shape) == 3:
                # Channels are (C, H, W)
                if is_tensor:
                    food_visible = state[food_channel].sum().item() > 0
                    poison_visible = state[poison_channel].sum().item() > 0 if poison_channel < shape[0] else False
                else:
                    food_visible = state[food_channel].sum() > 0
                    poison_visible = state[poison_channel].sum() > 0 if poison_channel < shape[0] else False
            else:
                # Old 2D format
                food_visible = False
                poison_visible = False
            
            if food_visible or poison_visible:
                weights.append(visible_multiplier)
            else:
                weights.append(base_weight)
        
        weights = np.array(weights)
        
        # Normalize so weights sum to len(dataset)
        weights = weights / weights.sum() * len(self.experiences)
        
        return weights
    
    def compute_combined_weights(self, visibility_mult: float = 5.0) -> np.ndarray:
        """
        Combine reward and visibility weighting.
        """
        reward_weights = self.compute_reward_weights()
        vis_weights = self.compute_visibility_weights(visible_multiplier=visibility_mult)
        
        combined = reward_weights * vis_weights
        combined = combined / combined.sum() * len(self.experiences)
        
        return combined
