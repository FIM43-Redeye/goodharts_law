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


class Experience(NamedTuple):
    """A single experience tuple from the simulation."""
    state: np.ndarray      # The agent's view (2D array)
    action: int            # Action index taken
    reward: float          # Energy delta or shaped reward
    next_state: np.ndarray # View after taking action (optional, can be None)
    done: bool             # Whether episode ended (agent died)


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
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray | None = None, done: bool = False):
        """Add a single experience to the buffer."""
        exp = Experience(
            state=state.copy(),  # Copy to avoid mutation issues
            action=action,
            reward=reward,
            next_state=next_state.copy() if next_state is not None else None,
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
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_tensors(self, batch_size: int, device: torch.device = None) -> dict[str, torch.Tensor]:
        """
        Sample a batch and convert to tensors ready for training.
        
        Returns:
            dict with keys: 'states', 'actions', 'rewards', 'dones'
            (next_states omitted if not needed)
        """
        if device is None:
            from goodharts.utils.device import get_device
            device = get_device(verbose=False)
            
        batch = self.sample(batch_size)
        
        states = np.stack([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        dones = np.array([e.done for e in batch])
        
        return {
            'states': torch.from_numpy(states).float().unsqueeze(1).to(device),  # (B, 1, H, W)
            'actions': torch.from_numpy(actions).long().to(device),              # (B,)
            'rewards': torch.from_numpy(rewards).float().to(device),             # (B,)
            'dones': torch.from_numpy(dones).bool().to(device),                  # (B,)
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
        
        Samples where food or poison is visible get higher weight,
        but samples with nothing visible still get a base weight
        (so agents learn some random exploration behavior).
        
        Args:
            food_channel: Channel index for food in state (default=2)
            poison_channel: Channel index for poison in state (default=3)
            visible_multiplier: Weight multiplier for samples with visible targets
            base_weight: Weight for samples with no visible targets
            
        Returns:
            Weights array, normalized so sum = len(dataset)
        """
        weights = []
        
        for exp in self.experiences:
            state = exp.state
            
            # Check if state has channel dimension
            if state.ndim == 3:
                # Channels are (C, H, W)
                food_visible = state[food_channel].sum() > 0
                poison_visible = state[poison_channel].sum() > 0 if poison_channel < state.shape[0] else False
            else:
                # Old 2D format - can't determine visibility
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
        
        This gives high weight to samples that:
        1. Have positive reward (ate food)
        2. Have visible targets (can learn to navigate)
        
        Samples with neither still get small weight for exploration.
        """
        reward_weights = self.compute_reward_weights()
        vis_weights = self.compute_visibility_weights(visible_multiplier=visibility_mult)
        
        combined = reward_weights * vis_weights
        combined = combined / combined.sum() * len(self.experiences)
        
        return combined
