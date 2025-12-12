import torch
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
    """
    A custom PyTorch Dataset for storing and serving simulation data.
    
    In PyTorch, a Dataset class handles fetching individual samples.
    A DataLoader class (which we'll use in the training loop) handles batching,
    shuffling, and parallel loading.
    """
    def __init__(self):
        # TODO: Define how you store data
        # For Reinforcement Learning (RL), this might be a "Replay Buffer".
        # It needs to store transitions: (state, action, reward, next_state, done)
        self.data = []

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample at the given index.
        
        This method must return tensors that the model can understand.
        """
        # TODO: Implement retrieval
        # sample = self.data[idx]
        
        # Example conversion:
        # state_tensor = torch.from_numpy(sample['state']).float()
        # action_tensor = torch.tensor(sample['action']).long()
        
        # return state_tensor, action_tensor
        pass

    def add(self, experience):
        """Custom method to add new data from the simulation."""
        self.data.append(experience)
