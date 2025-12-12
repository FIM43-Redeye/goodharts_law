import torch
import numpy as np
from behaviors import BehaviorStrategy
# from behaviors.brains.tiny_cnn import TinyCNN # Uncomment when implemented

class LearnedBehavior(BehaviorStrategy):
    """
    A behavior strategy that uses a neural network (TinyCNN) to decide actions.
    """
    def __init__(self, model_path=None):
        # TODO: Initialize your model here
        # self.brain = TinyCNN()
        
        # If a path is provided, load the weights
        if model_path:
            # self.brain.load_state_dict(torch.load(model_path))
            pass
            
        # Put model in evaluation mode (important for inference!)
        # self.brain.eval() 
        pass

    @property
    def requirements(self) -> list[str]:
        # We likely want 'ground_truth' or 'proxy_signal' or both depending on what we train on
        return ['ground_truth']

    def decide_action(self, agent, view: np.ndarray):
        """
        Decides the next action using the neural network.
        
        Args:
            agent: The organism instance.
            view (np.ndarray): The 2D view of the environment.
        
        Returns:
            (int, int): dx, dy movement vector.
        """
        
        # ---------------------------------------------------------------------
        # EDUCATIONAL NOTE: From Numpy to PyTorch
        #
        # 1. Models expect a "Batch" dimension. Even for a single agent, 
        #    we need input shape meant for many: (1, height, width).
        #    Use `unsqueeze(0)` to add this dimension.
        #
        # 2. PyTorch models expect (Batch, Channels, Height, Width).
        #    If your view is just 2D (H, W), add a channel dim: (1, 1, H, W).
        #
        # 3. Data type matters! Neural nets usually work with Float32.
        # ---------------------------------------------------------------------
        
        # TODO: Convert the numpy view to a torch tensor
        # tensor_view = torch.from_numpy(view).float()
        
        # TODO: Add batch and channel dimensions
        # input_tensor = tensor_view.unsqueeze(0).unsqueeze(0) 
        
        # TODO: Pass through the brain
        # with torch.no_grad(): # Use no_grad for inference to save memory!
        #    logits = self.brain(input_tensor)
        
        # TODO: Decipher the output
        # Usually, the output is a set of scores for each possible action.
        # We pick the one with the highest score (argmax).
        # action_idx = torch.argmax(logits, dim=1).item()
        
        # Map index to dx, dy (Example mapping)
        # moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Down, Up, Right, Left
        # return moves[action_idx]
        
        return (0, 0) # Placeholder: don't move
