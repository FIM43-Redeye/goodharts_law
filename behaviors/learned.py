import torch
import numpy as np
from behaviors import BehaviorStrategy
from behaviors.brains.tiny_cnn import TinyCNN

class LearnedBehavior(BehaviorStrategy):
    """
    A behavior strategy that uses a neural network (TinyCNN) to decide actions.
    """
    def __init__(self, model_path=None):
        self.brain = None
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def requirements(self) -> list[str]:
        # We likely want 'ground_truth' or 'proxy_signal' or both depending on what we train on
        return ['ground_truth']

    def _init_brain(self, input_shape):
        """Lazy initialization of the brain based on actual view shape."""
        # 4 actions: Up, Down, Left, Right
        self.brain = TinyCNN(input_shape=input_shape, input_channels=1, output_size=4)
        
        if self.model_path:
            try:
                self.brain.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"Loaded model from {self.model_path}")
            except FileNotFoundError:
                print(f"Model path {self.model_path} not found, starting fresh.")
        
        self.brain.to(self.device)
        self.brain.eval() # optimized for inference

    def decide_action(self, agent, view: np.ndarray):
        """
        Decides the next action using the neural network.
        
        Args:
            agent: The organism instance.
            view (np.ndarray): The 2D view of the environment.
        
        Returns:
            (int, int): dx, dy movement vector.
        """
        if self.brain is None:
            self._init_brain(view.shape)
            
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
        
        # Convert numpy to tensor
        tensor_view = torch.from_numpy(view).float().to(self.device)
        
        # Add batch (0) and channel (1) dimensions -> (1, 1, H, W)
        input_tensor = tensor_view.unsqueeze(0).unsqueeze(0)
        
        # Inference
        with torch.no_grad(): # Disable gradient calculation for speed/memory
            logits = self.brain(input_tensor)
            
        # Select action with highest score
        action_idx = torch.argmax(logits, dim=1).item()
        
        # Map index to dx, dy
        # 0: Up (0, -1)
        # 1: Down (0, 1)
        # 2: Left (-1, 0)
        # 3: Right (1, 0)
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        if 0 <= action_idx < len(moves):
            return moves[action_idx]
        
        return (0, 0) # Fallback

