#!/usr/bin/env python3
"""
Debug script to visualize what trained CNNs are "thinking".

Run: python debug_models.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from configs.default_config import get_config
from environments.world import World
from agents.organism import Organism
from behaviors import LearnedBehavior, LearnedGroundTruth, LearnedProxy, OmniscientSeeker
from behaviors.brains.tiny_cnn import TinyCNN
from behaviors.action_space import build_action_space, index_to_action, ACTION_LABELS_8


def load_model(path: str, input_shape=(4, 11, 11)) -> TinyCNN:
    """Load a trained model."""
    channels, h, w = input_shape
    model = TinyCNN(input_shape=(h, w), input_channels=channels, output_size=8)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def visualize_model_behavior(model_path: str, model_name: str):
    """Show what the model sees and decides."""
    config = get_config()
    world = World(30, 30, config)
    
    # Place some food and poison
    np.random.seed(42)
    for _ in range(5):
        world.place_food(1)
        world.place_poison(1)
    
    # Create a test agent at center
    behavior = OmniscientSeeker()  # Just for getting views
    agent = Organism(15, 15, 50.0, 5, world, behavior, config)
    
    # Get the observation
    view = agent.get_local_view()  # (4, 11, 11)
    print(f"\n{model_name}")
    print(f"  View shape: {view.shape}")
    print(f"  Channel sums: {[f'{view[i].sum():.1f}' for i in range(4)]}")
    
    # Load model and get action
    model = load_model(model_path)
    
    with torch.no_grad():
        input_tensor = torch.from_numpy(view).float().unsqueeze(0)  # (1, 4, 11, 11)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        action_idx = logits.argmax(dim=1).item()
    
    # Use centralized action mapping
    action = index_to_action(action_idx, max_move_distance=1)
    
    print(f"  Action probabilities: {[f'{p:.2f}' for p in probs[0].tolist()]}")
    print(f"  Chosen action: {action_idx} -> move ({action[0]}, {action[1]})")
    
    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    titles = ['Empty', 'Wall', 'Food', 'Poison']
    for i, (ax, title) in enumerate(zip(axes[:4], titles)):
        ax.imshow(view[i], cmap='viridis')
        ax.set_title(f'{title}\nsum={view[i].sum():.1f}')
        ax.axis('off')
    
    # Show probabilities as bar chart
    axes[4].bar(range(8), probs[0].tolist())
    axes[4].set_title(f'Action probs\nChosen: {action_idx}')
    axes[4].set_xticks(range(8))
    axes[4].set_xticklabels(['↑', '↓', '←', '→', '↖', '↗', '↙', '↘'], fontsize=8)
    
    fig.suptitle(f'{model_name}: Input Channels and Action Probabilities')
    plt.tight_layout()
    return fig


def analyze_training_data():
    """Check what actions were recorded during training."""
    from training.collect import collect_from_expert
    
    config = get_config()
    config['GRID_WIDTH'] = 30
    config['GRID_HEIGHT'] = 30
    
    print("\n" + "="*50)
    print("Analyzing what actions experts actually take...")
    print("="*50)
    
    # Collect a small sample from OmniscientSeeker
    from behaviors import OmniscientSeeker
    buffer = collect_from_expert(config, OmniscientSeeker, num_steps=100, num_agents=3)
    
    # Count actions
    action_counts = {}
    for exp in buffer.buffer:
        a = exp.action
        action_counts[a] = action_counts.get(a, 0) + 1
    
    print(f"\nOmniscientSeeker action distribution (n={len(buffer)}):")
    actions = ['↑(0,-1)', '↓(0,1)', '←(-1,0)', '→(1,0)', '↖', '↗', '↙', '↘']
    for a, label in enumerate(actions):
        count = action_counts.get(a, 0)
        bar = '█' * (count // 5)
        print(f"  {a} {label}: {count:3d} {bar}")


if __name__ == "__main__":
    print("="*50)
    print("CNN DEBUG VISUALIZATION")
    print("="*50)
    
    # First analyze training data
    analyze_training_data()
    
    # Visualize models
    try:
        fig1 = visualize_model_behavior('models/ground_truth.pth', 'Ground Truth Model')
        fig2 = visualize_model_behavior('models/proxy_trained.pth', 'Proxy Model')
        plt.show()
    except FileNotFoundError as e:
        print(f"\nError loading model: {e}")
        print("Run training first: python training/train.py --mode both")
