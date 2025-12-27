"""
Offline Headless Benchmark for Agents.

Running this script allows objective performance measurement of trained agents
without visual overhead. Uses the vectorized environment for high-throughput evaluation.
"""
import argparse
import torch
import time
from pathlib import Path
from torch.distributions import Categorical
import torch.nn.functional as F

from goodharts.configs.default_config import get_simulation_config
from goodharts.behaviors.brains import create_brain
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.utils.device import get_device

def benchmark(
    model_path: str,
    mode: str = 'ground_truth',
    n_episodes: int = 100,
    n_envs: int = 32,
    device_name: str = None,
):
    """
    Run benchmark for a trained model.
    """
    # Setup device
    if device_name:
        device = torch.device(device_name)
    else:
        device = get_device()
    
    print(f"\n[Benchmark] Agent: {Path(model_path).name}")
    print(f"   Mode: {mode}")
    print(f"   Device: {device}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Batch Size: {n_envs}")
    
    # Load config and spec
    config = get_simulation_config()
    try:
        spec = config['get_observation_spec'](mode)
    except Exception as e:
        print(f"[ERROR] Loading observation spec for mode '{mode}': {e}")
        return

    # Create Environment
    print(f"   Environment: {spec.view_size}x{spec.view_size} View, {spec.num_channels} Channels")
    vec_env = create_torch_vec_env(n_envs=n_envs, obs_spec=spec, config=config, device=device)
    

    # Define LegacyCNN for backward compatibility with older/smaller models
    class LegacyCNN(torch.nn.Module):
        def __init__(self, input_shape, output_size, input_channels):
            super().__init__()
            self.input_shape = input_shape
            self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
            flat_size = 32 * input_shape[0] * input_shape[1]
            self.fc1 = torch.nn.Linear(flat_size, 32)
            self.fc_out = torch.nn.Linear(32, output_size)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc_out(x)

    # Load Model
    print(f"   Loading model from: {model_path}")
    
    # Initialize main model (New Architecture)
    brain_type = 'base_cnn'  # Default architecture
    from goodharts.behaviors.action_space import num_actions
    n_actions = num_actions(1)
    
    # Create the modern brain
    input_channels = spec.num_channels
    policy = create_brain(brain_type, spec, output_size=n_actions).to(device)
    
    loaded = False
    try:
        # Try loading into new architecture
        state_dict = torch.load(model_path, map_location=device)
        policy.load_state_dict(state_dict)
        policy.eval()
        print("   [OK] Model loaded successfully (Current Architecture)")
        loaded = True
    except RuntimeError as e:
        # Check for size mismatch error which indicates legacy model
        if "size mismatch" in str(e):
            print("   [WARN] Architecture mismatch detected. Trying Legacy architecture...")
            try:
                # Instantiate Legacy Architecture
                policy = LegacyCNN(spec.input_shape, n_actions, input_channels).to(device)
                policy.load_state_dict(state_dict)
                policy.eval()
                print("   [OK] Legacy Model loaded successfully")
                loaded = True
            except Exception as e2:
                print(f"[ERROR] Failed to load as Legacy model: {e2}")
        else:
            print(f"[ERROR] Loading model: {e}")
            
    if not loaded:
        print("[ERROR] Could not load model.")
        return

    # Metrics
    episode_rewards = []
    episode_food = []
    episode_steps = []
    died_count = 0
    
    # Run Benchmark
    print("\nRunning benchmark", end="", flush=True)
    
    obs = vec_env.reset()
    completed_episodes = 0
    t_start = time.time()
    
    current_rewards = torch.zeros(n_envs, device=device)
    current_steps = torch.zeros(n_envs, device=device)
    
    while completed_episodes < n_episodes:
        # Get action
        with torch.no_grad():
            obs_t = obs.float()
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
        # Step
        next_obs, rewards, dones = vec_env.step(action)
        
        # Update trackers
        current_rewards += rewards
        current_steps += 1
        
        if dones.any():
            done_indices = dones.nonzero(as_tuple=True)[0]
            for idx in done_indices:
                if completed_episodes < n_episodes:
                    idx_item = idx.item()
                    # Record stats
                    episode_rewards.append(current_rewards[idx_item].item())
                    # Get final food count from env
                    episode_food.append(vec_env.last_episode_food[idx_item].item())
                    episode_steps.append(current_steps[idx_item].item())
                    
                    # Did it die or timeout?
                    if vec_env.agent_energy[idx_item] <= 0:
                        died_count += 1
                        
                    completed_episodes += 1
                    print(".", end="", flush=True)
            
            # Reset trackers for done environments
            current_rewards[dones] = 0
            current_steps[dones] = 0
            
            vec_env.reset(done_indices)
            
        obs = next_obs

    elapsed = time.time() - t_start
    print("\n")
    
    # Results
    import numpy as np
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_food = np.mean(episode_food)
    std_food = np.std(episode_food)
    avg_steps = np.mean(episode_steps)
    survival_rate = (1 - (died_count / len(episode_rewards))) * 100
    
    print("="*60)
    print(f"[RESULTS] BENCHMARK ({completed_episodes} episodes)")
    print("="*60)
    print(f"Reward:        {avg_reward:6.1f} +/- {std_reward:4.1f}")
    print(f"Food Eaten:    {avg_food:6.1f} +/- {std_food:4.1f}")
    print(f"Steps Alive:   {avg_steps:6.1f} / {vec_env.max_steps}")
    print(f"Survival Rate: {survival_rate:6.1f}%")
    print(f"Throughput:    {np.sum(episode_steps) / elapsed:.0f} steps/sec")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark trained agents offline")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--mode", type=str, default="ground_truth", help="Observation mode")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu/cuda)")
    
    args = parser.parse_args()
    
    benchmark(
        model_path=args.model,
        mode=args.mode,
        n_episodes=args.episodes,
        n_envs=args.envs,
        device_name=args.device,
    )
