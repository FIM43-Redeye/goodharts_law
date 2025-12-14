
import time
import numpy as np
import torch
from goodharts.environments.vec_env import create_vec_env
from goodharts.configs.observation_spec import ObservationSpec
from goodharts.configs.default_config import get_config

def test_vec_env_rewards():
    print("🧪 Testing VecEnv Rewards...")
    
    # Setup
    config = get_config()
    # Force high food density to ensure eating
    config['GRID_FOOD_INIT'] = 1000 
    config['GRID_WIDTH'] = 50
    config['GRID_HEIGHT'] = 50
    
    spec = ObservationSpec.for_mode('ground_truth', config)
    env = create_vec_env(n_envs=4, obs_spec=spec, config=config)
    
    # Reset
    obs = env.reset()
    print(f"Initial Food Count (Env 0): {np.sum(env.grids[0] == 2)}")
    
    total_reward = 0
    steps = 100
    eaten_count = 0
    
    print("\n🏃 Running 100 steps (random walk)...")
    for i in range(steps):
        # Random actions
        actions = np.random.randint(0, 8, size=4)
        
        obs, rewards, dones = env.step(actions)
        
        # Check specifically for food rewards (should be 15.0)
        # We know food reward is 15.0
        eaten = rewards > 10.0
        if np.any(eaten):
            n_eaten = np.sum(eaten)
            eaten_count += n_eaten
            print(f"  Step {i}: Eaten! Rewards: {rewards[eaten]}")
            
        total_reward += np.sum(rewards)
        
        if np.any(dones):
            print(f"  Step {i}: Done! Resetting.")
            env.reset(np.where(dones)[0])
            
    print(f"\n📊 Summary:")
    print(f"  Total Food Eaten: {eaten_count}")
    print(f"  Total Reward Sum: {total_reward}")
    
    if eaten_count > 0 and total_reward > 0:
        print("✅ PASS: Rewards are being generated.")
    else:
        print("❌ FAIL: No rewards generated despite random walk in dense food.")

if __name__ == "__main__":
    try:
        test_vec_env_rewards()
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
