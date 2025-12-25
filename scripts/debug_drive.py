
import torch
import torch.nn.functional as F
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.configs.default_config import get_config
from goodharts.config import get_training_config
from goodharts.modes import ObservationSpec

def debug_drive():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup
    sim_config = get_config()
    obs_spec = ObservationSpec.for_mode('ground_truth', sim_config)
    env = create_torch_vec_env(n_envs=1, obs_spec=obs_spec, device=device)
    
    # Reset
    obs = env.reset()
    print(f"Initial Obs Shape: {obs.shape}")
    
    # Teleport agent to a known location for testing
    # Or just scan grid for food
    print("Scanning grid for food...")
    food_val = env.CellType.FOOD.value
    grid = env.grids[0] # (H, W)
    
    food_locs = (grid == food_val).nonzero(as_tuple=False)
    if len(food_locs) == 0:
        print("CRITICAL: No food on grid!")
        return
    
    target_y, target_x = food_locs[0]
    print(f"Target Food at: {target_y.item()}, {target_x.item()}")
    
    # Teleport agent NEXT to food
    # Actions: 0=(-1,-1), ... 7=(1,1)
    # Let's say we place agent at (y, x-1) and move Right (action 6: (1,0))?
    # Actions_8: 
    # 0:(-1,-1), 1:(-1,0), 2:(-1,1)
    # 3:(0,-1) [Up], 4:(0,1) [Down]
    # 5:(1,-1), 6:(1,0) [Right], 7:(1,1)
    
    start_y, start_x = target_y, target_x - 1
    env.agent_y[0] = start_y
    env.agent_x[0] = start_x
    print(f"Teleported agent to: {start_y.item()}, {start_x.item()}")
    
    # Step 1: Check Observation
    obs = env._get_observations()
    # Agent is at (start_y, start_x). Food is at (start_y, start_x + 1).
    # View radius 5. Center is (5, 5).
    # Food should be at (5, 5+1) = (5, 6) in the observation.
    # Channel 2 is Food.
    
    food_channel = obs[0, 2, :, :]
    
    print("\nObservation Food Channel (Center Crop 5x5):")
    # Crop center 11x11 view to 5x5 for readability
    r = 5
    print(food_channel[r-2:r+3, r-2:r+3])
    
    is_food_visible = food_channel[5, 6] > 0
    print(f"\nFood Visible at (5,6)? {is_food_visible.item()}")
    
    if not is_food_visible:
        print("FAILURE: Agent cannot see food right in front of it!")
        return

    # Step 2: Move Right (Action 6)
    print("\nExecuting Action 6 (Right)...")
    actions = torch.tensor([6], device=device) 
    obs, rewards, dones = env.step(actions)
    
    print(f"Reward Received: {rewards[0].item()}")
    
    if rewards[0].item() > 0:
        print("SUCCESS: Agent ate food!")
    else:
        print(f"FAILURE: Agent did not eat! Reward: {rewards[0].item()}")
        
    # Verify Agent Position
    print(f"New Position: {env.agent_y[0].item()}, {env.agent_x[0].item()}")
    if env.agent_x[0] == target_x and env.agent_y[0] == target_y:
        print("Position Verified: Agent is ON target.")
    else:
        print("Position Mismatch!")

if __name__ == "__main__":
    debug_drive()
