
import torch
import numpy as np
from goodharts.configs.default_config import get_config, CellType
from goodharts.environments.world import World
from goodharts.agents.organism import Organism
from goodharts.behaviors import BehaviorStrategy
from goodharts.behaviors.brains import create_brain
from goodharts.behaviors.action_space import build_action_space, num_actions

def reproduction_train():
    print("🚀 Starting reproduction train script...")
    config = get_config()
    device = torch.device('cpu')
    
    # Setup world
    world = World(50, 50, config)
    world.place_food(100) # High density
    
    # Verify food placement
    food_count = np.sum(world.grid == CellType.FOOD.value)
    print(f"Food count on grid: {food_count}")
    
    # Create agent
    class MockBehavior(BehaviorStrategy):
        requirements = ['ground_truth']
        def decide_action(self, agent, view):
            return 0, 0 # Will be overridden
            
    agent = Organism(25, 25, 50.0, 5, world, MockBehavior(), config)
    
    # Create random policy
    spec = config['get_observation_spec']('ground_truth')
    n_actions = num_actions(1)
    action_space = build_action_space(1)
    
    print("Action space:", action_space)
    
    episode_food = 0
    
    for t in range(500):
        state = agent.get_local_view(mode='ground_truth')
        
        # Random action
        action_idx = np.random.randint(0, n_actions)
        dx, dy = action_space[action_idx]
        
        # Check if food is in view
        food_channel = state[2] # cell_food
        visible_food = np.sum(food_channel)
        
        # Execute
        agent.move(dx, dy)
        consumed = agent.eat()
        
        if consumed:
            print(f"[{t}] 🍎 EAT! {consumed}. Pos({agent.x}, {agent.y})")
            episode_food += 1
        else:
            # Check if we are standing on food (but eat failed?)
            if world.grid[agent.y, agent.x] == CellType.FOOD.value:
                print(f"[{t}] ❌ Standing on FOOD but eat() returned None!")
        
        if t % 50 == 0:
            print(f"[{t}] Pos: ({agent.x}, {agent.y}). Visible food: {visible_food}")
            
    print(f"Total food eaten: {episode_food}")

if __name__ == "__main__":
    reproduction_train()
