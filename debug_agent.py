
import numpy as np
import torch
from goodharts.environments.world import World
from goodharts.agents.organism import Organism
from goodharts.configs.default_config import get_config, CellType
from goodharts.behaviors import BehaviorStrategy

# Mock behavior
class MockBehavior(BehaviorStrategy):
    @property
    def requirements(self):
        return ['ground_truth']
    def decide_action(self, agent, view):
        return (0, 0)

def debug_agent():
    config = get_config()
    # Force bounded world
    config['WORLD_LOOP'] = False
    
    world = World(10, 10, config)
    agent = Organism(5, 5, 50.0, 2, world, MockBehavior(), config)
    
    print(f"Initial Energy: {agent.energy}")
    
    # Place food at (6, 5) - to the right of agent
    world.grid[5, 6] = CellType.FOOD.value
    print(f"Placed FOOD at (6, 5). Agent at ({agent.x}, {agent.y})")
    
    # Check view
    view = agent.get_local_view()
    print(f"View shape: {view.shape}")
    
    # Find channel indices
    # Ground truth channels: empty, wall, food, poison, prey, predator
    # cell_food should be index 2 (0=empty, 1=wall, 2=food)
    # Let's verify by printing channel sums
    from goodharts.configs.observation_spec import ObservationSpec
    spec = ObservationSpec.for_mode('ground_truth', config)
    print(f"Channels: {spec.channel_names}")
    
    food_idx = spec.channel_names.index('cell_food')
    print(f"Food channel index: {food_idx}")
    
    food_channel = view[food_idx]
    print(f"Food channel sum: {food_channel.sum()}")
    print("Food channel grid:")
    print(food_channel)
    
    # Center is at radius=2 => index 2
    # Food at relative (1, 0) => grid[2, 3] (y, x) ?
    # View is (2r+1) x (2r+1). 
    # y range: 5-2 to 5+2 => 3,4,5,6,7. Relative 0,1,2,3,4. Center=2.
    # x range: 5-2 to 5+2 => 3,4,5,6,7. Relative 0,1,2,3,4. Center=2.
    # Food at (6,5) => x=6, y=5.
    # In view: y=5 corresponds to relative index 2.
    # x=6 corresponds to relative index 3.
    # So we expect val at [2, 3].
    
    val = food_channel[2, 3]
    print(f"Value at relative (2, 3): {val}")
    
    if val != 1.0:
        print("❌ ERROR: Agent cannot see food at (6, 5)!")
    else:
        print("✅  Agent sees food correctly.")
        
    # Check blanking self
    # Place food UNDER agent
    world.grid[5, 5] = CellType.FOOD.value
    view_under = agent.get_local_view()
    val_center = view_under[food_idx, 2, 2]
    print(f"Value at center (under agent): {val_center}")
    if val_center != 0.0:
        print("❌ ERROR: Center cell not blanked!")
    else:
        print("✅ Center cell correctly blanked.")
        
    # Check eating
    print("Attempting to eat...")
    energy_before = agent.energy
    res = agent.eat()
    print(f"Eat result: {res}")
    print(f"Energy after: {agent.energy} (Delta: {agent.energy - energy_before})")
    
    if agent.energy > energy_before:
        print("✅ Eating works.")
    else:
        print("❌ Eating failed.")

    # Check shaping
    from goodharts.training.reward_shaping import compute_shaping_score, PREY_TARGETS
    score = compute_shaping_score(view, (2, 2), PREY_TARGETS)
    print(f"Shaping score with food at distance 1: {score}")
    # Dist 1, weight 0.5 => 0.5.
    if abs(score - 0.5) < 0.01:
         print("✅ Shaping score correct.")
    else:
         print(f"❌ Shaping score incorrect. Expected 0.5, got {score}")

if __name__ == "__main__":
    debug_agent()
