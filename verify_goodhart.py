from simulation import Simulation
from configs.default_config import get_config
from behaviors import OmniscientSeeker, ProxySeeker
import numpy as np

def run_verification():
    config = get_config()
    # Make it harder and more likely to hit poison
    config['ENERGY_START'] = 100.0
    config['ENERGY_MOVE_COST'] = 0.5
    config['GRID_POISON_INIT'] = 100 
    config['GRID_FOOD_INIT'] = 100
    config['AGENTS_SETUP'] = [
        {'behavior_class': 'OmniscientSeeker', 'count': 10},
        {'behavior_class': 'ProxySeeker', 'count': 10}
    ]
    
    sim = Simulation(config)
    print(f"Initialized Simulation.")
    
    omniscient_count = 0
    proxy_count = 0
    for agent in sim.agents:
        if isinstance(agent.behavior, OmniscientSeeker):
            omniscient_count += 1
        elif isinstance(agent.behavior, ProxySeeker):
            proxy_count += 1
            
    print(f"Start: Omniscient={omniscient_count}, Proxy={proxy_count}")
    
    # Run 500 steps
    initial_energy = [a.energy for a in sim.agents]
    print(f"Avg Initial Energy: {np.mean(initial_energy)}")
    
    for i in range(500):
        sim.step()
        if i % 100 == 0:
            o = sum(1 for a in sim.agents if isinstance(a.behavior, OmniscientSeeker))
            p = sum(1 for a in sim.agents if isinstance(a.behavior, ProxySeeker))
            print(f"Step {i}: Omniscient={o}, Proxy={p}")
            
    final_o = sum(1 for a in sim.agents if isinstance(a.behavior, OmniscientSeeker))
    final_p = sum(1 for a in sim.agents if isinstance(a.behavior, ProxySeeker))
    
    print(f"Result: Omniscient={final_o}, Proxy={final_p}")
    
    if final_o >= final_p:
        print("SUCCESS: Omniscient agents survived better or equal.")
    else:
        print("WARNING: Proxy agents survived better? Check logic.")

if __name__ == "__main__":
    run_verification()
