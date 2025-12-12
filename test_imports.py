from simulation import Simulation
from configs.default_config import get_config

try:
    print("Testing imports and instantiation...")
    config = get_config()
    print("Config loaded.")
    sim = Simulation(config)
    print("Simulation instantiated.")
    sim.step()
    print("Simulation step successful.")
    print("All checks passed.")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
