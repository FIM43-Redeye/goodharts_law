import cProfile
import pstats
from goodharts.simulation import Simulation
from goodharts.configs.default_config import get_config
from goodharts.utils.logging_config import setup_logging
import logging

# Disable logging for profiling to measure core logic
setup_logging(level=logging.WARNING)

def run_simulation_steps(steps=200):
    config = get_config()
    # Increase load for profiling
    config['AGENTS_SETUP'] = [
        {'behavior_class': 'OmniscientSeeker', 'count': 50},
        {'behavior_class': 'ProxySeeker', 'count': 50}
    ]
    sim = Simulation(config)
    for _ in range(steps):
        sim.step()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation_steps()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
