from environments import create_world
from agents import Organism
from behaviors import OmniscientSeeker, ProxySeeker
import numpy as np

# Mapping string names to classes
BEHAVIORS = {
    'OmniscientSeeker': OmniscientSeeker,
    'ProxySeeker': ProxySeeker
}

class Simulation:
    def __init__(self, config):
        self.config = config
        self.world = create_world(config)
        self.world.place_food(config['GRID_FOOD_INIT'])
        self.world.place_poison(config['GRID_POISON_INIT'])

        self.agents = []
        
        for setup in config['AGENTS_SETUP']:
            b_class_name = setup['behavior_class']
            count = setup['count']
            
            if b_class_name not in BEHAVIORS:
                raise ValueError(f"Unknown behavior class: {b_class_name}")
            
            BehaviorClass = BEHAVIORS[b_class_name]
            
            for _ in range(count):
                randx = np.random.randint(0, self.world.width)
                randy = np.random.randint(0, self.world.height)
                behavior = BehaviorClass()
                self.agents.append(Organism(randx, randy, config['ENERGY_START'], config['AGENT_VIEW_RANGE'], self.world, behavior, config))

    def step(self):
        for agent in self.agents[:]:
            if not agent.alive:
                self.agents.remove(agent)
                continue
            agent.update()

    def get_render_grid(self):
        render_grid = self.world.grid.copy()
        for agent in self.agents:
            if 0 <= agent.x < self.world.width and 0 <= agent.y < self.world.height:
                # 4 for Omniscient, 5 for Proxy
                val = 4
                if isinstance(agent.behavior, ProxySeeker):
                    val = 5
                render_grid[agent.y, agent.x] = val
        return render_grid

