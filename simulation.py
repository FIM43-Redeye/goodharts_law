from environments import World
from agents import Organism
from behaviors import GreedyFoodSeeker
import numpy as np

class Simulation:
    def __init__(self, config):
        self.config = config
        self.world = World(config['GRID_WIDTH'], config['GRID_HEIGHT'], config)
        self.world.place_food(config['GRID_FOOD_INIT'])
        self.world.place_poison(config['GRID_POISON_INIT'])

        self.agents = []
        for _ in range(config['GRID_AGENTS_INIT']):
            randx = np.random.randint(0, self.world.width)
            randy = np.random.randint(0, self.world.height)
            behavior = GreedyFoodSeeker()
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
                render_grid[agent.y, agent.x] = 4
        return render_grid

