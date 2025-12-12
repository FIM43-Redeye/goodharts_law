from environment import World
from agent import Organism  # or whatever you named the class in agent.py
from constants import *
import numpy as np


class Simulation:
    def __init__(self):
        # 1. Set up the World
        self.world = World(GRID_WIDTH, GRID_HEIGHT)
        self.world.place_food(GRID_FOOD_INIT)  # Start with some food
        self.world.place_poison(GRID_POISON_INIT)

        # 2. Set up the Agents
        self.agents = []
        # Inside the loop:
        #   - Pick a random x and y (within grid bounds)
        #   - Create a new Organism(x, y, ENERGY_START)
        #   - Append it to self.agents
        for x in range(GRID_AGENTS_INIT):
            randx = np.random.randint(0, self.world.width)
            randy = np.random.randint(0, self.world.height)
            self.agents.append(Organism(randx, randy, ENERGY_START, AGENT_VIEW_RANGE, self.world))

    def step(self):
        # This is one "frame" of the simulation

        # We iterate over a copy of the list [:] because we might remove dead agents
        for agent in self.agents[:]:
            if not agent.alive:
                self.agents.remove(agent)
                continue

            dx, dy = agent.think()
            agent.move(dx, dy)
            agent.eat()

    def get_render_grid(self):
        # Create a copy so we don't actually overwrite the simulation physics
        render_grid = self.world.grid.copy()

        # Paint the agents on top
        for agent in self.agents:
            # Safety check: ensure agent is still within bounds before drawing
            if 0 <= agent.x < self.world.width and 0 <= agent.y < self.world.height:
                render_grid[agent.y, agent.x] = 4  # 4 represents the Agent color

        return render_grid