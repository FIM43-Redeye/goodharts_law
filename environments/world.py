import numpy as np
from .base import Environment

class World(Environment):
    def __init__(self, width: int, height: int, config: dict):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), np.int8)
        self.proxy_grid = np.zeros((self.height, self.width), np.float32) # The "Goodhart" layer
        self.config = config
        self.CellType = self.config['CellType']

    @property
    def capabilities(self) -> list[str]:
        return ['ground_truth', 'proxy_metric']

    def _place_items(self, item_type: int, amount: int):
        count = 0
        while count < amount:
            randx = np.random.randint(0, self.width)
            randy = np.random.randint(0, self.height)
            if self.grid[randy, randx] == 0:
                self.grid[randy, randx] = item_type
                
                # Update Proxy Grid
                # Both Food and Poison look "interesting" (high value)
                if item_type == self.CellType.FOOD:
                    self.proxy_grid[randy, randx] = 1.0
                elif item_type == self.CellType.POISON:
                    self.proxy_grid[randy, randx] = 0.9 # Slightly less, or same. High enough to confuse.
                
                count += 1

    def place_food(self, amount: int):
        self._place_items(self.CellType.FOOD, amount)

    def place_poison(self, amount: int):
        self._place_items(self.CellType.POISON, amount)
