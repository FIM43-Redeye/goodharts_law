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
        cell_info = self.CellType.by_value(item_type)
        count = 0
        while count < amount:
            randx = np.random.randint(0, self.width)
            randy = np.random.randint(0, self.height)
            if self.grid[randy, randx] == 0:
                self.grid[randy, randx] = item_type
                
                # Update Proxy Grid from intrinsic cell properties
                # Both Food and Poison look "interesting" - the Goodhart trap!
                self.proxy_grid[randy, randx] = cell_info.interestingness if cell_info else 0.0
                
                count += 1

    def place_food(self, amount: int):
        self._place_items(self.CellType.FOOD, amount)

    def place_poison(self, amount: int):
        self._place_items(self.CellType.POISON, amount)
