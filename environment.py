import numpy as np
from constants import *


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), np.int8)

    def _place_items(self, item_type: int, amount: int):
        count = 0
        while count < amount:
            randx = np.random.randint(0, self.width)
            randy = np.random.randint(0, self.height)
            if self.grid[randy, randx] == 0:
                self.grid[randy, randx] = item_type
                count += 1

    def place_food(self, amount: int):
        self._place_items(CellType.FOOD, amount)  # 2 = Food

    def place_poison(self, amount: int):
        self._place_items(CellType.POISON, amount)  # 3 = Poison