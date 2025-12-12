import numpy as np

class World:
    def __init__(self, width: int, height: int, config: dict):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), np.int8)
        self.config = config
        self.CellType = self.config['CellType']

    def _place_items(self, item_type: int, amount: int):
        count = 0
        while count < amount:
            randx = np.random.randint(0, self.width)
            randy = np.random.randint(0, self.height)
            if self.grid[randy, randx] == 0:
                self.grid[randy, randx] = item_type
                count += 1

    def place_food(self, amount: int):
        self._place_items(self.CellType.FOOD, amount)

    def place_poison(self, amount: int):
        self._place_items(self.CellType.POISON, amount)
