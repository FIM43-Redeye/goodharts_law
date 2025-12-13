import numpy as np
from .base import Environment

# Import Numba-accelerated function (with fallback if Numba not available)
try:
    from ..utils.numba_utils import place_items_fast
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class World(Environment):
    def __init__(self, width: int, height: int, config: dict):
        self.width: int = width
        self.height: int = height
        self.grid: np.ndarray = np.zeros((self.height, self.width), np.int8)
        self.proxy_grid: np.ndarray = np.zeros((self.height, self.width), np.float32)
        self.config: dict = config
        self.CellType: type = self.config['CellType']

    @property
    def capabilities(self) -> list[str]:
        return ['ground_truth', 'proxy_metric']

    def _place_items(self, item_type, amount: int):
        cell_info = self.CellType.by_value(int(item_type))
        interestingness = cell_info.interestingness if cell_info else 0.0
        
        if HAS_NUMBA:
            # Use JIT-compiled placement
            place_items_fast(self.grid, self.proxy_grid, int(item_type), interestingness, amount)
        else:
            # Fallback to pure Python
            count = 0
            while count < amount:
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                if self.grid[y, x] == 0:
                    self.grid[y, x] = int(item_type)
                    self.proxy_grid[y, x] = interestingness
                    count += 1

    def place_food(self, amount: int):
        self._place_items(self.CellType.FOOD, amount)

    def place_poison(self, amount: int):
        self._place_items(self.CellType.POISON, amount)

