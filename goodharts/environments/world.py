import numpy as np
from goodharts.environments.base import Environment

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
        self.loop: bool = config.get('WORLD_LOOP', False)
        self.grid: np.ndarray = np.zeros((self.height, self.width), np.int8)
        self.proxy_grid: np.ndarray = np.zeros((self.height, self.width), np.float32)
        self.config: dict = config
        self.CellType: type = self.config['CellType']

    @property
    def capabilities(self) -> list[str]:
        return ['ground_truth', 'proxy_metric']

    def wrap_position(self, x: int, y: int) -> tuple[int, int]:
        """
        Normalize coordinates based on world topology.
        
        In loop mode: wraps around edges (toroidal)
        In bounded mode: clamps to valid range
        """
        if self.loop:
            return x % self.width, y % self.height
        return max(0, min(x, self.width - 1)), max(0, min(y, self.height - 1))

    def get_view(self, grid: np.ndarray, x: int, y: int, radius: int, 
                 fill_value: float = 0.0) -> np.ndarray:
        """
        Extract local view centered at (x, y) with given radius.
        
        In loop mode: wraps around edges
        In bounded mode: pads with fill_value
        
        Returns:
            Array of shape (2*radius+1, 2*radius+1)
        """
        size = 2 * radius + 1
        
        if self.loop:
            # Toroidal wrapping - use numpy advanced indexing
            xs = np.arange(x - radius, x + radius + 1) % self.width
            ys = np.arange(y - radius, y + radius + 1) % self.height
            return grid[np.ix_(ys, xs)]
        else:
            # Bounded - extract with padding
            x_min, x_max = x - radius, x + radius
            y_min, y_max = y - radius, y + radius
            
            x_slice_start = max(0, x_min)
            x_slice_end = min(self.width, x_max + 1)
            y_slice_start = max(0, y_min)
            y_slice_end = min(self.height, y_max + 1)
            
            world_view = grid[y_slice_start:y_slice_end, x_slice_start:x_slice_end]
            
            pad_top = y_slice_start - y_min
            pad_bottom = (y_max + 1) - y_slice_end
            pad_left = x_slice_start - x_min
            pad_right = (x_max + 1) - x_slice_end
            
            return np.pad(
                world_view,
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=fill_value
            )

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


