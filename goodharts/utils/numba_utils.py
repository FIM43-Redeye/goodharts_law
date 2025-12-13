"""
Numba-accelerated simulation utilities.

These are standalone functions (not methods) that can be JIT-compiled
for significant speedups on the hot simulation loops.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def extract_grid_view_fast(
    grid: np.ndarray,
    x: int, y: int,
    sight_radius: int,
    world_width: int,
    world_height: int,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Extract local view from a grid with padding for edges.
    JIT-compiled for speed.
    
    Args:
        grid: 2D numpy array (height, width)
        x, y: Agent position (x=col, y=row)
        sight_radius: How far agent can see
        world_width, world_height: Grid dimensions
        fill_value: Value for out-of-bounds cells
    
    Returns:
        (2*sight_radius+1, 2*sight_radius+1) view centered on agent
    """
    view_size = 2 * sight_radius + 1
    result = np.full((view_size, view_size), fill_value, dtype=grid.dtype)
    
    # Calculate source (grid) and destination (result) ranges
    x_min = x - sight_radius
    x_max = x + sight_radius
    y_min = y - sight_radius
    y_max = y + sight_radius
    
    # Clamp source to grid bounds
    src_x_start = max(0, x_min)
    src_x_end = min(world_width, x_max + 1)
    src_y_start = max(0, y_min)
    src_y_end = min(world_height, y_max + 1)
    
    # Calculate destination offsets
    dst_x_start = src_x_start - x_min
    dst_y_start = src_y_start - y_min
    
    # Copy valid region
    for dy in range(src_y_end - src_y_start):
        for dx in range(src_x_end - src_x_start):
            result[dst_y_start + dy, dst_x_start + dx] = grid[src_y_start + dy, src_x_start + dx]
    
    return result


@njit(cache=True)
def place_items_fast(
    grid: np.ndarray,
    proxy_grid: np.ndarray,
    item_value: int,
    interestingness: float,
    count: int
) -> None:
    """
    Place items randomly on empty cells. Modifies grids in-place.
    JIT-compiled for speed.
    
    Args:
        grid: Main cell grid
        proxy_grid: Proxy signal grid
        item_value: Cell value to place (2=food, 3=poison)
        interestingness: Proxy signal value
        count: Number of items to place
    """
    height, width = grid.shape
    placed = 0
    max_attempts = count * 100  # Prevent infinite loop
    attempts = 0
    
    while placed < count and attempts < max_attempts:
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        if grid[y, x] == 0:  # Empty cell
            grid[y, x] = item_value
            proxy_grid[y, x] = interestingness
            placed += 1
        
        attempts += 1


@njit(cache=True)
def compute_move_cost(dx: int, dy: int, base_cost: float, exponent: float) -> float:
    """Compute energy cost for movement."""
    distance = np.sqrt(dx * dx + dy * dy)
    return (distance ** exponent) * base_cost


@njit(cache=True)
def build_onehot_channel(grid_view: np.ndarray, cell_value: int) -> np.ndarray:
    """Create one-hot binary mask for a specific cell type."""
    result = np.zeros(grid_view.shape, dtype=np.float32)
    height, width = grid_view.shape
    for y in range(height):
        for x in range(width):
            if grid_view[y, x] == cell_value:
                result[y, x] = 1.0
    return result


@njit(cache=True)
def build_property_channel(grid_view: np.ndarray, cell_values: np.ndarray, 
                           property_values: np.ndarray) -> np.ndarray:
    """
    Map cell values to property values (e.g., interestingness).
    
    Args:
        grid_view: View of cell types
        cell_values: Array of cell type integer values
        property_values: Corresponding property values for each cell type
    """
    result = np.zeros(grid_view.shape, dtype=np.float32)
    height, width = grid_view.shape
    n_types = len(cell_values)
    
    for y in range(height):
        for x in range(width):
            cell = grid_view[y, x]
            for i in range(n_types):
                if cell == cell_values[i]:
                    result[y, x] = property_values[i]
                    break
    return result
