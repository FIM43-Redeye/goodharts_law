"""
Utility functions for converting PyTorch data to Plotly-compatible formats.

Provides efficient tensor-to-image conversion for visualization.
"""
import numpy as np
import torch
import plotly.graph_objects as go

from goodharts.configs.default_config import CellType


def tensor_to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert any tensor to CPU numpy array.

    Handles both PyTorch tensors and numpy arrays gracefully.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def grid_to_rgb(grid: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert grid tensor to RGB image for Plotly.

    Args:
        grid: (H, W) tensor/array with CellType values (0=empty, 1=food, 2=poison)

    Returns:
        (H, W, 3) uint8 numpy array with RGB colors
    """
    grid_np = tensor_to_numpy(grid).astype(int)
    h, w = grid_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Map cell types to colors
    for ct in CellType.all_types():
        mask = grid_np == ct.value
        rgb[mask] = ct.color

    return rgb


def observation_to_rgb(
    obs: torch.Tensor | np.ndarray,
    mode: str,
    agent_pos: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Convert 2-channel observation to RGB visualization.

    Args:
        obs: (2, H, W) observation tensor
        mode: Training mode ('ground_truth', 'proxy', etc.)
        agent_pos: Optional (y, x) position to mark agent (center if None)

    Returns:
        (H, W, 3) uint8 array with visualization colors
    """
    obs_np = tensor_to_numpy(obs)
    h, w = obs_np.shape[1], obs_np.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Background color (empty)
    rgb[:] = CellType.EMPTY.color

    if 'proxy' in mode or 'blinded' in mode:
        # Proxy/blinded mode: interestingness shown as intensity
        # Both channels have the same interestingness value
        # Higher interestingness = brighter (poison is brighter than food)
        intensity = obs_np[0]  # Channel 0 has interestingness
        # Normalize to 0-255 range
        if intensity.max() > 0:
            intensity_norm = (intensity / intensity.max() * 255).astype(np.uint8)
        else:
            intensity_norm = np.zeros_like(intensity, dtype=np.uint8)
        # Show as grayscale with slight purple tint for "interestingness"
        rgb[:, :, 0] = intensity_norm
        rgb[:, :, 1] = (intensity_norm * 0.6).astype(np.uint8)
        rgb[:, :, 2] = intensity_norm
    else:
        # Ground truth: channel 0 = food, channel 1 = poison
        food_mask = obs_np[0] > 0.5
        poison_mask = obs_np[1] > 0.5
        rgb[food_mask] = CellType.FOOD.color
        rgb[poison_mask] = CellType.POISON.color

    # Mark agent position
    if agent_pos is not None:
        y, x = agent_pos
        if 0 <= y < h and 0 <= x < w:
            rgb[y, x] = [255, 255, 255]  # White for agent
    else:
        # Default: agent at center
        cy, cx = h // 2, w // 2
        rgb[cy, cx] = [255, 255, 255]

    return rgb


def grid_with_agent_to_rgb(
    grid: torch.Tensor | np.ndarray,
    agent_x: int,
    agent_y: int,
) -> np.ndarray:
    """
    Convert grid to RGB with agent marker overlay.

    Args:
        grid: (H, W) grid tensor with CellType values
        agent_x: Agent X coordinate
        agent_y: Agent Y coordinate

    Returns:
        (H, W, 3) uint8 array with grid and white agent marker
    """
    rgb = grid_to_rgb(grid)

    # Mark agent position with white circle effect
    h, w = rgb.shape[:2]
    if 0 <= agent_y < h and 0 <= agent_x < w:
        # White center
        rgb[agent_y, agent_x] = [255, 255, 255]
        # Simple cross marker for visibility
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = agent_y + dy, agent_x + dx
            if 0 <= ny < h and 0 <= nx < w:
                # Bright cyan outline
                rgb[ny, nx] = [0, 255, 255]

    return rgb


def create_heatmap_figure(
    data: np.ndarray,
    title: str,
    colorscale: str = 'Hot',
    showscale: bool = True,
) -> go.Figure:
    """
    Create a Plotly heatmap figure.

    Args:
        data: 2D array to visualize
        title: Figure title
        colorscale: Plotly colorscale name
        showscale: Whether to show colorbar

    Returns:
        Plotly figure
    """
    from goodharts.visualization.components import apply_dark_theme

    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        showscale=showscale,
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor='x', constrain='domain', autorange='reversed'),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return apply_dark_theme(fig)


def create_image_figure(
    rgb: np.ndarray,
    title: str = '',
) -> go.Figure:
    """
    Create a Plotly figure from RGB array.

    Args:
        rgb: (H, W, 3) uint8 array
        title: Optional figure title

    Returns:
        Plotly figure displaying the image
    """
    from goodharts.visualization.components import apply_dark_theme

    fig = go.Figure(data=go.Image(z=rgb))

    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor='x', constrain='domain'),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return apply_dark_theme(fig)
