"""
Brain View: Single-agent neural network visualization using matplotlib.

Shows:
- Full grid with agent position marker
- Agent's local observation (what it sees)
- Neural network layer activations
- Action probabilities

Uses matplotlib with in-place artist updates for smooth animation.

GUI controls (bottom panel):
- Pause/Resume button
- Step button (advances one frame when paused)
- Speed text box (milliseconds per step)
- Sampling toggle (argmax vs multinomial)
"""
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from goodharts.utils.brain_viz import BrainVisualizer
from goodharts.visualization.utils import (
    observation_to_rgb, grid_with_agent_to_rgb
)


def get_action_labels(max_move_distance: int = 1) -> list[str]:
    """
    Generate action labels matching the actual action space ordering.

    Actions are ordered by iterating dx, dy from -max to +max (skipping 0,0).
    Coordinate system: +x=East, +y=South (screen coordinates).
    """
    from goodharts.behaviors.action_space import build_action_list

    actions = build_action_list(max_move_distance)
    labels = []

    for dx, dy in actions:
        # Map (dx, dy) to compass direction
        # +x = East, +y = South (screen coords where y increases downward)
        if dx == 0 and dy < 0:
            d = 'N'
        elif dx > 0 and dy < 0:
            d = 'NE'
        elif dx > 0 and dy == 0:
            d = 'E'
        elif dx > 0 and dy > 0:
            d = 'SE'
        elif dx == 0 and dy > 0:
            d = 'S'
        elif dx < 0 and dy > 0:
            d = 'SW'
        elif dx < 0 and dy == 0:
            d = 'W'
        elif dx < 0 and dy < 0:
            d = 'NW'
        else:
            d = '?'
        labels.append(d)

    return labels


# Default labels for 8-action space (max_move_distance=1)
# Order: NW, W, SW, N, S, NE, E, SE (matches build_action_list iteration)
ACTION_LABELS = get_action_labels(1)


class MatplotlibBrainView:
    """
    Matplotlib-based neural network visualization.

    Creates a figure with subplots for grid, observation, activations, and actions.
    Uses in-place artist updates for efficient animation.

    GUI controls are provided in a bottom panel for pause/step/speed/sampling.
    """

    def __init__(
        self,
        mode: str,
        model: torch.nn.Module,
        grid_size: tuple[int, int] = (128, 128),
        initial_speed_ms: int = 50,
    ):
        """
        Args:
            mode: Training mode name (ground_truth, proxy, etc.)
            model: Neural network model to visualize
            grid_size: Grid dimensions (height, width)
            initial_speed_ms: Initial step interval in milliseconds
        """
        self.mode = mode
        self.model = model
        self.grid_size = grid_size

        # Set up activation capture
        self.visualizer = BrainVisualizer(model)
        self.layer_names = self.visualizer.get_displayable_layers()

        # Configure matplotlib for dark theme
        plt.style.use('dark_background')

        # Playback control state
        self.paused = False
        self.step_requested = False
        self.quit_requested = False
        self.use_multinomial = False  # False = argmax, True = multinomial sampling

        # Speed control (milliseconds per step)
        self._speed_ms = initial_speed_ms

        # Create figure and artists
        self._setup_figure()

        # State
        self.step_count = 0
        self._closed = False

    @property
    def speed_ms(self) -> int:
        """Current speed in milliseconds."""
        return self._speed_ms

    @speed_ms.setter
    def speed_ms(self, value: int):
        """Set speed, clamped to reasonable range."""
        self._speed_ms = max(1, min(5000, value))

    def _setup_figure(self):
        """Create the figure layout with proper sizing and control panel."""
        n_layers = len(self.layer_names)

        # Dynamic layout based on layer count
        # Row 1: Grid | Activation layers... | Actions
        # Row 2: Observation | More activations... | Status
        # Bottom: Control panel with buttons

        # Calculate columns: 1 (grid/obs) + ceil(n_layers/2) + 1 (actions/status)
        layer_cols = max(1, (n_layers + 1) // 2)
        n_cols = 1 + layer_cols + 1

        # Width ratios: main views are larger than activations
        width_ratios = [2] + [1] * layer_cols + [1]

        # Create figure with extra space at bottom for controls
        fig_height = 7  # Slightly taller to accommodate controls
        self.fig, self.axes = plt.subplots(
            2, n_cols,
            figsize=(3 * n_cols, fig_height),
            gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.1, 'hspace': 0.15}
        )

        self.fig.canvas.manager.set_window_title(f'Brain View: {self.mode}')

        # Store artists for in-place updates
        self._artists = {}

        # Row 1, Col 1: Full grid
        ax = self.axes[0, 0]
        ax.set_title('Full Grid', fontsize=10, color='white')
        placeholder = np.zeros((*self.grid_size, 3), dtype=np.uint8)
        self._artists['grid'] = ax.imshow(placeholder, aspect='equal')
        ax.axis('off')

        # Row 2, Col 1: Agent observation
        ax = self.axes[1, 0]
        ax.set_title('Agent View', fontsize=10, color='white')
        obs_placeholder = np.zeros((11, 11, 3), dtype=np.uint8)
        self._artists['observation'] = ax.imshow(obs_placeholder, aspect='equal', interpolation='nearest')
        ax.axis('off')

        # Middle columns: Layer activations
        self._artists['activations'] = {}
        for i, layer_name in enumerate(self.layer_names):
            row = i % 2
            col = 1 + (i // 2)

            if col >= n_cols - 1:
                break

            ax = self.axes[row, col]
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            ax.set_title(short_name, fontsize=9, color='#888888')

            # Create colormap with background color for masked values
            cmap = plt.cm.hot.copy()
            cmap.set_bad(color='#0d0d0d')  # Dark background for masked cells

            # Placeholder heatmap
            placeholder = np.zeros((8, 8))
            im = ax.imshow(placeholder, cmap=cmap, aspect='auto', interpolation='nearest')
            self._artists['activations'][layer_name] = im
            ax.axis('off')

        # Clear unused activation slots
        for i in range(len(self.layer_names), layer_cols * 2):
            row = i % 2
            col = 1 + (i // 2)
            if col < n_cols - 1:
                self.axes[row, col].axis('off')

        # Last column, row 1: Action probabilities (horizontal bar)
        ax = self.axes[0, -1]
        ax.set_title('Actions', fontsize=10, color='white')
        self._action_bars = ax.barh(
            range(8), [0] * 8,
            color='#00d9ff',
            height=0.7
        )
        ax.set_yticks(range(8))
        ax.set_yticklabels(ACTION_LABELS, fontsize=8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.tick_params(axis='x', colors='#666666', labelsize=7)
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#444444')
        ax.spines['left'].set_color('#444444')

        # Last column, row 2: Energy/status
        ax = self.axes[1, -1]
        ax.set_title('Status', fontsize=10, color='white')
        ax.axis('off')
        self._status_text = ax.text(
            0.5, 0.5, '',
            ha='center', va='center',
            fontsize=12, color='white',
            transform=ax.transAxes
        )

        # Mode indicator as suptitle - use canonical colors
        from goodharts.visualization.components import MODE_COLORS
        mode_color = MODE_COLORS.get(self.mode, '#ffffff')
        self.fig.suptitle(
            f'{self.mode}',
            fontsize=14,
            color=mode_color,
            fontweight='bold'
        )

        # Connect close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # Adjust layout with space at bottom for controls
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        # Add control panel widgets at the bottom
        self._setup_controls()

        plt.ion()
        plt.show(block=False)

        # Try to prevent always-on-top behavior (backend-dependent)
        try:
            manager = self.fig.canvas.manager
            if hasattr(manager, 'window'):
                # For TkAgg backend
                if hasattr(manager.window, 'attributes'):
                    manager.window.attributes('-topmost', False)
                # For Qt backends
                elif hasattr(manager.window, 'setWindowFlags'):
                    from matplotlib.backends.qt_compat import QtCore
                    flags = manager.window.windowFlags()
                    manager.window.setWindowFlags(flags & ~QtCore.Qt.WindowStaysOnTopHint)
                    manager.window.show()
        except Exception:
            pass  # Not all backends support this

    def _setup_controls(self):
        """Set up the control panel with buttons and text input."""
        # Button styling
        button_color = '#333333'
        button_hover = '#555555'
        text_color = 'white'

        # Pause/Resume button
        ax_pause = self.fig.add_axes([0.05, 0.02, 0.12, 0.04])
        self._btn_pause = Button(ax_pause, 'Pause', color=button_color, hovercolor=button_hover)
        self._btn_pause.label.set_color(text_color)
        self._btn_pause.on_clicked(self._on_pause_clicked)

        # Step button
        ax_step = self.fig.add_axes([0.19, 0.02, 0.10, 0.04])
        self._btn_step = Button(ax_step, 'Step', color=button_color, hovercolor=button_hover)
        self._btn_step.label.set_color(text_color)
        self._btn_step.on_clicked(self._on_step_clicked)

        # Speed label and text box
        ax_speed_label = self.fig.add_axes([0.32, 0.02, 0.08, 0.04])
        ax_speed_label.axis('off')
        ax_speed_label.text(0.5, 0.5, 'Speed (ms):', ha='center', va='center',
                           color=text_color, fontsize=9, transform=ax_speed_label.transAxes)

        ax_speed = self.fig.add_axes([0.41, 0.02, 0.08, 0.04])
        self._txt_speed = TextBox(ax_speed, '', initial=str(self._speed_ms),
                                  color=button_color, hovercolor=button_hover)
        self._txt_speed.text_disp.set_color(text_color)
        self._txt_speed.on_submit(self._on_speed_submit)

        # Sampling mode toggle button
        ax_sampling = self.fig.add_axes([0.52, 0.02, 0.14, 0.04])
        sampling_label = 'multinomial' if self.use_multinomial else 'argmax'
        self._btn_sampling = Button(ax_sampling, f'Sample: {sampling_label}',
                                    color=button_color, hovercolor=button_hover)
        self._btn_sampling.label.set_color(text_color)
        self._btn_sampling.on_clicked(self._on_sampling_clicked)

        # Quit button
        ax_quit = self.fig.add_axes([0.85, 0.02, 0.10, 0.04])
        self._btn_quit = Button(ax_quit, 'Quit', color='#662222', hovercolor='#883333')
        self._btn_quit.label.set_color(text_color)
        self._btn_quit.on_clicked(self._on_quit_clicked)

    def _on_pause_clicked(self, event):
        """Handle pause button click."""
        self.paused = not self.paused
        label = 'Resume' if self.paused else 'Pause'
        self._btn_pause.label.set_text(label)

    def _on_step_clicked(self, event):
        """Handle step button click."""
        if self.paused:
            self.step_requested = True

    def _on_speed_submit(self, text):
        """Handle speed text box submission."""
        try:
            value = int(text.strip())
            self.speed_ms = value
            # Update text box to show clamped value
            self._txt_speed.set_val(str(self._speed_ms))
        except ValueError:
            # Reset to current value on invalid input
            self._txt_speed.set_val(str(self._speed_ms))

    def _on_sampling_clicked(self, event):
        """Handle sampling mode toggle."""
        self.use_multinomial = not self.use_multinomial
        label = 'multinomial' if self.use_multinomial else 'argmax'
        self._btn_sampling.label.set_text(f'Sample: {label}')

    def _on_quit_clicked(self, event):
        """Handle quit button click."""
        self.quit_requested = True

    def _on_close(self, event):
        """Handle figure close event."""
        self._closed = True

    def update(
        self,
        grid: torch.Tensor,
        agent_x: int,
        agent_y: int,
        agent_energy: float,
        obs: torch.Tensor,
        action_probs: np.ndarray,
        step_count: int,
    ):
        """
        Update visualization with new state.

        Args:
            grid: (H, W) grid tensor with CellType values
            agent_x, agent_y: Agent position
            agent_energy: Current energy
            obs: (C, H, W) observation tensor
            action_probs: (8,) action probability distribution
            step_count: Current simulation step
        """
        if self._closed:
            return

        self.step_count = step_count

        # Update grid image
        grid_rgb = grid_with_agent_to_rgb(grid, agent_x, agent_y)
        self._artists['grid'].set_data(grid_rgb)

        # Update observation image
        obs_rgb = observation_to_rgb(obs, self.mode)
        self._artists['observation'].set_data(obs_rgb)

        # Update activation heatmaps
        for layer_name, artist in self._artists['activations'].items():
            activation = self.visualizer.get_activation_display(layer_name)
            if activation is not None:
                artist.set_data(activation)
                # Auto-scale color limits
                vmin, vmax = activation.min(), activation.max()
                if vmax > vmin:
                    artist.set_clim(vmin, vmax)

        # Update action bars
        chosen_idx = int(action_probs.argmax())
        for i, (bar, prob) in enumerate(zip(self._action_bars, action_probs)):
            bar.set_width(prob)
            bar.set_color('#ff6b6b' if i == chosen_idx else '#00d9ff')

        # Update status text (controls are in bottom panel)
        self._status_text.set_text(
            f'Step: {step_count:,}\n'
            f'Energy: {agent_energy:.1f}\n'
            f'Action: {ACTION_LABELS[chosen_idx]}'
        )

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def pause(self, interval: float = 0.001):
        """
        Process matplotlib events and pause briefly without raising window.

        Args:
            interval: Pause duration in seconds
        """
        # plt.pause() raises the window, so we use the backend directly
        self.fig.canvas.start_event_loop(interval)
        # If start_event_loop doesn't work well, fall back to manual approach
        # time.sleep(interval)
        # self.fig.canvas.flush_events()

    def is_open(self) -> bool:
        """Check if figure is still open."""
        return not self._closed and plt.fignum_exists(self.fig.number)

    def close(self):
        """Close the figure."""
        if not self._closed:
            plt.close(self.fig)
            self._closed = True


def create_brain_view(
    mode: str,
    model: torch.nn.Module,
    grid_size: tuple[int, int] = (128, 128),
    initial_speed_ms: int = 50,
) -> MatplotlibBrainView:
    """
    Factory function for brain view visualization.

    Args:
        mode: Training mode name
        model: Neural network model to visualize
        grid_size: Grid dimensions (H, W)
        initial_speed_ms: Initial step interval in milliseconds

    Returns:
        MatplotlibBrainView instance ready for updates
    """
    return MatplotlibBrainView(mode, model, grid_size, initial_speed_ms)


# Legacy compatibility alias
BrainViewApp = MatplotlibBrainView
create_brain_view_app = create_brain_view
