"""
Brain View: Single-agent neural network visualization.

Shows:
- Full grid with agent position marker
- Agent's local observation (what it sees)
- Neural network layer activations
- Action probabilities

Uses Plotly/Dash in a separate process for isolation from the simulation loop.
"""
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from goodharts.visualization.components import (
    THEME, get_mode_color, get_dark_theme_index_string, apply_dark_theme
)
from goodharts.visualization.utils import (
    tensor_to_numpy, grid_to_rgb, observation_to_rgb, grid_with_agent_to_rgb
)


# Update interval for dashboard polling
UPDATE_INTERVAL_MS = 50

# Direction labels for action bar chart
ACTION_LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


@dataclass
class BrainViewState:
    """
    State data sent from simulation to dashboard.

    All arrays are numpy (already on CPU) for serialization.
    """
    # Grid and agent position
    grid_rgb: np.ndarray               # (H, W, 3) full grid as RGB
    agent_x: int
    agent_y: int
    agent_energy: float

    # Agent's observation
    local_view_rgb: np.ndarray         # (view_size, view_size, 3)

    # Neural network state
    layer_activations: dict = field(default_factory=dict)  # layer_name -> (H, W) display array
    action_probs: np.ndarray = field(default_factory=lambda: np.zeros(8))

    # Metadata
    step_count: int = 0
    mode: str = 'ground_truth'


class BrainViewDashboard:
    """
    Plotly/Dash dashboard for neural network visualization.

    Runs in a separate process and receives state updates via queue.
    """

    def __init__(self, mode: str, layer_names: list[str], grid_size: tuple[int, int]):
        self.mode = mode
        self.layer_names = layer_names
        self.grid_size = grid_size
        self.state: Optional[BrainViewState] = None
        self.update_queue: queue.Queue = queue.Queue()

    def send_state(self, state: BrainViewState):
        """Queue a state update (called from main process via IPC)."""
        try:
            self.update_queue.put_nowait(state)
        except queue.Full:
            pass

    def _process_updates(self):
        """Process pending state updates from queue."""
        # Keep only the most recent state
        latest = None
        while not self.update_queue.empty():
            try:
                latest = self.update_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.state = latest

    def _calculate_layout(self) -> tuple[int, int, list[list]]:
        """
        Calculate subplot layout based on number of layers.

        Returns:
            (n_rows, n_cols, specs) for make_subplots
        """
        n_layers = len(self.layer_names)

        # Layout: Grid | Layers... | Actions
        #         View |           |

        # Minimum 3 columns (grid/view, layers, actions)
        # Layers go in pairs vertically
        layer_cols = max(1, (n_layers + 1) // 2)
        n_cols = 1 + layer_cols + 1  # grid + layers + actions

        # Always 2 rows
        n_rows = 2

        # Build specs - all are images except action bar
        specs = []
        for row in range(n_rows):
            row_specs = []
            for col in range(n_cols):
                if col == n_cols - 1:  # Last column is actions
                    row_specs.append({'type': 'xy'})
                else:
                    row_specs.append({'type': 'image'})
            specs.append(row_specs)

        return n_rows, n_cols, specs

    def _create_figure(self) -> go.Figure:
        """Create the main visualization figure with dynamic layout."""
        n_rows, n_cols, specs = self._calculate_layout()

        # Build subplot titles
        titles = ['Full Grid', 'Agent View']
        # Add layer names for middle columns
        for i, name in enumerate(self.layer_names):
            titles.append(name)
        # Pad if we have fewer layers than slots
        layer_slots = (n_cols - 2) * 2
        while len(titles) < 2 + layer_slots:
            titles.append('')
        # Add action title
        titles.append('Actions')
        titles.append('')  # Second row of actions column

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=titles[:n_rows * n_cols],
            specs=specs,
            horizontal_spacing=0.03,
            vertical_spacing=0.1,
        )

        return apply_dark_theme(fig)

    def _build_figure(self) -> go.Figure:
        """Build figure from current state."""
        fig = self._create_figure()
        n_rows, n_cols, _ = self._calculate_layout()

        if self.state is None:
            # Return empty figure with placeholder
            fig.add_trace(go.Image(z=np.zeros((10, 10, 3), dtype=np.uint8)), row=1, col=1)
            return fig

        s = self.state

        # 1. Full grid (row 1, col 1)
        fig.add_trace(go.Image(z=s.grid_rgb), row=1, col=1)

        # 2. Agent's local view (row 2, col 1)
        fig.add_trace(go.Image(z=s.local_view_rgb), row=2, col=1)

        # 3. Layer activations (middle columns)
        layer_col_start = 2
        for i, layer_name in enumerate(self.layer_names):
            row = (i % 2) + 1  # Alternate rows
            col = layer_col_start + (i // 2)

            if col > n_cols - 1:  # Don't overflow into actions column
                break

            activation = s.layer_activations.get(layer_name)
            if activation is not None:
                fig.add_trace(
                    go.Heatmap(z=activation, colorscale='Hot', showscale=False),
                    row=row, col=col
                )

        # 4. Action probabilities (last column, spans both rows conceptually)
        chosen_idx = int(s.action_probs.argmax())
        colors = ['#ff6b6b' if i == chosen_idx else '#00d9ff'
                  for i in range(len(s.action_probs))]

        fig.add_trace(
            go.Bar(
                y=ACTION_LABELS,
                x=s.action_probs.tolist(),
                orientation='h',
                marker_color=colors,
                showlegend=False,
            ),
            row=1, col=n_cols
        )

        # Update action axis
        fig.update_xaxes(range=[0, 1], row=1, col=n_cols)

        # Title with status
        mode_color = get_mode_color(self.mode)
        title = (f'<span style="color:{mode_color}">{self.mode}</span> | '
                 f'Step {s.step_count:,} | Energy: {s.agent_energy:.1f}')

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color=THEME['text'])),
            height=700,
            showlegend=False,
        )

        # Hide tick labels on image axes
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols):  # Exclude action bar column
                fig.update_xaxes(showticklabels=False, row=row, col=col)
                fig.update_yaxes(showticklabels=False, row=row, col=col)

        return fig

    def run(self, port: int = 8050):
        """Start the Dash server (blocking)."""
        app = Dash(__name__)
        app.index_string = get_dark_theme_index_string('Brain View')

        initial_fig = self._create_figure()

        app.layout = html.Div([
            dcc.Graph(id='brain-view', figure=initial_fig,
                      style={'height': '95vh'},
                      config={'displayModeBar': False}),
            dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),
        ], style={'backgroundColor': THEME['background'], 'padding': '5px'})

        @app.callback(
            Output('brain-view', 'figure'),
            Input('interval', 'n_intervals')
        )
        def update(n):
            self._process_updates()
            return self._build_figure()

        print(f"[BrainView] Dashboard running at http://localhost:{port}")
        app.run(debug=False, use_reloader=False, port=port)


class BrainViewApp:
    """
    Process-isolated brain view application.

    Runs visualization in separate process to avoid blocking simulation.
    Communication happens via multiprocessing Queue.
    """

    def __init__(
        self,
        mode: str,
        model: torch.nn.Module,
        grid_size: tuple[int, int] = (128, 128),
        port: int = 8050,
    ):
        self.mode = mode
        self.model = model
        self.grid_size = grid_size
        self.port = port

        # Import here to avoid circular imports
        from goodharts.utils.brain_viz import BrainVisualizer
        self.visualizer = BrainVisualizer(model)
        self.layer_names = self.visualizer.get_displayable_layers()

        # Process isolation setup
        ctx = mp.get_context('spawn')
        self._queue: MPQueue = ctx.Queue(maxsize=100)
        self._stop_event = ctx.Event()
        self._process: Optional[Process] = None

    def start(self):
        """Start dashboard in separate process."""
        ctx = mp.get_context('spawn')
        self._process = ctx.Process(
            target=_brain_view_worker,
            args=(self.mode, self.layer_names, self.grid_size,
                  self._queue, self._stop_event, self.port),
            daemon=False
        )
        self._process.start()
        print(f"[BrainView] Started (PID {self._process.pid})")

    def update(
        self,
        grid: torch.Tensor,
        agent_x: int,
        agent_y: int,
        agent_energy: float,
        obs: torch.Tensor,
        step_count: int,
    ):
        """
        Send state update to dashboard.

        Args:
            grid: (H, W) grid tensor with CellType values
            agent_x, agent_y: Agent position
            agent_energy: Current energy
            obs: (C, H, W) observation tensor
            step_count: Current simulation step
        """
        # Run forward pass to capture activations
        device = next(self.model.parameters()).device
        obs_tensor = obs.unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Collect activations
        activations = {}
        for name in self.layer_names:
            display = self.visualizer.get_activation_display(name)
            if display is not None:
                activations[name] = display

        # Build state object
        state = BrainViewState(
            grid_rgb=grid_with_agent_to_rgb(grid, agent_x, agent_y),
            agent_x=agent_x,
            agent_y=agent_y,
            agent_energy=agent_energy,
            local_view_rgb=observation_to_rgb(obs, self.mode),
            layer_activations=activations,
            action_probs=probs,
            step_count=step_count,
            mode=self.mode,
        )

        # Send to dashboard process
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass  # Drop if queue full - dashboard will catch up

    def stop(self, timeout: float = 2.0):
        """Stop dashboard gracefully."""
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()

    def wait(self):
        """Wait for dashboard to close (user closes browser)."""
        if self._process:
            self._process.join()

    def is_running(self) -> bool:
        """Check if dashboard process is still running."""
        return self._process is not None and self._process.is_alive()


def _brain_view_worker(
    mode: str,
    layer_names: list[str],
    grid_size: tuple[int, int],
    update_queue: MPQueue,
    stop_event,
    port: int,
):
    """
    Worker function for brain view process.

    Runs in separate process, polls queue for updates, serves Dash app.
    """
    dashboard = BrainViewDashboard(mode, layer_names, grid_size)

    # Background thread to poll queue and update dashboard
    def poll_queue():
        while not stop_event.is_set():
            try:
                state = update_queue.get(timeout=0.02)
                dashboard.send_state(state)
            except queue.Empty:
                pass
            except Exception:
                pass

    poll_thread = threading.Thread(target=poll_queue, daemon=True)
    poll_thread.start()

    try:
        dashboard.run(port=port)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


def create_brain_view_app(
    mode: str,
    model: torch.nn.Module,
    grid_size: tuple[int, int] = (128, 128),
    port: int = 8050,
) -> BrainViewApp:
    """
    Factory function for brain view application.

    Args:
        mode: Training mode name
        model: Neural network model to visualize
        grid_size: Grid dimensions (H, W)
        port: Dashboard server port

    Returns:
        BrainViewApp instance (call .start() to launch)
    """
    return BrainViewApp(mode, model, grid_size, port)
