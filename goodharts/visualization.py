"""
Visualization module for Goodhart's Law simulation.

Provides matplotlib-based visualization layouts and update functions.
Separated from main.py for cleaner organization.
"""
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.widgets import RadioButtons
import numpy as np
import torch

from goodharts.behaviors.utils import get_behavior_name
from goodharts.configs.default_config import CellType


def _to_numpy(data) -> np.ndarray:
    """
    Convert data to numpy array, handling both tensors and arrays.

    Args:
        data: A torch.Tensor or numpy array

    Returns:
        numpy array on CPU
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data


def build_colormap() -> colors.ListedColormap:
    """
    Build colormap from CellType colors.
    
    Returns:
        ListedColormap with colors for each CellType in value order.
    """
    cell_colors = ['#%02x%02x%02x' % ct.color for ct in CellType.all_types()]
    return colors.ListedColormap(cell_colors)


def render_with_agents(grid: np.ndarray, agent_positions: list, 
                       width: int, height: int) -> np.ndarray:
    """
    Composite grid with agent overlay as RGB image.
    
    Args:
        grid: Environmental grid (CellType values)
        agent_positions: List of (x, y, rgb_color) from sim.get_agent_positions()
        width, height: Grid dimensions
        
    Returns:
        RGB array (H, W, 3) suitable for imshow
    """
    # Start with grid colors
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            cell_type = CellType.by_value(grid[y, x])
            if cell_type:
                rgb[y, x] = cell_type.color
            else:
                rgb[y, x] = (128, 128, 128)  # Unknown = gray
    
    # Overlay agents
    for x, y, color in agent_positions:
        if 0 <= x < width and 0 <= y < height:
            rgb[y, x] = color
    
    return rgb


def build_legend_elements(sim) -> list[Patch]:
    """
    Build legend patches for agent types in the simulation.
    
    Returns color patches for environmental elements plus each unique
    behavior type currently in the simulation.
    """
    elements = [
        Patch(facecolor='#%02x%02x%02x' % CellType.FOOD.color, label='Food'),
        Patch(facecolor='#%02x%02x%02x' % CellType.POISON.color, label='Poison'),
    ]
    
    # Add unique behavior types from current agents
    seen_behaviors = set()
    for agent in sim.agents:
        behavior_name = get_behavior_name(agent.behavior)
        if behavior_name not in seen_behaviors:
            seen_behaviors.add(behavior_name)
            color = '#%02x%02x%02x' % agent.behavior.color
            elements.append(Patch(facecolor=color, label=behavior_name))
    
    return elements


def create_standard_layout(sim):
    """
    Create the standard 2x2 visualization layout.
    
    Layout:
    - Top-left: World view with agents
    - Top-right: Energy over time (Dynamic lines per behavior)
    - Bottom-left: Activity heatmap (With RadioButtons)
    - Bottom-right: Death statistics (Stacked bars)
    
    Returns:
    Dict with figure, axes, and image/line objects for updates.
    """
    # Create figure with extra space on left for radio buttons
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Goodhart's Law Simulation", fontsize=14, fontweight='bold')
    
    # Grid: 2 rows, 2 columns.
    # We'll use a manually placed axes for radio buttons later
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, left=0.15)
    
    ax_sim = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # 1. Simulation view
    grid = sim.get_render_grid()
    agent_positions = sim.get_agent_positions()
    rgb_grid = render_with_agents(grid, agent_positions, sim.world.width, sim.world.height)
    img_sim = ax_sim.imshow(rgb_grid)
    ax_sim.set_title("World View")
    ax_sim.axis('off')
    
    # Legend
    legend_elements = build_legend_elements(sim)
    ax_sim.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # 2. Energy plot (Dynamic lines)
    ax_energy.set_title("Average Energy Over Time")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_xlim(0, 100)
    ax_energy.set_ylim(0, 100)
    ax_energy.grid(True, alpha=0.3)
    
    # Identify all potential behaviors from current agents
    behavior_names = sorted(list(set(
        get_behavior_name(a.behavior) for a in sim.agents
    )))
    
    energy_lines = {}
    energy_histories = {}
    
    # Create a line for each behavior type
    colors_cycle = ['#00d9ff', '#ff00ff', '#8a2be2', '#ffa500', '#00ff00', '#ffff00']
    for i, b_name in enumerate(behavior_names):
        color = colors_cycle[i % len(colors_cycle)]
        line, = ax_energy.plot([], [], label=b_name, color=color, linewidth=2)
        energy_lines[b_name] = line
        energy_histories[b_name] = []
        
    ax_energy.legend(loc='upper right', fontsize=8)
    
    # 3. Heatmap
    # Initialize with 'all' - convert Torch tensor to NumPy for matplotlib
    heatmap_data = _to_numpy(sim.stats['heatmap']['all'])
    img_heatmap = ax_heatmap.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    ax_heatmap.set_title("Activity Heatmap (All)")
    ax_heatmap.axis('off')
    plt.colorbar(img_heatmap, ax=ax_heatmap, fraction=0.046)
    
    # Radio Buttons for Heatmap Selection
    # Place in left margin
    ax_radio = fig.add_axes([0.02, 0.4, 0.10, 0.2])  # [left, bottom, width, height]
    ax_radio.set_title("Heatmap Source", fontsize=9)
    # Options: 'All' + individual behaviors
    radio_options = ['all'] + behavior_names
    radio = RadioButtons(ax_radio, radio_options)
    
    # 4. Death stats (Stacked Bars)
    ax_stats.set_title("Cause of Death by Type")
    ax_stats.set_ylabel("Count")
    
    # X-axis = behavior names
    x_pos = np.arange(len(behavior_names))
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels(behavior_names, rotation=45, ha='right', fontsize=8)
    
    # Initial empty bars (Starvation usually bottom, Poison top)
    # We keep references to bar containers to update height later
    bar_width = 0.5
    bars_starved = ax_stats.bar(x_pos, [0]*len(behavior_names), width=bar_width, 
                                label='Starvation', color='#4a4a4a', edgecolor='white')
    bars_poisoned = ax_stats.bar(x_pos, [0]*len(behavior_names), width=bar_width, 
                                 bottom=[0]*len(behavior_names),
                                 label='Poison', color='#ff6b6b', edgecolor='white')
    ax_stats.legend(loc='upper right', fontsize=8)
    ax_stats.set_ylim(0, 10)

    # Callback for radio button (must store ref to prevent garbage collection)
    def change_heatmap(label):
        # We'll handle the actual data update in update_frame by checking radio.value_selected
        # But we can update title here immediately if we want
        ax_heatmap.set_title(f"Activity Heatmap ({label})")
        
    radio.on_clicked(change_heatmap)

    return {
        'fig': fig,
        'ax_sim': ax_sim, 'img_sim': img_sim,
        'ax_energy': ax_energy, 'energy_lines': energy_lines, 'energy_histories': energy_histories,
        'ax_heatmap': ax_heatmap, 'img_heatmap': img_heatmap,
        'radio': radio, 'radio_ax': ax_radio,
        'ax_stats': ax_stats, 'bars_starved': bars_starved, 'bars_poisoned': bars_poisoned,
        'behavior_names': behavior_names,
    }


def create_brain_layout(sim, agent, visualizer):
    """
    Create brain visualization layout - adapts to network architecture.
    
    Layout:
    - Left column: Agent Observation (RGB Reconstructed)
    - Right columns: One panel per discovered layer + Action probs
    
    Returns:
        Dict with figure, axes, panels, and objects for updates.
    """
    from goodharts.utils.brain_viz import get_action_label
    
    layers = visualizer.get_displayable_layers()
    n_layers = len(layers)
    
    # Dynamic grid: 2 rows, columns = 2 + ceil(layers/2)
    # Col 0 is Main View (Obs), Col 1..N are layers
    n_cols = 1 + (n_layers + 1) // 2
    if n_cols < 2: n_cols = 2 # Ensure at least 2 cols
    n_rows = 2
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle("Brain View - Neural Network Visualization", fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    # 1. Main View (Left column, spanning both rows)
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Placeholder for RGB Observation
    # Size depends on view range, will act as placeholder
    img_main = ax_main.imshow(np.zeros((21, 21, 3), dtype=np.uint8))
    ax_main.set_title("Agent Observation")
    ax_main.axis('off')
    
    # 2. Layer panels - dynamically placed
    layer_panels = {}
    for i, layer_name in enumerate(layers):
        row = i % 2
        col = 1 + i // 2
        ax = fig.add_subplot(gs[row, col])
        
        layer_type = visualizer.get_layer_type(layer_name)
        short_name = layer_name.split('.')[-1]  # Last part of name
        ax.set_title(f"{short_name} ({layer_type})")
        ax.axis('off')
        
        # Placeholder image
        img = ax.imshow(np.zeros((4, 4)), cmap='hot')
        layer_panels[layer_name] = {'ax': ax, 'img': img, 'type': layer_type}
    
    # 3. Action probabilities panel
    ax_actions = fig.add_subplot(gs[:, -1])  # Right-most column, span rows
    n_actions = 8  # Will be updated dynamically
    bars_actions = ax_actions.barh(range(n_actions), np.zeros(n_actions), color='#00d9ff')
    ax_actions.set_xlim(0, 1)
    ax_actions.set_yticks(range(n_actions))
    ax_actions.set_yticklabels([get_action_label(i, n_actions) for i in range(n_actions)])
    ax_actions.set_title("Action Probabilities")
    ax_actions.set_xlabel("Probability")
    
    return {
        'fig': fig,
        'ax_main': ax_main, 'img_main': img_main,
        'layer_panels': layer_panels,
        'ax_actions': ax_actions, 'bars_actions': bars_actions,
        'agent': agent,
        'visualizer': visualizer,
    }


def update_brain_frame(frame, sim, viz, args):
    """
    Update brain visualization for one frame.
    
    Args:
        frame: Frame number (from FuncAnimation)
        sim: Simulation instance
        viz: Visualization state dict from create_brain_layout
        args: Parsed CLI arguments
        
    Returns:
        List of updated artists (for blitting)
    """
    import torch
    import torch.nn.functional as F
    
    agent = viz['agent']
    visualizer = viz['visualizer']
    
    # Check if agent died - find new one
    if not agent.alive:
        learned_agents = [a for a in sim.agents if hasattr(a.behavior, 'get_brain')]
        if learned_agents:
            viz['agent'] = learned_agents[0]
            agent = viz['agent']
            print(f"Switched to new agent at ({agent.x}, {agent.y})")
        else:
            viz['fig'].suptitle("All agents dead!", fontsize=14, fontweight='bold')
            return []
    
    # Step simulation
    sim.step()
    
    # -------------------------------------------------------------------------
    # 1. Get Agent Observation & Reconstruct RGB
    # -------------------------------------------------------------------------
    # In vectorized sim, get_local_view returns ndarray (C, H, W)
    # We use agent.get_local_view('ground_truth') to ensure we get all channels
    # But wait, VecEnv outputs one specific spec.
    # We assume VecEnv uses ground_truth spec (or superset).
    obs = agent.get_local_view(mode='ground_truth') # returns (C, H, W)
    
    # We need to know which channel is which
    channel_names = sim.vec_env.channel_names
    
    # Convert to CPU/Numpy for RGBA construction
    if isinstance(obs, torch.Tensor):
        obs_cpu = obs.cpu().numpy()
    else:
        obs_cpu = obs
    
    # Initialize RGB image (White background for empty? No, black typically)
    c, h, w = obs_cpu.shape
    rgb_img = np.full((h, w, 3), 0, dtype=np.uint8) 
    
    # Dynamic Legend tracking
    legend_elements = []
    seen_types = set()
    
    # Iterate CellTypes to overlay colors
    CellType = sim.config['CellType']
    render_order = [CellType.WALL, CellType.FOOD, CellType.POISON, CellType.PREDATOR, CellType.PREY]
    
    for c_type in render_order:
        # Find explicit channel index for this type
        # Naming convention provided by vec_env: "cell_{name}"
        channel_name = f"cell_{c_type.name.lower()}"
        
        if channel_name in channel_names:
            idx = channel_names.index(channel_name)
            # Use CPU observation for masking
            mask = obs_cpu[idx] > 0.5
            if mask.any():
                rgb_img[mask] = c_type.color
                
                if c_type.name not in seen_types:
                    seen_types.add(c_type.name)
                    legend_elements.append(Patch(facecolor='#%02x%02x%02x' % c_type.color, label=f"{c_type.name}"))
                    
    # Paint Self (Center)
    cx, cy = h // 2, w // 2
    rgb_img[cx, cy] = agent.behavior.color
    legend_elements.append(Patch(facecolor='#%02x%02x%02x' % agent.behavior.color, label="Self"))
    
    # Update Main View
    viz['img_main'].set_data(rgb_img)
    viz['ax_main'].set_title(f"Agent View (E={agent.energy:.0f})")
    
    # Update Legend
    viz['ax_main'].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # -------------------------------------------------------------------------
    # 2. Network Inference
    # -------------------------------------------------------------------------
    # obs is already the right input for network (C, H, W)


    
    # Forward pass through model to trigger hooks
    brain = agent.behavior.get_brain() if hasattr(agent.behavior, 'get_brain') else None
    if brain:
        first_param = next(brain.parameters())
        device = first_param.device
        dtype = first_param.dtype
        
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(device=device, dtype=dtype).unsqueeze(0)
        else:
            obs_tensor = torch.from_numpy(obs).to(device=device, dtype=dtype).unsqueeze(0)
        
        with torch.no_grad():
            logits = brain(obs_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Update layer panels
        for layer_name, panel in viz['layer_panels'].items():
            display = visualizer.get_activation_display(layer_name)
            if display is not None:
                panel['img'].set_data(display)
                panel['img'].set_clim(vmin=display.min(), vmax=display.max() or 1)
        
        # Update action probabilities
        for bar, prob in zip(viz['bars_actions'], probs):
            bar.set_width(prob)
        
        # Highlight chosen action
        chosen = probs.argmax()
        for i, bar in enumerate(viz['bars_actions']):
            bar.set_color('#ff6b6b' if i == chosen else '#00d9ff')
    
    # Check if we should stop (manual step limit)
    if args.steps and sim.step_count >= args.steps:
        print(f"\nBrain view complete after {sim.step_count} steps")
        plt.close(viz['fig'])
    
    return []


def _update_grid_view(sim, viz) -> None:
    """Update the simulation grid view with agent positions."""
    grid = sim.get_render_grid()
    agent_positions = sim.get_agent_positions()
    rgb_grid = render_with_agents(grid, agent_positions, sim.world.width, sim.world.height)
    viz['img_sim'].set_data(rgb_grid)


def _update_energy_plot(sim, viz) -> None:
    """Update the energy plot with current agent energies by behavior type."""
    for b_name, line in viz['energy_lines'].items():
        agents_of_type = [
            a for a in sim.agents
            if get_behavior_name(a.behavior) == b_name
        ]
        avg_energy = np.mean([a.energy for a in agents_of_type]) if agents_of_type else 0
        viz['energy_histories'][b_name].append(avg_energy)
        line.set_data(range(len(viz['energy_histories'][b_name])), viz['energy_histories'][b_name])

    # Dynamic scaling
    viz['ax_energy'].set_xlim(0, max(100, sim.step_count + 10))
    all_energies = [val for hist in viz['energy_histories'].values() for val in hist]
    max_energy = max(all_energies + [1])
    viz['ax_energy'].set_ylim(0, max(100, max_energy * 1.1))


def _update_heatmap(sim, viz) -> None:
    """Update the activity heatmap based on radio button selection."""
    selected_source = viz['radio'].value_selected
    heatmap_data = sim.stats['heatmap'].get(selected_source, sim.stats['heatmap']['all'])
    heatmap_np = _to_numpy(heatmap_data)

    viz['img_heatmap'].set_data(heatmap_np)
    hmap_max = np.max(heatmap_np)
    if hmap_max > 0:
        viz['img_heatmap'].set_clim(vmax=hmap_max)


def _update_death_stats(sim, viz) -> None:
    """Update the death statistics stacked bar chart."""
    deaths = sim.stats['deaths']
    behavior_names = viz['behavior_names']

    starved_counts = []
    poisoned_counts = []

    for b_name in behavior_names:
        relevant_deaths = [d for d in deaths if d.get('behavior', 'Unknown') == b_name]
        s = sum(1 for d in relevant_deaths if d['reason'] == 'Starvation')
        p = sum(1 for d in relevant_deaths if d['reason'] == 'Poison')
        starved_counts.append(s)
        poisoned_counts.append(p)

    for bar, h in zip(viz['bars_starved'], starved_counts):
        bar.set_height(h)

    for bar, h_p, h_s in zip(viz['bars_poisoned'], poisoned_counts, starved_counts):
        bar.set_height(h_p)
        bar.set_y(h_s)

    max_deaths = max(sum(starved_counts) + sum(poisoned_counts), 10)
    viz['ax_stats'].set_ylim(0, max_deaths + 2)

    alive_count = len(sim.agents)
    viz['ax_stats'].set_title(f"Deaths (Step {sim.step_count}) | Alive: {alive_count}")


def _check_end_conditions(sim, viz, args) -> None:
    """Check if simulation should end (all dead or step limit reached)."""
    alive_count = len(sim.agents)

    if alive_count == 0 and not getattr(viz, 'death_announced', False):
        viz['death_announced'] = True
        print(f"\nAll agents dead at step {sim.step_count}. Closing in 5s...")
        import threading
        def close_window():
            import time
            time.sleep(5)
            plt.close(viz['fig'])
        threading.Thread(target=close_window, daemon=True).start()

    if args.steps and sim.step_count >= args.steps:
        print(f"\nSimulation complete after {sim.step_count} steps")
        plt.close(viz['fig'])


def update_frame(frame, sim, viz, args):
    """
    Update standard visualization for one frame.

    Args:
        frame: Frame number (from FuncAnimation)
        sim: Simulation instance
        viz: Visualization state dict from create_standard_layout
        args: Parsed CLI arguments

    Returns:
        List of updated artists (for blitting)
    """
    sim.step()

    _update_grid_view(sim, viz)
    _update_energy_plot(sim, viz)
    _update_heatmap(sim, viz)
    _update_death_stats(sim, viz)
    _check_end_conditions(sim, viz, args)

    return [viz['img_sim'], viz['img_heatmap']]
