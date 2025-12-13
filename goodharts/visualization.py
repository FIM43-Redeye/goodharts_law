"""
Visualization module for Goodhart's Law simulation.

Provides matplotlib-based visualization layouts and update functions.
Separated from main.py for cleaner organization.
"""
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import numpy as np

from goodharts.configs.default_config import CellType


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
        behavior_name = type(agent.behavior).__name__
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
    - Top-right: Energy over time
    - Bottom-left: Activity heatmap
    - Bottom-right: Death statistics
    
    Returns:
        Dict with figure, axes, and image/line objects for updates.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Goodhart's Law Simulation", fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax_sim = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Simulation view - use RGB rendering
    grid = sim.get_render_grid()
    agent_positions = sim.get_agent_positions()
    rgb_grid = render_with_agents(grid, agent_positions, sim.world.width, sim.world.height)
    img_sim = ax_sim.imshow(rgb_grid)
    ax_sim.set_title("World View")
    ax_sim.axis('off')
    
    # Legend for agent types
    legend_elements = build_legend_elements(sim)
    ax_sim.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Energy plot
    ax_energy.set_title("Average Energy Over Time")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Energy")
    line_gt, = ax_energy.plot([], [], label='Ground-Truth', color='#00d9ff', linewidth=2)
    line_proxy, = ax_energy.plot([], [], label='Proxy', color='#ff00ff', linewidth=2)
    ax_energy.legend(loc='upper right')
    ax_energy.set_xlim(0, 100)
    ax_energy.set_ylim(0, 100)
    ax_energy.grid(True, alpha=0.3)
    
    # Heatmap
    img_heatmap = ax_heatmap.imshow(sim.stats['heatmap'], cmap='hot', interpolation='nearest')
    ax_heatmap.set_title("Activity Heatmap")
    ax_heatmap.axis('off')
    plt.colorbar(img_heatmap, ax=ax_heatmap, fraction=0.046)
    
    # Death stats
    death_reasons = ['Starvation', 'Poison']
    bar_colors = ['#4a4a4a', '#ff6b6b']
    bars = ax_stats.bar(death_reasons, [0, 0], color=bar_colors, edgecolor='white')
    ax_stats.set_title("Cause of Death")
    ax_stats.set_ylim(0, 10)
    ax_stats.set_ylabel("Count")
    
    return {
        'fig': fig,
        'ax_sim': ax_sim, 'img_sim': img_sim,
        'ax_energy': ax_energy, 'line_gt': line_gt, 'line_proxy': line_proxy,
        'ax_heatmap': ax_heatmap, 'img_heatmap': img_heatmap,
        'ax_stats': ax_stats, 'bars': bars,
        'history_gt': [], 'history_proxy': []
    }


def create_brain_layout(sim, agent, visualizer):
    """
    Create brain visualization layout - adapts to network architecture.
    
    Layout:
    - Left column: World view (zoomed), Agent observation
    - Right columns: One panel per discovered layer + Action probs
    
    Returns:
        Dict with figure, axes, panels, and objects for updates.
    """
    from goodharts.utils.brain_viz import get_action_label
    
    layers = visualizer.get_displayable_layers()
    n_layers = len(layers)
    
    # Dynamic grid: 2 rows, columns = 2 + ceil(layers/2)
    n_cols = 2 + (n_layers + 1) // 2  # +1 for action probs
    n_rows = 2
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle("🧠 Brain View - Neural Network Visualization", fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    # Left column - World and Observation
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[1, 0])
    
    # World view (zoomed on agent) - use RGB rendering
    cmap_sim = build_colormap()
    img_world = ax_world.imshow(np.zeros((21, 21, 3), dtype=np.uint8))
    ax_world.set_title("World View (Zoomed)")
    ax_world.axis('off')
    
    # Observation (what agent sees)
    obs = agent.get_local_view(mode='ground_truth')
    if obs.ndim == 3:
        obs_display = obs.sum(axis=0)
    else:
        obs_display = obs
    img_obs = ax_obs.imshow(obs_display, cmap='viridis')
    ax_obs.set_title("Agent Observation")
    ax_obs.axis('off')
    
    # Layer panels - dynamically placed
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
    
    # Action probabilities panel
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
        'ax_world': ax_world, 'img_world': img_world,
        'ax_obs': ax_obs, 'img_obs': img_obs,
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
    
    # Get zoomed world view centered on agent
    view_range = 10
    x, y = agent.x, agent.y
    
    # Extract zoomed region and render with agents
    grid = sim.get_render_grid()
    agent_positions = sim.get_agent_positions()
    h, w = grid.shape
    
    x1 = max(0, x - view_range)
    x2 = min(w, x + view_range + 1)
    y1 = max(0, y - view_range)
    y2 = min(h, y + view_range + 1)
    
    # Render zoomed region with agents
    zoomed_grid = grid[y1:y2, x1:x2]
    zoomed_positions = [(ax - x1, ay - y1, color) 
                        for ax, ay, color in agent_positions 
                        if x1 <= ax < x2 and y1 <= ay < y2]
    zoomed_rgb = render_with_agents(zoomed_grid, zoomed_positions, x2 - x1, y2 - y1)
    
    viz['img_world'].set_data(zoomed_rgb)
    viz['ax_world'].set_title(f"World ({agent.x}, {agent.y}) E={agent.energy:.0f}")
    
    # Update observation
    obs = agent.get_local_view(mode='ground_truth')
    if obs.ndim == 3:
        obs_display = obs.sum(axis=0)
    else:
        obs_display = obs
    viz['img_obs'].set_data(obs_display)
    viz['img_obs'].set_clim(vmin=obs_display.min(), vmax=obs_display.max() or 1)
    
    # Forward pass through model to trigger hooks
    brain = agent.behavior.get_brain() if hasattr(agent.behavior, 'get_brain') else None
    if brain:
        device = next(brain.parameters()).device
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
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
        print(f"\n📊 Brain view complete after {sim.step_count} steps")
        plt.close(viz['fig'])
    
    return []


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
    
    # Update simulation view with RGB rendering
    grid = sim.get_render_grid()
    agent_positions = sim.get_agent_positions()
    rgb_grid = render_with_agents(grid, agent_positions, sim.world.width, sim.world.height)
    viz['img_sim'].set_data(rgb_grid)
    
    # Update energy plot
    gt_agents = [a for a in sim.agents if 'Omniscient' in a.behavior.__class__.__name__ 
                 or 'GroundTruth' in a.behavior.__class__.__name__]
    proxy_agents = [a for a in sim.agents if 'Proxy' in a.behavior.__class__.__name__]
    
    avg_gt = np.mean([a.energy for a in gt_agents]) if gt_agents else 0
    avg_proxy = np.mean([a.energy for a in proxy_agents]) if proxy_agents else 0
    
    viz['history_gt'].append(avg_gt)
    viz['history_proxy'].append(avg_proxy)
    
    viz['line_gt'].set_data(range(len(viz['history_gt'])), viz['history_gt'])
    viz['line_proxy'].set_data(range(len(viz['history_proxy'])), viz['history_proxy'])
    
    viz['ax_energy'].set_xlim(0, max(100, sim.step_count + 10))
    max_energy = max(viz['history_gt'] + viz['history_proxy'] + [1])
    viz['ax_energy'].set_ylim(0, max(100, max_energy * 1.1))
    
    # Update heatmap
    viz['img_heatmap'].set_data(sim.stats['heatmap'])
    hmap_max = np.max(sim.stats['heatmap'])
    if hmap_max > 0:
        viz['img_heatmap'].set_clim(vmax=hmap_max)
    
    # Update death stats
    deaths = sim.stats['deaths']
    starved = sum(1 for d in deaths if d['reason'] == 'Starvation')
    poisoned = sum(1 for d in deaths if d['reason'] == 'Poison')
    
    for bar, h in zip(viz['bars'], [starved, poisoned]):
        bar.set_height(h)
    
    max_deaths = max(starved, poisoned, 10)
    viz['ax_stats'].set_ylim(0, max_deaths + 2)
    
    # Update title with stats
    alive_gt = len(gt_agents)
    alive_proxy = len(proxy_agents)
    
    viz['ax_stats'].set_title(
        f"Deaths (Step {sim.step_count}) | "
        f"Alive: GT={alive_gt}, Proxy={alive_proxy}"
    )
    
    # Check if everyone died
    if len(sim.agents) == 0 and not getattr(viz, 'death_announced', False):
        viz['death_announced'] = True
        print("\n" + "="*60)
        print("  All agents are DEAD. You'll have to check the leaderboard yourself.")
        print("="*60)
        print(f"\n  Final step: {sim.step_count}")
        print(f"  Deaths by starvation: {starved}")
        print(f"  Deaths by poison: {poisoned}")
        print("\n  Closing in 5 seconds...")
        print("="*60 + "\n")
        
        # Schedule window close after 5 seconds
        import threading
        def close_window():
            import time
            time.sleep(5)
            plt.close(viz['fig'])
        threading.Thread(target=close_window, daemon=True).start()
    
    # Check if we should stop (manual step limit)
    if args.steps and sim.step_count >= args.steps:
        print(f"\n📊 Simulation complete after {sim.step_count} steps")
        print(f"   Survivors: {len(sim.agents)} ({alive_gt} GT, {alive_proxy} Proxy)")
        print(f"   Deaths: {starved} starvation, {poisoned} poison")
        plt.close(viz['fig'])
    
    return [viz['img_sim'], viz['line_gt'], viz['line_proxy'], 
            viz['img_heatmap'], *viz['bars']]
