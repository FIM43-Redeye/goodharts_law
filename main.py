#!/usr/bin/env python3
"""
Goodhart's Law Simulation - Visual Demo

Run with: python main.py [options]

Examples:
    python main.py                          # Default: OmniscientSeeker vs ProxySeeker
    python main.py --learned                # Use trained CNN agents (requires models/)
    python main.py --show-observations      # Show what agents "see"
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import numpy as np

from goodharts.simulation import Simulation
from goodharts.configs.default_config import get_config
from goodharts.utils.logging_config import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Goodhart's Law Simulation")
    parser.add_argument('--learned', action='store_true',
                        help='Use learned CNN agents instead of hardcoded (requires trained models)')
    parser.add_argument('--show-observations', action='store_true',
                        help='Show a panel of what agents observe (one-hot channels)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Run for fixed number of steps then exit (default: run forever)')
    parser.add_argument('--speed', type=int, default=None,
                        help='Animation interval in ms (default: 50, lower=faster)')
    parser.add_argument('--agents', type=int, default=None,
                        help='Number of each agent type (default: 5)')
    parser.add_argument('--brain-view', action='store_true',
                        help='Show single agent with live neural network visualization')
    parser.add_argument('--loop', action='store_true',
                        help='Enable looping world (edges wrap around, no walls)')
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to TOML config file (CLI flags override config)')
    # Brain-view specific options (CLI overrides)
    parser.add_argument('--agent', type=str, default=None,
                        choices=['LearnedGroundTruth', 'LearnedProxy', 'LearnedProxyIllAdjusted'],
                        help='Agent type for brain-view')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--food', type=int, default=None,
                        help='Override food count')
    parser.add_argument('--poison', type=int, default=None,
                        help='Override poison count')
    return parser.parse_args()


def setup_config(args):
    """
    Load config and apply CLI overrides.
    
    Config is loaded from TOML (config.toml > config.default.toml).
    CLI flags override config values for quick experiments.
    """
    from goodharts.config import load_config, get_config as get_toml_config, get_brain_view_config, get_visualization_config
    
    # Load TOML config
    if args.config:
        load_config(args.config)
        print(f"Using config: {args.config}")
    
    # Get runtime config
    config = get_config(args.config)
    toml_cfg = get_toml_config()
    
    # Get visualization settings from TOML
    viz_cfg = get_visualization_config()
    if args.speed is None:
        args.speed = viz_cfg.get('speed', 50)
    if args.steps is None:
        steps = viz_cfg.get('steps', 0)
        args.steps = steps if steps > 0 else None
    
    # Check brain_view from config
    bv_cfg = get_brain_view_config()
    if bv_cfg.get('enabled', False) and not args.brain_view:
        args.brain_view = True
        if args.agent is None:
            args.agent = bv_cfg.get('agent_type')
        if args.model is None and bv_cfg.get('model'):
            args.model = bv_cfg['model']
    
    # CLI overrides
    if args.loop:
        config['WORLD_LOOP'] = True
    if args.food is not None:
        config['GRID_FOOD_INIT'] = args.food
    if args.poison is not None:
        config['GRID_POISON_INIT'] = args.poison
    
    # Print status
    if config.get('WORLD_LOOP'):
        print("Loop mode: World wraps at edges (toroidal)")
    
    # Mode selection
    if args.brain_view:
        if args.agent is None:
            print("ERROR: brain_view requires agent_type")
            print("  Set in config: [brain_view] agent_type = \"LearnedGroundTruth\"")
            print("  Or use CLI: --brain-view --agent LearnedGroundTruth")
            import sys
            sys.exit(1)
        
        agent_cfg = {'behavior_class': args.agent, 'count': 1}
        if args.model:
            agent_cfg['model_path'] = args.model
        config['AGENTS_SETUP'] = [agent_cfg]
        print(f"BRAIN VIEW: {args.agent}" + (f" ({args.model})" if args.model else ""))
    elif args.learned:
        count = args.agents or 5
        config['AGENTS_SETUP'] = [
            {'behavior_class': 'LearnedGroundTruth', 'count': count},
            {'behavior_class': 'LearnedProxy', 'count': count},
            {'behavior_class': 'LearnedProxyIllAdjusted', 'count': count},
        ]
        print("Using LEARNED agents (CNN-based)")
    
    return config


def create_standard_layout(sim):
    """Create the standard 2x2 visualization layout."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Goodhart's Law Simulation", fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax_sim = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Simulation view
    cmap_sim = colors.ListedColormap([
        '#1a1a2e',  # Empty - dark blue
        '#4a4a4a',  # Wall - gray
        '#16c79a',  # Food - teal green
        '#ff6b6b',  # Poison - coral red
        '#00d9ff',  # Ground-truth agent - cyan
        '#ff00ff',  # Proxy agent - magenta
        '#8A2BE2',  # Proxy Ill-Adjusted agent - blue-violet
    ])
    bounds_sim = [0, 1, 2, 3, 4, 5, 6]
    norm_sim = colors.BoundaryNorm(bounds_sim, cmap_sim.N)
    img_sim = ax_sim.imshow(sim.get_render_grid(), cmap=cmap_sim, norm=norm_sim)
    ax_sim.set_title("World View")
    ax_sim.axis('off')
    
    # Legend for agent types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#16c79a', label='Food'),
        Patch(facecolor='#ff6b6b', label='Poison'),
        Patch(facecolor='#00d9ff', label='Ground-Truth Agent'),
        Patch(facecolor='#ff00ff', label='Proxy Agent'),
        Patch(facecolor='#8A2BE2', label='Proxy Ill-Adjusted Agent'),    
    ]
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
    
    # World view (zoomed on agent)
    cmap_sim = colors.ListedColormap([
        '#1a1a2e', '#4a4a4a', '#16c79a', '#ff6b6b', '#00d9ff', '#ff00ff', '#8A2BE2'
    ])
    img_world = ax_world.imshow(np.zeros((21, 21)), cmap=cmap_sim, vmin=0, vmax=6)
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
    """Update brain visualization for one frame."""
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
    config = sim.config
    view_range = 10
    x, y = agent.x, agent.y
    
    # Extract zoomed region from render grid
    grid = sim.get_render_grid()
    h, w = grid.shape
    
    x1 = max(0, x - view_range)
    x2 = min(w, x + view_range + 1)
    y1 = max(0, y - view_range)
    y2 = min(h, y + view_range + 1)
    
    zoomed = grid[y1:y2, x1:x2]
    viz['img_world'].set_data(zoomed)
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
    """Update all visualizations for one frame."""
    sim.step()
    
    # Update simulation view
    viz['img_sim'].set_data(sim.get_render_grid())
    
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


def main():
    args = parse_args()
    setup_logging()
    
    print("\n" + "="*50)
    print("  GOODHART'S LAW SIMULATION")
    print("  'When a measure becomes a target,")
    print("   it ceases to be a good measure.'")
    print("="*50 + "\n")
    
    config = setup_config(args)
    sim = Simulation(config)
    
    print(f"\n🌍 World: {config['GRID_WIDTH']}x{config['GRID_HEIGHT']}")
    print(f"🍎 Food: {config['GRID_FOOD_INIT']} | ☠️  Poison: {config['GRID_POISON_INIT']}")
    print(f"👁️  View range: {config['AGENT_VIEW_RANGE']}")
    print(f"\n▶️  Starting simulation... (close window to exit)\n")
    
    if args.brain_view:
        # Brain view mode - visualize neural network internals
        from goodharts.utils.brain_viz import BrainVisualizer
        
        # Find the first learned agent (has get_brain method)
        learned_agents = [a for a in sim.agents if hasattr(a.behavior, 'get_brain')]
        if not learned_agents:
            print("❌ No learned agents found! Use --learned flag.")
            return
        
        agent = learned_agents[0]
        
        # Force lazy initialization of brain by getting a view first
        obs = agent.get_local_view(mode='ground_truth')
        _ = agent.behavior.decide_action(agent, obs)  # Triggers brain init
        
        model = agent.behavior.get_brain()
        if model is None:
            print("❌ Brain failed to initialize!")
            return
        
        # Create visualizer
        visualizer = BrainVisualizer(model)
        print(f"🔍 Discovered layers: {visualizer.get_displayable_layers()}")
        
        # Do one forward pass to initialize activations
        import torch
        device = next(model.parameters()).device
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            model(obs_tensor)
        
        viz = create_brain_layout(sim, agent, visualizer)
        update_func = lambda f: update_brain_frame(f, sim, viz, args)
    else:
        # Standard visualization
        viz = create_standard_layout(sim)
        update_func = lambda f: update_frame(f, sim, viz, args)
    
    ani = animation.FuncAnimation(
        viz['fig'], 
        update_func,
        interval=args.speed, 
        blit=False,
        cache_frame_data=False
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.show()



if __name__ == "__main__":
    main()
