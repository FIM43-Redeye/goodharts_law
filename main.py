#!/usr/bin/env python3
"""
Goodhart's Law Simulation - Visual Demo

Run with: python main.py [options]

Examples:
    python main.py                    # Default: all trained agents (shows Goodhart's Law)
    python main.py --baseline         # Baseline: OmniscientSeeker vs ProxySeeker
    python main.py -a ground_truth    # Single agent type
    python main.py --brain-view -a proxy  # Neural network visualization
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from goodharts.simulation import Simulation
from goodharts.configs.default_config import get_simulation_config
from goodharts.utils.logging_config import setup_logging
from goodharts.visualization import (
    create_standard_layout,
    create_brain_layout,
    update_frame,
    update_brain_frame
)
from goodharts.behaviors.learned import LEARNED_PRESETS


def parse_args():
    # Get available preset names for help text
    preset_names = list(LEARNED_PRESETS.keys())
    all_agent_choices = preset_names + ['OmniscientSeeker', 'ProxySeeker']

    parser = argparse.ArgumentParser(
        description="Goodhart's Law Simulation - Visual Demo",
        epilog='Default: Shows all trained agents demonstrating Goodhart\'s Law',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Agent selection
    agents_group = parser.add_argument_group('Agent Selection')
    agents_group.add_argument('-a', '--agent', type=str, default=None, nargs='+',
                              metavar='TYPE',
                              help=f'Specific agent(s): {", ".join(all_agent_choices)}')
    agents_group.add_argument('--baseline', action='store_true',
                              help='Show only baseline agents (OmniscientSeeker vs ProxySeeker)')
    agents_group.add_argument('--all', action='store_true', dest='show_all',
                              help='Show all agents (baseline + learned)')
    agents_group.add_argument('-n', '--count', type=int, default=5, metavar='N',
                              help='Number of each agent type [default: 5]')

    # Visualization modes
    viz_group = parser.add_argument_group('Visualization')
    viz_group.add_argument('--brain-view', action='store_true',
                           help='Focus on single agent with neural network visualization')
    viz_group.add_argument('--show-observations', action='store_true',
                           help='Show observation panels (what agents see)')
    viz_group.add_argument('--speed', type=int, default=None, metavar='MS',
                           help='Animation interval in ms [default: 50, lower=faster]')
    viz_group.add_argument('--steps', type=int, default=None, metavar='N',
                           help='Run for N steps then exit [default: forever]')

    # World settings
    world_group = parser.add_argument_group('World')
    world_group.add_argument('--food', type=int, default=None, metavar='N',
                             help='Override food count')
    world_group.add_argument('--poison', type=int, default=None, metavar='N',
                             help='Override poison count')

    # Advanced
    advanced_group = parser.add_argument_group('Advanced')
    advanced_group.add_argument('--config', type=str, default=None, metavar='PATH',
                                help='Custom TOML config file')
    advanced_group.add_argument('--model', type=str, default=None, metavar='PATH',
                                help='Custom model weights (for --brain-view)')

    return parser.parse_args()


def setup_config(args):
    """
    Load config and apply CLI overrides.

    Agent selection priority:
    1. --brain-view with -a: Single agent for neural network visualization
    2. -a/--agent: Specific agent(s) by name
    3. --baseline: Hardcoded agents only (OmniscientSeeker vs ProxySeeker)
    4. --all: All agents (baseline + learned)
    5. Default: All learned agents (demonstrates Goodhart's Law)
    """
    from goodharts.config import (
        load_config, get_config as get_toml_config,
        get_brain_view_config, get_visualization_config
    )
    from goodharts.behaviors import list_behavior_names

    # Load TOML config
    if args.config:
        load_config(args.config)
        print(f"Using config: {args.config}")

    # Get runtime config
    config = get_simulation_config(args.config)

    # Get visualization settings from TOML
    viz_cfg = get_visualization_config()
    if args.speed is None:
        args.speed = viz_cfg.get('speed', 50)
    if args.steps is None:
        steps = viz_cfg.get('steps', 0)
        args.steps = steps if steps > 0 else None

    # Check brain_view from TOML config
    bv_cfg = get_brain_view_config()
    if bv_cfg.get('enabled', False) and not args.brain_view:
        args.brain_view = True
        if args.agent is None:
            agent_type = bv_cfg.get('agent_type')
            if agent_type:
                args.agent = [agent_type]
        if args.model is None and bv_cfg.get('model'):
            args.model = bv_cfg['model']

    # CLI overrides for world
    if args.food is not None:
        config['GRID_FOOD_INIT'] = args.food
    if args.poison is not None:
        config['GRID_POISON_INIT'] = args.poison

    # Determine agent setup based on args
    count = args.count

    if args.brain_view:
        # Brain view: single agent with neural network visualization
        if not args.agent:
            print("ERROR: --brain-view requires -a/--agent")
            print("  Example: python main.py --brain-view -a ground_truth")
            import sys
            sys.exit(1)

        agent_name = args.agent[0]  # Only first agent for brain view
        agent_cfg = {'behavior_class': agent_name, 'count': 1}
        if args.model:
            agent_cfg['model_path'] = args.model
        config['AGENTS_SETUP'] = [agent_cfg]

        # Freeze energy in brain view by default
        freeze_energy = bv_cfg.get('freeze_energy', True)
        config['FREEZE_ENERGY'] = freeze_energy

        status = [f"BRAIN VIEW: {agent_name}"]
        if args.model:
            status.append(f"model={args.model}")
        if freeze_energy:
            status.append("energy=frozen")
        print(" | ".join(status))

    elif args.agent:
        # Specific agent(s) requested
        config['AGENTS_SETUP'] = [
            {'behavior_class': name, 'count': count}
            for name in args.agent
        ]
        print(f"Agents: {', '.join(args.agent)} (x{count} each)")

    elif args.baseline:
        # Baseline: hardcoded agents only
        config['AGENTS_SETUP'] = [
            {'behavior_class': 'OmniscientSeeker', 'count': count},
            {'behavior_class': 'ProxySeeker', 'count': count},
        ]
        print(f"Baseline agents: OmniscientSeeker vs ProxySeeker (x{count} each)")

    elif args.show_all:
        # All agents: baseline + learned
        skip_list = {'BehaviorStrategy', 'LearnedBehavior', 'Agent'}
        config['AGENTS_SETUP'] = []

        # Hardcoded behaviors
        for name in list_behavior_names():
            if name not in skip_list and name not in LEARNED_PRESETS:
                config['AGENTS_SETUP'].append({'behavior_class': name, 'count': count})

        # Learned presets
        for preset_name in LEARNED_PRESETS:
            config['AGENTS_SETUP'].append({'behavior_class': preset_name, 'count': count})

        print(f"All agents: {len(config['AGENTS_SETUP'])} types (x{count} each)")

    else:
        # Default: all learned agents (demonstrates Goodhart's Law)
        config['AGENTS_SETUP'] = [
            {'behavior_class': preset_name, 'count': count}
            for preset_name in LEARNED_PRESETS
        ]
        print(f"Learned agents: {', '.join(LEARNED_PRESETS.keys())} (x{count} each)")

    return config


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
    
    print(f"\n[World] {config['GRID_WIDTH']}x{config['GRID_HEIGHT']}")
    print(f"[Food] {config['GRID_FOOD_INIT']} | [Poison] {config['GRID_POISON_INIT']}")
    print(f"[View] range: {config['AGENT_VIEW_RANGE']}")
    print(f"\n[Start] Starting simulation... (close window to exit)\n")
    
    if args.brain_view:
        # Brain view mode - visualize neural network internals
        from goodharts.utils.brain_viz import BrainVisualizer
        
        # Find the first learned agent (has get_brain method)
        learned_agents = [a for a in sim.agents if hasattr(a.behavior, 'get_brain')]
        if not learned_agents:
            print("[ERROR] No learned agents found! Use --learned flag.")
            return
        
        agent = learned_agents[0]
        
        # Force lazy initialization of brain by getting a view first
        obs = agent.get_local_view(mode='ground_truth')
        _ = agent.behavior.decide_action(agent, obs)  # Triggers brain init
        
        model = agent.behavior.get_brain()
        if model is None:
            print("[ERROR] Brain failed to initialize!")
            return
        
        # Create visualizer
        visualizer = BrainVisualizer(model)
        print(f"[Layers] Discovered: {visualizer.get_displayable_layers()}")
        
        # Do one forward pass to initialization activations
        import torch
        device = next(model.parameters()).device
        
        # Obs is already a tensor (C, H, W)
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().unsqueeze(0).to(device)
        else:
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
    
    # Note: tight_layout() conflicts with manually positioned axes (RadioButtons),
    # so we skip it here - layouts are already configured in visualization.py
    plt.show()


if __name__ == "__main__":
    main()
