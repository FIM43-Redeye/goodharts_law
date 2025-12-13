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

from goodharts.simulation import Simulation
from goodharts.configs.default_config import get_config
from goodharts.utils.logging_config import setup_logging
from goodharts.visualization import (
    create_standard_layout, 
    create_brain_layout,
    update_frame, 
    update_brain_frame
)


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
        
        # Freeze energy in brain view by default (agent doesn't know its energy)
        freeze_energy = bv_cfg.get('freeze_energy', True)
        config['FREEZE_ENERGY'] = freeze_energy
        
        status_parts = [f"BRAIN VIEW: {args.agent}"]
        if args.model:
            status_parts.append(f"model={args.model}")
        if freeze_energy:
            status_parts.append("energy=frozen")
        print(" | ".join(status_parts))
    elif args.learned:
        count = args.agents or 5
        config['AGENTS_SETUP'] = [
            {'behavior_class': 'LearnedGroundTruth', 'count': count},
            {'behavior_class': 'LearnedProxy', 'count': count},
            {'behavior_class': 'LearnedProxyIllAdjusted', 'count': count},
        ]
        print("Using LEARNED agents (CNN-based)")
    
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
