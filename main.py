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
    parser.add_argument('--speed', type=int, default=50,
                        help='Animation interval in ms (default: 50, lower=faster)')
    parser.add_argument('--agents', type=int, default=5,
                        help='Number of each agent type (default: 5)')
    return parser.parse_args()


def setup_config(args):
    """Configure simulation based on command-line args."""
    config = get_config()
    
    if args.learned:
        # Use learned behaviors
        config['AGENTS_SETUP'] = [
            {'behavior_class': 'LearnedGroundTruth', 'count': args.agents},
            {'behavior_class': 'LearnedProxy', 'count': args.agents}
            {'behavior_class': 'LearnedProxyIllAdjusted', 'count': args.agents}
        ]
        print("🧠 Using LEARNED agents (CNN-based)")
        print("   Ground-truth trained: cyan | Proxy trained: magenta")
    else:
        # Use hardcoded behaviors
        config['AGENTS_SETUP'] = [
            {'behavior_class': 'OmniscientSeeker', 'count': args.agents},
            {'behavior_class': 'ProxySeeker', 'count': args.agents}
        ]
        print("📐 Using HARDCODED agents (heuristic-based)")
        print("   OmniscientSeeker: cyan | ProxySeeker: magenta")
    
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
    
    viz = create_standard_layout(sim)
    
    ani = animation.FuncAnimation(
        viz['fig'], 
        lambda f: update_frame(f, sim, viz, args),
        interval=args.speed, 
        blit=False,
        cache_frame_data=False
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.show()


if __name__ == "__main__":
    main()
