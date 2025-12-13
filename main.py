import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from simulation import Simulation
from configs.default_config import get_config
from utils.logging_config import setup_logging
import numpy as np

# 1. Setup Simulation
setup_logging()
config = get_config()
sim = Simulation(config)

# 2. Setup Plotting Layout
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

ax_sim = fig.add_subplot(gs[0, 0])
ax_energy = fig.add_subplot(gs[0, 1])
ax_heatmap = fig.add_subplot(gs[1, 0])
ax_stats = fig.add_subplot(gs[1, 1])

# -- Simulation View Setup --
# Colors: 0=Empty, 1=Wall, 2=Food, 3=Poison, 4=Omniscient, 5=Proxy
cmap_sim = colors.ListedColormap(['black', 'gray', 'green', 'red', 'cyan', 'magenta'])
bounds_sim = [0, 1, 2, 3, 4, 5, 6]
norm_sim = colors.BoundaryNorm(bounds_sim, cmap_sim.N)
img_sim = ax_sim.imshow(sim.get_render_grid(), cmap=cmap_sim, norm=norm_sim)
ax_sim.set_title("Live Simulation")
ax_sim.grid(False)

# -- Energy Plot Setup --
ax_energy.set_title("Avg Energy per Species")
ax_energy.set_xlabel("Step")
ax_energy.set_ylabel("Energy")
line_omni, = ax_energy.plot([], [], label='Omniscient', color='cyan')
line_proxy, = ax_energy.plot([], [], label='Proxy', color='magenta')
ax_energy.legend()
ax_energy.set_xlim(0, 100)
ax_energy.set_ylim(0, 200)

# -- Heatmap Setup --
# Just a hot colormap
img_heatmap = ax_heatmap.imshow(sim.stats['heatmap'], cmap='hot', interpolation='nearest')
ax_heatmap.set_title("Activity Heatmap")
plt.colorbar(img_heatmap, ax=ax_heatmap)

# -- Stats/Deaths Setup --
# We'll use a bar chart for deaths
death_reasons = ['Starvation', 'Poison']
bars = ax_stats.bar(death_reasons, [0, 0], color=['gray', 'red'])
ax_stats.set_title("Cause of Death")
ax_stats.set_ylim(0, 10)

# 3. The Animation Loop
def update(frame):
    sim.step()  # Advance physics

    # 1. Update Sim View
    data_sim = sim.get_render_grid()
    img_sim.set_data(data_sim)

    # 2. Update Energy Plot
    # Calculate avg energy for surviving agents of each type
    omni_energies = [a.energy for a in sim.agents if 'Omniscient' in a.behavior.__class__.__name__]
    proxy_energies = [a.energy for a in sim.agents if 'Proxy' in a.behavior.__class__.__name__]
    
    avg_omni = np.mean(omni_energies) if omni_energies else 0
    avg_proxy = np.mean(proxy_energies) if proxy_energies else 0
    
    # Append to plot data (inefficient but works for simple demo)
    xdata = np.arange(sim.step_count)
    # We need history. simpler: just re-calculate history from stats
    # Re-calculating from stats history might be better if we want full lines
    
    # Actually, let's use the stats history which stores EVERY agent's energy every step
    # We need to process it to get averages per step.
    # This might be slow for animation. Let's just append current avg to a local list for plotting
    
    # Optimization: Store avg history in the update closure or global
    if not hasattr(update, 'history_omni'):
        update.history_omni = []
        update.history_proxy = []
    
    update.history_omni.append(avg_omni)
    update.history_proxy.append(avg_proxy)
    
    line_omni.set_data(range(len(update.history_omni)), update.history_omni)
    line_proxy.set_data(range(len(update.history_proxy)), update.history_proxy)
    
    ax_energy.set_xlim(0, max(100, sim.step_count))
    ax_energy.set_ylim(0, max(100, max(update.history_omni + update.history_proxy + [1])))

    # 3. Update Heatmap
    # Log scale might be better eventually, but linear for now
    img_heatmap.set_data(sim.stats['heatmap'])
    img_heatmap.set_clim(vmax=np.max(sim.stats['heatmap']) + 1)

    # 4. Update Death Stats
    deaths = sim.stats['deaths']
    starved = sum(1 for d in deaths if d['reason'] == 'Starvation')
    poisoned = sum(1 for d in deaths if d['reason'] == 'Poison')
    
    for bar, h in zip(bars, [starved, poisoned]):
        bar.set_height(h)
    
    # Add suspicion text info to Stats
    # Calculate avg suspicion for living proxy agents
    proxy_sus_scores = [a.suspicion_score for a in sim.agents if 'Proxy' in a.behavior.__class__.__name__]
    avg_sus = np.mean(proxy_sus_scores) if proxy_sus_scores else 0
    
    # Update title of stats with suspicion
    ax_stats.set_title(f"Deaths | Avg Proxy Suspicion: {avg_sus:.2f}")

    return img_sim, line_omni, line_proxy, img_heatmap, *bars

# Run animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
plt.tight_layout()
plt.show()

