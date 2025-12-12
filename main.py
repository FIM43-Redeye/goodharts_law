import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from simulation_refactored import Simulation
from configs.default_config import get_config

# 1. Setup Simulation
config = get_config()
sim = Simulation(config)

# 2. Setup Plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Define colors: 0=Empty(Black), 1=Wall(Gray), 2=Food(Green), 3=Poison(Red), 4=Agent(Blue)
cmap = colors.ListedColormap(['black', 'gray', 'green', 'red', 'cyan'])
bounds = [0, 1, 2, 3, 4, 5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Initialize the image object
img = ax.imshow(sim.get_render_grid(), cmap=cmap, norm=norm)
plt.grid(False)  # Turn off grid lines for a cleaner look


# 3. The Animation Loop
def update(frame):
    sim.step()  # Advance physics

    # Update visual
    data = sim.get_render_grid()
    img.set_data(data)

    # Update title with stats
    ax.set_title(f"Step: {frame} | Agents: {len(sim.agents)}", color='black')
    return img,


# Run animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
plt.show()
