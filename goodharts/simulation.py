"""
Simulation orchestrator for Goodhart's Law demonstration.

Manages the world, agents, and simulation loop.
"""
from goodharts.environments import create_world
from goodharts.agents import Organism
from goodharts.behaviors import get_behavior
from goodharts.utils.logging_config import get_logger
import numpy as np


logger = get_logger("simulation")


class Simulation:
    """
    Main simulation class that orchestrates agents in a world.
    
    Attributes:
        config: Runtime configuration dictionary
        world: The environment grid
        agents: List of living organisms
        step_count: Number of simulation steps completed
        stats: Dictionary of collected statistics
    """
    
    def __init__(self, config: dict):
        logger.info("Initializing Simulation")
        self.config: dict = config
        self.world = create_world(config)
        self.world.place_food(config['GRID_FOOD_INIT'])
        self.world.place_poison(config['GRID_POISON_INIT'])

        self.agents: list[Organism] = []
        
        for setup in config['AGENTS_SETUP']:
            b_class_name = setup['behavior_class']
            count = setup['count']
            
            # Use registry for behavior lookup (auto-discovery)
            BehaviorClass = get_behavior(b_class_name)
            
            # Extract optional behavior kwargs from setup
            behavior_kwargs = {k: v for k, v in setup.items() 
                             if k not in ('behavior_class', 'count')}
            
            for _ in range(count):
                randx = np.random.randint(0, self.world.width)
                randy = np.random.randint(0, self.world.height)
                behavior = BehaviorClass(**behavior_kwargs)
                self.agents.append(Organism(
                    randx, randy, 
                    config['ENERGY_START'], 
                    config['AGENT_VIEW_RANGE'], 
                    self.world, behavior, config
                ))
        
        self.step_count = 0
        self.stats = {
            'deaths': [],  # list of {'step': int, 'id': int, 'reason': str}
            'energy_history': {a.id: [] for a in self.agents},
            'heatmap': np.zeros((self.world.height, self.world.width)),
            'suspicion_history': {a.id: [] for a in self.agents}
        }

    def step(self):
        """Advance simulation by one timestep."""
        self.step_count += 1
        CellType = self.config['CellType']
        
        # Mark all living agents on grid BEFORE they take actions
        # (so agents can see each other in their observations)
        agent_positions = []
        for agent in self.agents:
            if agent.alive and 0 <= agent.x < self.world.width and 0 <= agent.y < self.world.height:
                # Only mark if cell is empty (don't overwrite food/poison)
                if self.world.grid[agent.y, agent.x] == CellType.EMPTY:
                    # Use behavior's role to determine cell type
                    cell_type = CellType.PREDATOR if agent.behavior.role == 'predator' else CellType.PREY
                    self.world.grid[agent.y, agent.x] = cell_type.value
                    agent_positions.append((agent.x, agent.y, cell_type.value))
        
        for agent in self.agents[:]:
            if not agent.alive:
                self.stats['deaths'].append({
                    'step': self.step_count,
                    'id': agent.id,
                    'reason': agent.death_reason
                })
                logger.debug(f"Agent {agent.id} died from {agent.death_reason}")
                self.agents.remove(agent)
                continue
            
            agent.update()
            
            # Update Stats
            if agent.id not in self.stats['energy_history']:
                 self.stats['energy_history'][agent.id] = []
            self.stats['energy_history'][agent.id].append(agent.energy)
            
            if agent.id not in self.stats['suspicion_history']:
                 self.stats['suspicion_history'][agent.id] = []
            self.stats['suspicion_history'][agent.id].append(agent.suspicion_score)
            
            # Update Heatmap
            if 0 <= agent.y < self.world.height and 0 <= agent.x < self.world.width:
                self.stats['heatmap'][agent.y, agent.x] += 1
        
        # Clear agent markers (agents move, so positions change)
        for x, y, cell_value in agent_positions:
            if self.world.grid[y, x] == cell_value:
                self.world.grid[y, x] = CellType.EMPTY.value
        
        if self.step_count % 100 == 0:
            logger.info(f"Completed step {self.step_count}. Alive agents: {len(self.agents)}")

    def get_render_grid(self) -> np.ndarray:
        """
        Get the environmental grid for rendering (no agent overlay).
        
        Returns:
            Copy of world grid with CellType values.
            Use get_agent_positions() to overlay agents.
        """
        return self.world.grid.copy()
    
    def get_agent_positions(self) -> list[tuple[int, int, tuple[int, int, int]]]:
        """
        Get agent positions and colors for overlay rendering.
        
        Returns:
            List of (x, y, rgb_color) tuples for each alive agent.
            Color comes from behavior.color property.
        """
        return [
            (a.x, a.y, a.behavior.color)
            for a in self.agents
            if a.alive and 0 <= a.x < self.world.width and 0 <= a.y < self.world.height
        ]
