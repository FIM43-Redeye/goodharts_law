from .environments import create_world
from .agents import Organism
from .behaviors import OmniscientSeeker, ProxySeeker, LearnedGroundTruth, LearnedProxy
from .utils.logging_config import get_logger
import numpy as np

# Mapping string names to classes
BEHAVIORS = {
    'OmniscientSeeker': OmniscientSeeker,
    'ProxySeeker': ProxySeeker,
    'LearnedGroundTruth': LearnedGroundTruth,
    'LearnedProxy': LearnedProxy,
}

logger = get_logger("simulation")

class Simulation:
    def __init__(self, config: dict):
        logger.info("Initializing Simulation")
        self.config: dict = config
        self.world: 'World' = create_world(config)
        self.world.place_food(config['GRID_FOOD_INIT'])
        self.world.place_poison(config['GRID_POISON_INIT'])

        self.agents: list[Organism] = []
        
        for setup in config['AGENTS_SETUP']:
            b_class_name = setup['behavior_class']
            count = setup['count']
            
            if b_class_name not in BEHAVIORS:
                raise ValueError(f"Unknown behavior class: {b_class_name}")
            
            BehaviorClass = BEHAVIORS[b_class_name]
            
            for _ in range(count):
                randx = np.random.randint(0, self.world.width)
                randy = np.random.randint(0, self.world.height)
                behavior = BehaviorClass()
                self.agents.append(Organism(randx, randy, config['ENERGY_START'], config['AGENT_VIEW_RANGE'], self.world, behavior, config))
        
        self.step_count = 0
        self.stats = {
            'deaths': [],  # list of {'step': int, 'id': int, 'reason': str}
            'energy_history': {a.id: [] for a in self.agents},  # {agent_id: [energy values]}
            'heatmap': np.zeros((self.world.height, self.world.width)),  # count of visits
            'suspicion_history': {a.id: [] for a in self.agents} # {agent_id: [suspicion values]}
        }

    def step(self):
        self.step_count += 1
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
        
        if self.step_count % 100 == 0:
            logger.info(f"Completed step {self.step_count}. Alive agents: {len(self.agents)}")

    def get_render_grid(self) -> np.ndarray:
        render_grid = self.world.grid.copy()
        for agent in self.agents:
            if 0 <= agent.x < self.world.width and 0 <= agent.y < self.world.height:
                # 4 for Ground-Truth agents, 5 for Proxy agents
                # Check for both hardcoded and learned proxy behaviors
                val = 4  # Default: Ground-Truth (cyan)
                if isinstance(agent.behavior, (ProxySeeker, LearnedProxy)):
                    val = 5  # Proxy (magenta)
                render_grid[agent.y, agent.x] = val
        return render_grid

