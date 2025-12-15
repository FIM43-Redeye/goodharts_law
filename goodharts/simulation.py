"""
Simulation orchestrator for Goodhart's Law demonstration.

Manages the valid vectorized environment and agents.
Refactored to use VecEnv (with shared_grid=True) replacing the legacy World/Organism.
"""
from goodharts.behaviors.learned import LEARNED_PRESETS
from goodharts.environments.vec_env import create_vec_env
from goodharts.behaviors import get_behavior, create_learned_behavior
from goodharts.utils.logging_config import get_logger
from goodharts.modes import ObservationSpec, get_mode_for_requirement
import numpy as np


logger = get_logger("simulation")


class AgentWrapper:
    """
    Wraps an agent in the VecEnv to provide an object-oriented interface 
    compatible with Behavior strategies and Visualization.
    """
    def __init__(self, idx: int, behavior, vec_env, config: dict):
        self.idx = idx
        self.behavior = behavior
        self.vec_env = vec_env
        self.config = config
        self.id = id(self)
        self.sight_radius = vec_env.view_radius
        self.death_reason = None
        self.suspicion_score = 0
        self.steps_alive = 0
        
        # Backward compatibility
        self.initial_energy = vec_env.initial_energy

    @property
    def x(self):
        return self.vec_env.agent_x[self.idx]
    
    @property
    def y(self):
        return self.vec_env.agent_y[self.idx]
        
    @property
    def energy(self):
        return self.vec_env.agent_energy[self.idx]
        
    @property
    def alive(self):
        # In VecEnv, agents are reset on done, so they are always "alive" techinically.
        # But for visualization of death events, we track them.
        # Simulation.step handles death events.
        return True

    def get_observation(self):
        # Helper mostly for brain viz getting scalars
        # We can reconstruct what Organism.get_observation returned if needed
        # But mostly we need specific channels
        
        # We will need to mock Observation if dependent. 
        # For now, just return a dummy if needed, but behaviors use get_local_view which we provide.
        pass

    def get_local_view(self, mode: str | None = None) -> np.ndarray:
        # This is inefficient if called individually (re-extracts entire batch).
        # Should be used sparingly (e.g. brain visualizer).
        # For main loop, we use batch extraction.
        
        # We need to trigger the full batch extraction
        all_obs = self.vec_env._get_observations()
        # But we need to filter for the specific mode...
        # VecEnv only supports ONE ObsSpec (one set of channels).
        # We assume VecEnv was initialized with a superset of channels needed?
        # Or we rely on VecEnv ObsSpec matching the agent requirements?
        # In Shared World, all agents share the ObsSpec.
        return all_obs[self.idx]


class Simulation:
    """
    Main simulation class that orchestrates agents in a vectorized world.
    
    Attributes:
        config: Runtime configuration dictionary
        vec_env: The vectorized environment
        agents: List of AgentWrapper objects
        step_count: Number of simulation steps completed
        stats: Dictionary of collected statistics
    """
    
    def __init__(self, config: dict):
        logger.info("Initializing Simulation (Vectorized)")
        self.config: dict = config
        
        # 1. Setup Agents Configuration
        agents_setup = []
        behaviors = []
        agent_types = []  # For visibility
        CellType = config['CellType']
        
        for setup in config['AGENTS_SETUP']:
            b_class_name = setup['behavior_class']
            count = setup['count']
            
            # Extract kwargs
            behavior_kwargs = {k: v for k, v in setup.items() 
                             if k not in ('behavior_class', 'count')}
            
            for _ in range(count):
                if b_class_name in LEARNED_PRESETS:
                    behavior = create_learned_behavior(b_class_name, **behavior_kwargs)
                else:
                    BehaviorClass = get_behavior(b_class_name)
                    behavior = BehaviorClass(**behavior_kwargs)
                
                behaviors.append(behavior)
                
                # Determine Agent Type for Visibility
                role = getattr(behavior, 'role', 'prey')
                a_type = CellType.PREDATOR.value if role == 'predator' else CellType.PREY.value
                agent_types.append(a_type)
        
        num_agents = len(behaviors)
        
        # 2. Initialize VecEnv
        # We need an ObservationSpec that covers all agents?
        # For now, we pick a default or 'ground_truth' which usually covers everything.
        # Or better: Union of requirements?
        # Simpler: 'ground_truth' mode spec usually has all channels.
        req = 'ground_truth'
        mode = get_mode_for_requirement(req, config)
        spec = ObservationSpec.for_mode(mode, config)
        
        self.vec_env = create_vec_env(
            n_envs=num_agents, 
            obs_spec=spec, 
            config=config,
            shared_grid=True,
            agent_types=agent_types
        )
        
        # 3. Create Agent Wrappers
        self.agents = [
            AgentWrapper(i, behaviors[i], self.vec_env, config) 
            for i in range(num_agents)
        ]
        
        self.step_count = 0
        self.stats = {
            'deaths': [],  # list of {'step': int, 'id': int, 'reason': str}
            'energy_history': {a.id: [] for a in self.agents},
            'heatmap': {'all': np.zeros((self.vec_env.height, self.vec_env.width))},
            'suspicion_history': {a.id: [] for a in self.agents}
        }
        
    @property
    def world(self):
        # For backward compatibility with visualizer
        return self # We expose 'width', 'height', 'grid' directly?
        
    @property
    def width(self):
        return self.vec_env.width
        
    @property
    def height(self):
        return self.vec_env.height
        
    @property
    def grid(self):
        # Visualization expects a grid
        return self.vec_env.grids[0]

    def step(self):
        """Advance simulation by one timestep."""
        self.step_count += 1
        
        # 1. Get Observations (Vectorized)
        obs_batch = self.vec_env._get_observations() # (N, C, H, W)
        
        # 2. Decide Actions (Loop over behaviors)
        actions = []
        for i, agent in enumerate(self.agents):
            try:
                # AgentWrapper passes itself as 'agent'
                # Obs is (C, H, W)
                dx, dy = agent.behavior.decide_action(agent, obs_batch[i])
                
                # Convert (dx, dy) to action index for VecEnv
                # We assume behavior uses same Action Space logic
                from goodharts.behaviors.action_space import action_to_index
                action_idx = action_to_index(dx, dy)
                actions.append(action_idx)
            except Exception as e:
                logger.error(f"Error in agent {i} decide_action: {e}")
                actions.append(0) # No-Op
        
        actions_np = np.array(actions, dtype=np.int32)
        
        # 3. Step VecEnv
        # returns next_obs, rewards, dones
        _, rewards, dones = self.vec_env.step(actions_np)
        
        # 4. Process Events (Deaths, Stats)
        for i, agent in enumerate(self.agents):
            # Check for death/reset
            if dones[i]:
                # Infer reason from reward (poison penalty is very negative: -150)
                if rewards[i] <= -100:
                    reason = "Poisoned"
                elif agent.energy <= 0:
                    reason = "Starvation"
                else:
                    reason = "Old Age"
                
                # Log death
                b_name = str(agent.behavior)
                if b_name.startswith('<'):
                    b_name = type(agent.behavior).__name__
                    
                self.stats['deaths'].append({
                    'step': self.step_count,
                    'id': agent.id,
                    'reason': reason,
                    'behavior': b_name
                })
                logger.debug(f"Agent {agent.id} died from {reason}")
                
            # Update Stats
            if agent.id not in self.stats['energy_history']:
                 self.stats['energy_history'][agent.id] = []
            self.stats['energy_history'][agent.id].append(agent.energy)
            
            # Suspicion (NOT tracked by VecEnv, logic was in Organism)
            # We skip suspicion for now or need to reimplement logic?
            # It required checking Proxy vs Ground Truth.
            # Skipping for efficiency unless critical.
            
            # Update Heatmap
            # Global heatmap
            self.stats['heatmap']['all'][agent.y, agent.x] += 1
            
            # Per-behavior heatmap (initialize lazily)
            b_name = str(agent.behavior)
            if b_name.startswith('<'):
                b_name = type(agent.behavior).__name__
            
            if b_name not in self.stats['heatmap']:
                self.stats['heatmap'][b_name] = np.zeros((self.vec_env.height, self.vec_env.width))
            self.stats['heatmap'][b_name][agent.y, agent.x] += 1

        if self.step_count % 100 == 0:
            logger.info(f"Completed step {self.step_count}. Agents running: {len(self.agents)}")

    def get_render_grid(self) -> np.ndarray:
        """
        Get the environmental grid for rendering (no agent overlay).
        """
        return self.vec_env.grids[0].copy()
    
    def get_agent_positions(self) -> list[tuple[int, int, tuple[int, int, int]]]:
        """
        Get agent positions and colors for overlay rendering.
        """
        return [
            (a.x, a.y, a.behavior.color)
            for a in self.agents
            if a.alive # Always true currently
        ]

