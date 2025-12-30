"""
Simulation orchestrator for Goodhart's Law demonstration.

Manages the valid vectorized environment and agents.
Refactored to use TorchVecEnv replacing the legacy World/Organism.

NOTE: Visualization mode is currently limited - all agents share grid[0] but have
independent environments. Full shared_grid support removed for training optimization.
"""
import torch
from goodharts.behaviors.learned import LEARNED_PRESETS
from goodharts.environments import create_vec_env
from goodharts.behaviors import get_behavior, create_learned_behavior
from goodharts.behaviors.utils import get_behavior_name
from goodharts.utils.logging_config import get_logger
from goodharts.utils.device import get_device
from goodharts.modes import ObservationSpec, get_mode_for_requirement


logger = get_logger("simulation")


# Death detection thresholds based on reward values
# Poison penalty is -10 (plus move cost and death penalty), so <= -15 indicates poison
POISON_REWARD_THRESHOLD = -15
# Death penalty alone is -10, so starvation (energy depleted) gives ~ -10
STARVATION_REWARD_THRESHOLD = -9


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
        return self.vec_env.agent_x[self.idx].item()
    
    @property
    def y(self):
        return self.vec_env.agent_y[self.idx].item()
        
    @property
    def energy(self):
        return self.vec_env.agent_energy[self.idx].item()
        
    @property
    def alive(self):
        # In VecEnv, agents are reset on done, so they are always "alive" techinically.
        # But for visualization of death events, we track them.
        # Simulation.step handles death events.
        return True

    def get_local_view(self, mode: str | None = None):
        # This is inefficient if called individually (re-extracts entire batch).
        # Should be used sparingly (e.g. brain visualizer).
        # For main loop, we use batch extraction.
        
        # We need to trigger the full batch extraction
        all_obs = self.vec_env._get_observations()
        # Returns Tensor (C, H, W) on device
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
        logger.info("Initializing Simulation (Vectorized - PyTorch)")
        self.config: dict = config
        self.device = get_device()
        
        # 1. Setup Agents Configuration
        behaviors = []

        for setup in config['AGENTS_SETUP']:
            b_class_name = setup['behavior_class']
            count = setup['count']

            # Extract kwargs
            behavior_kwargs = {k: v for k, v in setup.items()
                             if k not in ('behavior_class', 'count')}

            for _ in range(count):
                if b_class_name in LEARNED_PRESETS:
                    behavior = create_learned_behavior(b_class_name, **behavior_kwargs)
                    if hasattr(behavior, 'agent'):
                        behavior.agent.to(self.device)
                else:
                    BehaviorClass = get_behavior(b_class_name)
                    behavior = BehaviorClass(**behavior_kwargs)

                behaviors.append(behavior)

        num_agents = len(behaviors)

        # 2. Initialize VecEnv
        req = 'ground_truth'
        mode = get_mode_for_requirement(req, config)
        spec = ObservationSpec.for_mode(mode, config)

        # Uses TorchVecEnv via the updated import
        # Agents are tracked by coordinates only (not marked on grid)
        self.vec_env = create_vec_env(
            n_envs=num_agents,
            obs_spec=spec,
            config=config,
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
            'heatmap': {'all': torch.zeros((self.vec_env.height, self.vec_env.width), device=self.device)},
            'suspicion_history': {a.id: [] for a in self.agents}
        }
        
    @property
    def world(self):
        # For backward compatibility with visualizer
        return self 
        
    @property
    def width(self):
        return self.vec_env.width
        
    @property
    def height(self):
        return self.vec_env.height
        
    @property
    def grid(self):
        # Visualization expects a grid
        # Return cpu numpy for now to avoid breaking visualizer?
        # Or return tensor and let others handle it?
        # existing code: grids[0]
        # TorchEnv grids is (N, H, W)
        return self.vec_env.grids[0]

    def step(self):
        """Advance simulation by one timestep."""
        self.step_count += 1
        
        # 1. Get Observations (Vectorized Tensor)
        obs_batch = self.vec_env._get_observations() # (N, C, H, W)
        
        # 2. Decide Actions (Loop over behaviors)
        actions = []
        for i, agent in enumerate(self.agents):
            try:
                # AgentWrapper passes itself as 'agent'
                # Obs is (C, H, W) tensor
                dx, dy = agent.behavior.decide_action(agent, obs_batch[i])
                
                # Convert (dx, dy) to action index for VecEnv
                from goodharts.behaviors.action_space import action_to_index
                action_idx = action_to_index(dx, dy)
                actions.append(action_idx)
            except Exception as e:
                logger.error(f"Error in agent {i} decide_action: {e}")
                actions.append(0) # No-Op
        
        actions_tensor = torch.tensor(actions, dtype=torch.int32, device=self.vec_env.device)

        # 3. Step VecEnv
        # returns next_obs, eating_info, terminated, truncated
        _, eating_info, terminated, truncated = self.vec_env.step(actions_tensor)
        food_mask, poison_mask, starved_mask = eating_info
        dones = terminated | truncated  # Combined for reset detection

        # 4. Process Events (Deaths, Stats)
        # We iterate to log events - might be slow for huge N, but fine for sim viz
        # Move relevant tensors to CPU once if needed, or index scalar

        # Accessing single scalars from CUDA tensor is slow due to synchronization
        # But for 'simulation mode' (visualized), we tolerate it.
        # For training mode, we wouldn't use this class (Trainer uses VecEnv directly).

        dones_cpu = dones.cpu().numpy()
        terminated_cpu = terminated.cpu().numpy()
        truncated_cpu = truncated.cpu().numpy()
        poison_cpu = poison_mask.cpu().numpy()

        for i, agent in enumerate(self.agents):
            # Check for death/reset
            if dones_cpu[i]:
                # Determine reason from terminated/truncated signals
                if terminated_cpu[i]:
                    # Agent truly died - check if poisoned or starved
                    if poison_cpu[i]:
                        reason = "Poisoned"
                    else:
                        reason = "Starvation"
                else:
                    # truncated_cpu[i] must be True - hit time limit
                    reason = "Old Age"
                
                # Log death
                b_name = get_behavior_name(agent.behavior)
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
            
            # Update Heatmap (Torch tensors on device)
            # Global heatmap
            self.stats['heatmap']['all'][agent.y, agent.x] += 1
            
            # Per-behavior heatmap
            b_name = get_behavior_name(agent.behavior)
            if b_name not in self.stats['heatmap']:
                self.stats['heatmap'][b_name] = torch.zeros((self.vec_env.height, self.vec_env.width), device=self.device)
            self.stats['heatmap'][b_name][agent.y, agent.x] += 1

        if self.step_count % 100 == 0:
            logger.info(f"Completed step {self.step_count}. Agents running: {len(self.agents)}")

    def get_render_grid(self):
        """
        Get the environmental grid for rendering (no agent overlay).
        Converts to NumPy for matplotlib compatibility.
        """
        import numpy as np
        return self.vec_env.grids[0].cpu().numpy().copy()
    
    def get_agent_positions(self) -> list[tuple[int, int, tuple[int, int, int]]]:
        """
        Get agent positions and colors for overlay rendering.
        """
        return [
            (a.x, a.y, a.behavior.color)
            for a in self.agents
            if a.alive # Always true currently
        ]

