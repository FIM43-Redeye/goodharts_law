"""
Data collection utilities for training learned behaviors.

Runs simulations with exploratory agents and records experiences
for subsequent training via behavior cloning or RL.
"""
import numpy as np
from typing import Callable

from ..simulation import Simulation
from ..agents import Organism
from ..behaviors import LearnedBehavior, OmniscientSeeker, ProxySeeker
from ..behaviors.action_space import action_to_index, build_action_space
from .dataset import ReplayBuffer, Experience


def collect_experiences(
    config: dict,
    behavior_factory: Callable[[], LearnedBehavior | OmniscientSeeker | ProxySeeker],
    num_steps: int = 500,
    epsilon: float = 0.3,
    seed: int | None = None,
) -> ReplayBuffer:
    """
    Run a simulation and collect experiences for training.
    
    Args:
        config: Simulation configuration dict
        behavior_factory: Callable that creates a behavior instance
        num_steps: Number of simulation steps to run
        epsilon: Exploration rate (for LearnedBehavior)
        seed: Random seed for reproducibility
        
    Returns:
        ReplayBuffer containing collected experiences
    """
    if seed is not None:
        np.random.seed(seed)
    
    buffer = ReplayBuffer()
    
    # Create simulation
    sim = Simulation(config)
    
    for step in range(num_steps):
        # Collect experience BEFORE taking action
        for agent in sim.agents:
            if not agent.alive:
                continue
            
            # Get current state
            state = agent.get_local_view()
            energy_before = agent.energy
            
            # Store for later (we need to record after action)
            agent._collect_state = state.copy()
            agent._collect_energy = energy_before
        
        # Step simulation (agents take actions)
        sim.step()
        
        # Record experiences AFTER action
        for agent in sim.agents:
            if not hasattr(agent, '_collect_state'):
                continue
            
            state = agent._collect_state
            energy_delta = agent.energy - agent._collect_energy
            
            # Infer action from behavior (approximate - ideally we'd instrument decide_action)
            # For now, we use the energy delta as reward signal
            reward = energy_delta
            done = not agent.alive
            
            # Get action index from behavior
            if hasattr(agent.behavior, 'action_to_index'):
                # For learned behaviors, we can get the last action
                # This is imperfect - ideally behavior would store last_action
                action_idx = 0  # Placeholder
            else:
                # For hardcoded behaviors, we'd need to instrument them
                action_idx = 0  # Placeholder
            
            next_state = agent.get_local_view() if agent.alive else None
            
            buffer.add(
                state=state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # Cleanup
            del agent._collect_state
            del agent._collect_energy
        
        # Also record deaths
        for death in sim.stats['deaths']:
            if death['step'] == sim.step_count:
                # Agent died this step - terminal experience with negative reward
                # (already recorded above with done=True)
                pass
    
    return buffer


def collect_from_expert(
    config: dict,
    expert_class: type = OmniscientSeeker,
    num_steps: int = 500,
    num_agents: int = 10,
    seed: int | None = None,
) -> ReplayBuffer:
    """
    Collect experiences by observing an expert (heuristic) behavior.
    
    This is for imitation learning / behavior cloning:
    we watch the OmniscientSeeker and learn to mimic it.
    
    Args:
        config: Simulation configuration
        expert_class: The expert behavior class to observe
        num_steps: Steps to run
        num_agents: Number of expert agents
        seed: Random seed
        
    Returns:
        ReplayBuffer with expert demonstrations
    """
    if seed is not None:
        np.random.seed(seed)
    
    buffer = ReplayBuffer()
    
    # Override config to use expert agents only
    modified_config = config.copy()
    modified_config['AGENTS_SETUP'] = [
        {'behavior_class': expert_class.__name__, 'count': num_agents}
    ]
    
    sim = Simulation(modified_config)
    
    for step in range(num_steps):
        for agent in sim.agents:
            if not agent.alive:
                continue
            
            # Get state BEFORE action
            state = agent.get_local_view()
            energy_before = agent.energy
            pos_before = (agent.x, agent.y)
            
            # Let expert decide action
            action = agent.behavior.decide_action(agent, state)
            dx, dy = action
            
            # Convert to action index using centralized action space
            action_idx = action_to_index(dx, dy, max_move_distance=1)
            
            # Store for reward calculation
            agent._bc_state = state.copy()
            agent._bc_action_idx = action_idx
            agent._bc_energy = energy_before
        
        # Step simulation
        sim.step()
        
        # Record with actual rewards
        for agent in sim.agents:
            if not hasattr(agent, '_bc_state'):
                continue
            
            reward = agent.energy - agent._bc_energy
            done = not agent.alive
            next_state = agent.get_local_view() if agent.alive else None
            
            buffer.add(
                state=agent._bc_state,
                action=agent._bc_action_idx,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            del agent._bc_state
            del agent._bc_action_idx
            del agent._bc_energy
    
    return buffer


def collect_parallel(
    config: dict,
    behavior_factory: Callable[[], LearnedBehavior | OmniscientSeeker | ProxySeeker],
    num_workers: int = 4,
    steps_per_worker: int = 500,
    seed_base: int = 0,
) -> ReplayBuffer:
    """
    Collect experiences using multiple simulation workers.
    
    TODO: Implement with multiprocessing for speed.
    For now, runs sequentially.
    
    Args:
        config: Simulation configuration
        behavior_factory: Creates behavior instances
        num_workers: Number of parallel simulations
        steps_per_worker: Steps per simulation
        seed_base: Base seed (each worker gets seed_base + worker_id)
        
    Returns:
        Combined ReplayBuffer from all workers
    """
    combined_buffer = ReplayBuffer()
    
    for worker_id in range(num_workers):
        worker_buffer = collect_experiences(
            config=config,
            behavior_factory=behavior_factory,
            num_steps=steps_per_worker,
            seed=seed_base + worker_id
        )
        
        # Merge buffers
        for exp in worker_buffer.buffer:
            combined_buffer.buffer.append(exp)
    
    return combined_buffer


def generate_poison_avoidance_samples(
    num_samples: int = 500,
    view_size: int = 11,
    reward_weight: float = 10.0,
) -> ReplayBuffer:
    """
    Generate synthetic training samples teaching poison avoidance.
    
    Creates observations with poison at various positions and labels
    with the "correct" action being to move AWAY from poison.
    
    This fills a gap in behavior cloning: expert never eats poison,
    so the CNN never learns what to do when it sees poison.
    
    Args:
        num_samples: Number of samples to generate
        view_size: Size of the observation grid (default 11x11)
        reward_weight: Reward value for these samples (high = important)
        
    Returns:
        ReplayBuffer with poison-avoidance samples
    """
    buffer = ReplayBuffer()
    center = view_size // 2
    
    # Map poison direction to "escape" action
    # If poison is left -> go right, etc.
    escape_actions = {
        'left': (1, 0),     # Go right
        'right': (-1, 0),   # Go left  
        'up': (0, 1),       # Go down
        'down': (0, -1),    # Go up
        'up_left': (1, 1),  # Go down-right
        'up_right': (-1, 1),  # Go down-left
        'down_left': (1, -1), # Go up-right
        'down_right': (-1, -1), # Go up-left
    }
    
    # Poison positions relative to center
    poison_positions = {
        'left': (center, center - 2),
        'right': (center, center + 2),
        'up': (center - 2, center),
        'down': (center + 2, center),
        'up_left': (center - 2, center - 2),
        'up_right': (center - 2, center + 2),
        'down_left': (center + 2, center - 2),
        'down_right': (center + 2, center + 2),
    }
    
    directions = list(escape_actions.keys())
    
    for i in range(num_samples):
        # Pick a random direction for poison
        direction = directions[i % len(directions)]
        poison_pos = poison_positions[direction]
        escape_action = escape_actions[direction]
        
        # Build observation: 4 channels (empty, wall, food, poison)
        state = np.zeros((4, view_size, view_size), dtype=np.float32)
        state[0, :, :] = 1.0  # All empty
        state[0, poison_pos[0], poison_pos[1]] = 0.0  # Not empty at poison
        state[3, poison_pos[0], poison_pos[1]] = 1.0  # Poison channel
        
        # Sometimes add food elsewhere to make it more realistic
        if np.random.random() < 0.3:
            # Add food on the escape side
            food_offset = (escape_action[0] * 3, escape_action[1] * 3)
            food_pos = (center + food_offset[0], center + food_offset[1])
            if 0 <= food_pos[0] < view_size and 0 <= food_pos[1] < view_size:
                state[0, food_pos[0], food_pos[1]] = 0.0
                state[2, food_pos[0], food_pos[1]] = 1.0
        
        action_idx = action_to_index(escape_action[0], escape_action[1])
        
        buffer.add(
            state=state,
            action=action_idx,
            reward=reward_weight,  # High reward encourages learning this
            next_state=None,
            done=False
        )
    
    return buffer
