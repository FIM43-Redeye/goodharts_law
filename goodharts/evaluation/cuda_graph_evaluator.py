"""
CUDA Graph-accelerated evaluation for Goodhart agents.

Captures the evaluation loop (obs -> forward -> action -> env step) as a CUDA graph
and replays it with minimal CPU overhead. Supports multiple models in a single graph
for maximum GPU utilization.

Key benefits:
- Eliminates kernel launch overhead (~5-10us per kernel, 20+ kernels per step)
- Enables kernel interleaving across models
- Single graph launch for multi-model evaluation

Requirements:
- CUDA device (graphs not supported on CPU)
- Fixed tensor shapes (no dynamic allocation)
- No CPU-GPU sync in the captured region
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

import torch
import torch.cuda

from goodharts.utils.device import get_device
from goodharts.environments.torch_env import create_torch_vec_env
from goodharts.behaviors.learned import create_learned_behavior
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_simulation_config


@dataclass
class GraphConfig:
    """Configuration for CUDA graph evaluation."""
    n_envs: int = 256
    total_timesteps: int = 100000
    sync_interval: int = 1000  # Steps between metric collection syncs
    warmup_steps: int = 10  # Steps before graph capture
    deterministic: bool = True  # Argmax actions for reproducibility
    temperature: float = 1.0


@dataclass
class GraphMetrics:
    """Accumulated metrics from graph evaluation."""
    mode: str
    total_steps: int = 0
    total_deaths: int = 0
    total_food: int = 0
    total_poison: int = 0
    total_survival_steps: int = 0  # Sum of steps alive before each death
    wall_time: float = 0.0
    graph_replays: int = 0

    @property
    def steps_per_second(self) -> float:
        return self.total_steps / self.wall_time if self.wall_time > 0 else 0

    @property
    def deaths_per_1k(self) -> float:
        return (self.total_deaths / self.total_steps) * 1000 if self.total_steps > 0 else 0

    @property
    def efficiency(self) -> float:
        total = self.total_food + self.total_poison
        return self.total_food / total if total > 0 else 0

    @property
    def mean_survival(self) -> float:
        """Average steps survived before death."""
        return self.total_survival_steps / self.total_deaths if self.total_deaths > 0 else float('inf')


class CUDAGraphEvaluator:
    """
    Single-model evaluation using CUDA graph replay.

    Captures the inference + environment step loop as a graph and replays it
    for maximum throughput.

    Usage:
        evaluator = CUDAGraphEvaluator('ground_truth', 'models/ppo_ground_truth.pth')
        metrics = evaluator.run()
        print(f"Throughput: {metrics.steps_per_second:.0f} steps/sec")
    """

    def __init__(
        self,
        mode: str,
        model_path: Optional[str] = None,
        config: Optional[GraphConfig] = None,
    ):
        self.mode = mode
        self.model_path = model_path or f'models/ppo_{mode}.pth'
        self.config = config or GraphConfig()
        self.device = get_device()

        if self.device.type != 'cuda':
            raise RuntimeError("CUDAGraphEvaluator requires CUDA device")

        self._setup()

    def _setup(self):
        """Initialize environment, model, and pre-allocated tensors."""
        cfg = self.config
        sim_config = get_simulation_config()

        # Create environment
        obs_spec = ObservationSpec.for_mode(self.mode, sim_config)
        self.env = create_torch_vec_env(
            n_envs=cfg.n_envs,
            obs_spec=obs_spec,
            config=sim_config,
            device=self.device,
        )
        # Evaluation requires real energy dynamics (agents must be able to die)
        # freeze_energy_in_training exists only for proxy training exploration
        self.env.freeze_energy = False

        # Load model
        self.behavior = create_learned_behavior(
            preset=self.mode,
            model_path=self.model_path,
        )

        # Get initial observations and infer shape
        initial_obs = self.env.reset()
        obs_shape = initial_obs.shape[1:]  # (channels, height, width)

        # Trigger lazy brain initialization with a single observation
        # get_action_logits expects (C, H, W) for a single obs
        _ = self.behavior.get_action_logits(initial_obs[0])
        self.behavior.brain.eval()

        # Pre-allocate static tensors for graph capture
        # These will be reused every iteration
        self.obs_buffer = torch.empty(
            (cfg.n_envs, *obs_shape),
            dtype=torch.float32,
            device=self.device,
        )
        # Actions are indices into the environment's action_deltas lookup table
        self.action_buffer = torch.empty(
            cfg.n_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.logits_buffer = torch.empty(
            (cfg.n_envs, self.behavior.action_space.n_outputs),
            dtype=torch.float32,
            device=self.device,
        )

        # Metric accumulation tensors (read periodically)
        self.death_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.food_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.poison_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.step_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.survival_steps = torch.zeros(1, dtype=torch.long, device=self.device)

        # Graph will be captured during first run
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.stream = torch.cuda.Stream(device=self.device)

    def _warmup(self):
        """Run warmup steps to stabilize CUDA state before capture."""
        obs = self.env.reset()
        self.obs_buffer.copy_(obs)

        for _ in range(self.config.warmup_steps):
            self._step_no_graph()

        # Sync before capture
        torch.cuda.synchronize(self.device)

    def _step_no_graph(self):
        """Single step without graph capture (for warmup or fallback)."""
        with torch.no_grad():
            # Forward pass
            logits = self.behavior.brain(self.obs_buffer)
            self.logits_buffer.copy_(logits)

            # Get action indices (for discrete action space)
            if self.config.deterministic:
                action_indices = self.logits_buffer.argmax(dim=-1)
            else:
                probs = torch.softmax(self.logits_buffer / self.config.temperature, dim=-1)
                action_indices = torch.multinomial(probs, 1).squeeze(-1)

            self.action_buffer.copy_(action_indices)

            # Environment step
            # TorchVecEnv returns: obs, eating_info, terminated, truncated
            # eating_info = (food_mask, poison_mask, starved_mask)
            obs, eating_info, terminated, truncated = self.env.step(self.action_buffer)
            food_mask, poison_mask, starved_mask = eating_info
            self.obs_buffer.copy_(obs)

            # Accumulate metrics (all in-place operations for graph compatibility)
            self.step_count.add_(self.config.n_envs)
            self.death_count.add_(terminated.sum())
            self.food_count.add_(food_mask.sum())
            self.poison_count.add_(poison_mask.sum())
            # Sum survival steps for agents that just died (masked sum for fixed shapes)
            # last_episode_steps was updated by env for terminated agents
            survival_this_step = torch.where(
                terminated, self.env.last_episode_steps.long(), torch.zeros_like(self.env.last_episode_steps, dtype=torch.long)
            ).sum()
            self.survival_steps.add_(survival_this_step)

    def _capture_graph(self):
        """Capture the evaluation step as a CUDA graph."""
        self.graph = torch.cuda.CUDAGraph()

        # Capture on dedicated stream
        with torch.cuda.stream(self.stream):
            # Warm the stream
            torch.cuda.synchronize(self.device)

            # Capture
            with torch.cuda.graph(self.graph, stream=self.stream):
                self._step_no_graph()

        torch.cuda.synchronize(self.device)

    def _replay_graph(self):
        """Replay the captured graph."""
        self.graph.replay()

    def run(self) -> GraphMetrics:
        """
        Run evaluation and return metrics.

        Returns:
            GraphMetrics with throughput and survival statistics
        """
        metrics = GraphMetrics(mode=self.mode)

        # Warmup and capture
        print(f"[{self.mode}] Warming up...")
        self._warmup()

        print(f"[{self.mode}] Capturing CUDA graph...")
        self._capture_graph()

        # Reset counters
        self.death_count.zero_()
        self.food_count.zero_()
        self.poison_count.zero_()
        self.step_count.zero_()
        self.survival_steps.zero_()

        # Calculate replay count
        steps_per_replay = self.config.n_envs
        total_replays = self.config.total_timesteps // steps_per_replay
        sync_replays = self.config.sync_interval // steps_per_replay

        print(f"[{self.mode}] Running {total_replays:,} graph replays...")
        start_time = time.perf_counter()

        for replay_idx in range(total_replays):
            self._replay_graph()
            metrics.graph_replays += 1

            # Periodic sync for progress reporting
            if (replay_idx + 1) % sync_replays == 0:
                torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - start_time
                steps_done = (replay_idx + 1) * steps_per_replay
                rate = steps_done / elapsed
                print(f"[{self.mode}] {steps_done:,}/{self.config.total_timesteps:,} "
                      f"({rate:.0f} steps/sec)")

        # Final sync and metric collection
        torch.cuda.synchronize(self.device)
        metrics.wall_time = time.perf_counter() - start_time
        metrics.total_steps = self.step_count.item()
        metrics.total_deaths = self.death_count.item()
        metrics.total_food = self.food_count.item()
        metrics.total_poison = self.poison_count.item()
        metrics.total_survival_steps = self.survival_steps.item()

        print(f"[{self.mode}] Complete: {metrics.steps_per_second:.0f} steps/sec, "
              f"efficiency={metrics.efficiency:.1%}, survival={metrics.mean_survival:.1f}")

        return metrics


class MultiModelGraphEvaluator:
    """
    Multi-model evaluation in a single CUDA graph.

    Captures forward passes and environment steps for all models together,
    allowing the GPU to interleave operations for maximum utilization.

    Usage:
        evaluator = MultiModelGraphEvaluator(
            modes=['ground_truth', 'proxy'],
            n_envs_per_mode=128,
        )
        results = evaluator.run()
        for mode, metrics in results.items():
            print(f"{mode}: {metrics.efficiency:.1%} efficiency")
    """

    def __init__(
        self,
        modes: list[str],
        model_paths: Optional[dict[str, str]] = None,
        n_envs_per_mode: int = 128,
        config: Optional[GraphConfig] = None,
    ):
        self.modes = modes
        self.model_paths = model_paths or {m: f'models/ppo_{m}.pth' for m in modes}
        self.config = config or GraphConfig()
        self.config.n_envs = n_envs_per_mode  # Per mode
        self.device = get_device()

        if self.device.type != 'cuda':
            raise RuntimeError("MultiModelGraphEvaluator requires CUDA device")

        self._setup()

    def _setup(self):
        """Initialize environments, models, and buffers for all modes."""
        sim_config = get_simulation_config()
        self.envs: dict = {}
        self.behaviors: dict = {}
        self.buffers: dict = {}

        for mode in self.modes:
            # Environment
            obs_spec = ObservationSpec.for_mode(mode, sim_config)
            env = create_torch_vec_env(
                n_envs=self.config.n_envs,
                obs_spec=obs_spec,
                config=sim_config,
                device=self.device,
            )
            # Evaluation requires real energy dynamics
            env.freeze_energy = False
            self.envs[mode] = env

            # Model
            behavior = create_learned_behavior(
                preset=mode,
                model_path=self.model_paths[mode],
            )
            # Get initial obs and trigger lazy brain initialization
            initial_obs = env.reset()
            obs_shape = initial_obs.shape[1:]  # (channels, height, width)
            _ = behavior.get_action_logits(initial_obs[0])
            behavior.brain.eval()
            self.behaviors[mode] = behavior

            # Pre-allocated buffers
            self.buffers[mode] = {
                'obs': torch.empty(
                    (self.config.n_envs, *obs_shape),
                    dtype=torch.float32,
                    device=self.device,
                ),
                'actions': torch.empty(
                    self.config.n_envs,
                    dtype=torch.long,
                    device=self.device,
                ),
                'logits': torch.empty(
                    (self.config.n_envs, behavior.action_space.n_outputs),
                    dtype=torch.float32,
                    device=self.device,
                ),
                'death_count': torch.zeros(1, dtype=torch.long, device=self.device),
                'food_count': torch.zeros(1, dtype=torch.long, device=self.device),
                'poison_count': torch.zeros(1, dtype=torch.long, device=self.device),
                'step_count': torch.zeros(1, dtype=torch.long, device=self.device),
                'survival_steps': torch.zeros(1, dtype=torch.long, device=self.device),
            }

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.stream = torch.cuda.Stream(device=self.device)

    def _warmup(self):
        """Warmup all models and environments."""
        # Reset all environments
        for mode in self.modes:
            obs = self.envs[mode].reset()
            self.buffers[mode]['obs'].copy_(obs)

        # Warmup steps
        for _ in range(self.config.warmup_steps):
            self._step_all_no_graph()

        torch.cuda.synchronize(self.device)

    def _step_all_no_graph(self):
        """Step all models without graph capture."""
        with torch.no_grad():
            # Forward pass for all models (can be interleaved by GPU)
            for mode in self.modes:
                buf = self.buffers[mode]
                behavior = self.behaviors[mode]

                logits = behavior.brain(buf['obs'])
                buf['logits'].copy_(logits)

            # Decode actions for all models (get action indices)
            for mode in self.modes:
                buf = self.buffers[mode]

                if self.config.deterministic:
                    action_indices = buf['logits'].argmax(dim=-1)
                else:
                    probs = torch.softmax(buf['logits'] / self.config.temperature, dim=-1)
                    action_indices = torch.multinomial(probs, 1).squeeze(-1)

                buf['actions'].copy_(action_indices)

            # Environment steps for all models
            for mode in self.modes:
                buf = self.buffers[mode]
                env = self.envs[mode]

                # TorchVecEnv returns: obs, eating_info, terminated, truncated
                obs, eating_info, terminated, truncated = env.step(buf['actions'])
                food_mask, poison_mask, starved_mask = eating_info
                buf['obs'].copy_(obs)

                # Accumulate metrics (all in-place for graph compatibility)
                buf['step_count'].add_(self.config.n_envs)
                buf['death_count'].add_(terminated.sum())
                buf['food_count'].add_(food_mask.sum())
                buf['poison_count'].add_(poison_mask.sum())
                # Survival steps (masked sum for fixed shapes)
                survival_this_step = torch.where(
                    terminated, env.last_episode_steps.long(), torch.zeros_like(env.last_episode_steps, dtype=torch.long)
                ).sum()
                buf['survival_steps'].add_(survival_this_step)

    def _capture_graph(self):
        """Capture all models' steps in a single graph."""
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(self.stream):
            torch.cuda.synchronize(self.device)

            with torch.cuda.graph(self.graph, stream=self.stream):
                self._step_all_no_graph()

        torch.cuda.synchronize(self.device)

    def _replay_graph(self):
        """Replay the multi-model graph."""
        self.graph.replay()

    def run(self) -> dict[str, GraphMetrics]:
        """
        Run evaluation for all models and return per-mode metrics.

        Returns:
            Dict mapping mode name to GraphMetrics
        """
        results = {mode: GraphMetrics(mode=mode) for mode in self.modes}

        # Warmup and capture
        print(f"[Multi-Model] Warming up {len(self.modes)} models...")
        self._warmup()

        print(f"[Multi-Model] Capturing combined CUDA graph...")
        self._capture_graph()

        # Reset all counters
        for mode in self.modes:
            for key in ['death_count', 'food_count', 'poison_count', 'step_count', 'survival_steps']:
                self.buffers[mode][key].zero_()

        # Calculate replay count
        steps_per_replay = self.config.n_envs  # Per mode
        total_replays = self.config.total_timesteps // steps_per_replay
        sync_replays = self.config.sync_interval // steps_per_replay

        print(f"[Multi-Model] Running {total_replays:,} graph replays "
              f"({len(self.modes)} models x {self.config.n_envs} envs each)...")
        start_time = time.perf_counter()

        for replay_idx in range(total_replays):
            self._replay_graph()

            # Periodic sync
            if (replay_idx + 1) % sync_replays == 0:
                torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - start_time
                steps_done = (replay_idx + 1) * steps_per_replay * len(self.modes)
                rate = steps_done / elapsed
                print(f"[Multi-Model] {steps_done:,} total steps ({rate:.0f} steps/sec)")

        # Final sync and collect metrics
        torch.cuda.synchronize(self.device)
        wall_time = time.perf_counter() - start_time

        for mode in self.modes:
            buf = self.buffers[mode]
            metrics = results[mode]
            metrics.wall_time = wall_time
            metrics.total_steps = buf['step_count'].item()
            metrics.total_deaths = buf['death_count'].item()
            metrics.total_food = buf['food_count'].item()
            metrics.total_poison = buf['poison_count'].item()
            metrics.total_survival_steps = buf['survival_steps'].item()
            metrics.graph_replays = total_replays

        # Summary
        total_steps = sum(m.total_steps for m in results.values())
        combined_rate = total_steps / wall_time
        print(f"\n[Multi-Model] Complete: {combined_rate:.0f} combined steps/sec")
        for mode, metrics in results.items():
            print(f"  {mode}: efficiency={metrics.efficiency:.1%}, "
                  f"deaths/1k={metrics.deaths_per_1k:.2f}, survival={metrics.mean_survival:.1f}")

        return results


def compare_graph_vs_standard(mode: str = 'ground_truth', timesteps: int = 50000):
    """
    Benchmark CUDA graph evaluation against standard evaluation.

    Useful for measuring the actual speedup from graph capture.
    """
    from goodharts.evaluation import EvaluationConfig, ModelTester

    print("=" * 60)
    print("CUDA Graph vs Standard Evaluation Benchmark")
    print("=" * 60)

    # Standard evaluation
    print("\n[Standard] Running...")
    std_config = EvaluationConfig.from_config(
        mode=mode,
        total_timesteps=timesteps,
        deterministic=True,
    )
    std_tester = ModelTester(std_config)
    std_start = time.perf_counter()
    std_result = std_tester.run()
    std_time = time.perf_counter() - std_start
    std_rate = timesteps * std_config.n_envs / std_time

    # CUDA graph evaluation
    print("\n[CUDA Graph] Running...")
    graph_config = GraphConfig(
        n_envs=std_config.n_envs,
        total_timesteps=timesteps * std_config.n_envs,
        deterministic=True,
    )
    graph_runner = CUDAGraphEvaluator(mode, config=graph_config)
    graph_metrics = graph_runner.run()

    # Comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Standard:   {std_rate:,.0f} steps/sec")
    print(f"CUDA Graph: {graph_metrics.steps_per_second:,.0f} steps/sec")
    print(f"Speedup:    {graph_metrics.steps_per_second / std_rate:.2f}x")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CUDA Graph Evaluation')
    parser.add_argument('--mode', default='ground_truth',
                        help='Mode to evaluate (or comma-separated for multi)')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps per mode')
    parser.add_argument('--n-envs', type=int, default=256,
                        help='Environments per mode')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run comparison benchmark')
    parser.add_argument('--multi', action='store_true',
                        help='Use multi-model graph evaluator')

    args = parser.parse_args()

    if args.benchmark:
        compare_graph_vs_standard(args.mode, args.timesteps)
    elif args.multi or ',' in args.mode:
        modes = args.mode.split(',')
        config = GraphConfig(
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
        )
        runner = MultiModelGraphEvaluator(modes, config=config)
        results = runner.run()
    else:
        config = GraphConfig(
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
        )
        runner = CUDAGraphEvaluator(args.mode, config=config)
        metrics = runner.run()
