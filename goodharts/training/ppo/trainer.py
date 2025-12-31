"""
PPO Trainer - Main training orchestrator.

Provides a clean, subclassable interface for PPO training.
Can be extended for multi-agent scenarios.
"""
import logging
import os
import time
import numpy as np
import torch

logger = logging.getLogger(__name__)
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
from torch.profiler import record_function
from typing import Optional

from goodharts.configs.default_config import get_simulation_config
from goodharts.config import get_training_config
from goodharts.utils.device import get_device, apply_system_optimizations, is_tpu, sync_device
from goodharts.utils.seed import set_seed
from goodharts.behaviors.brains import create_brain, save_brain, _clean_state_dict
from goodharts.behaviors.action_space import create_action_space, ActionSpace
from goodharts.environments.torch_env import create_torch_vec_env
from datetime import datetime
from goodharts.modes import RewardComputer

from .models import Profiler, ValueHead, PopArtValueHead
from .algorithms import compute_gae, ppo_update
from .async_logger import AsyncLogger, LogPayload

# Extracted modules
from .monitoring import GPUMonitor
from .ppo_config import PPOConfig
from .metrics import (
    METRICS_SCHEMA, N_SCALAR_METRICS, unpack_metrics,
    PendingMetrics, BookkeepingWork, BackgroundBookkeeper
)
from .globals import (
    request_abort, clear_abort, is_abort_requested, reset_training_state,
    mark_warmup_done, check_warmup_done, get_compile_lock, get_warmup_lock,
)


def _vprint(msg: str, verbose: bool):
    """Verbose print helper."""
    if verbose:
        print(f"[VERBOSE] {msg}", flush=True)


class PPOTrainer:
    """
    PPO training orchestrator.
    
    Handles the full training loop including:
    - Environment creation
    - Experience collection
    - GAE computation
    - PPO updates
    - Logging and checkpointing
    
    Can be subclassed for multi-agent or custom scenarios.
    """
    
    def __init__(
        self,
        config: PPOConfig,
        device: Optional[torch.device] = None,
        dashboard = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: PPO configuration
            device: Torch device (auto-detected if None)
            dashboard: Optional training dashboard for live visualization
        """
        self.config = config
        self.device = device or get_device()
        self.dashboard = dashboard
        
        # Will be initialized on train()
        self.vec_env = None
        self.policy = None
        self.value_head = None
        self.optimizer = None
        self.tb_log_dir = None
        self.reward_computer = None
        self.profiler = None
        self.async_logger = None

        # External profiler callback (called after each update, returns False to stop)
        self._profiler_callback = None

        # GPU monitor (started after warmup if enabled)
        self.gpu_monitor = None

        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.best_reward = float('-inf')
        self.episode_count = 0
        self._aborted = False  # Track if training was aborted

    def _warmup_forward_backward(self, batch_size: int, include_backward: bool = True, label: str = "Warmup") -> float:
        """
        Run forward (and optionally backward) pass to warm up JIT/cuDNN.

        This triggers algorithm selection for cuDNN benchmark mode and compiles
        torch.compile graphs. Running this once at startup avoids the compilation
        penalty during actual training.

        Args:
            batch_size: Batch size for dummy tensors (should match training batch)
            include_backward: Whether to run backward pass (needed for gradient kernels)
            label: Label for progress messages

        Returns:
            Time taken in seconds
        """
        cfg = self.config
        start_time = time.time()

        # Create dummy tensors matching training shapes
        dummy_obs = torch.zeros(
            (batch_size, self.vec_env.n_channels, self.vec_env.view_size, self.vec_env.view_size),
            device=self.device, requires_grad=False
        )

        with autocast(device_type=self.device_type, enabled=cfg.use_amp):
            logits = self.policy(dummy_obs)
            features = self.policy.get_features(dummy_obs)
            values = self.value_head(features).squeeze(-1)

            if include_backward:
                dummy_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                dummy_returns = torch.zeros(batch_size, device=self.device)

                dist = Categorical(logits=logits, validate_args=False)
                log_probs = dist.log_prob(dummy_actions)
                dummy_loss = -log_probs.mean() + F.mse_loss(values, dummy_returns)
                dummy_loss.backward()

        # Synchronize to ensure kernels complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Clean up
        if include_backward:
            self.policy.zero_grad()
            self.value_head.zero_grad()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return time.time() - start_time

    def _run_warmup_update(self, states: torch.Tensor, episode_rewards: torch.Tensor) -> torch.Tensor:
        """
        Run one full training update as warmup (discarded).

        This triggers all lazy initialization in a realistic context:
        - Real environment steps (not synthetic data)
        - Full forward/backward through compiled models
        - All MIOpen/cuDNN algorithm selection

        The results are discarded; this just warms up the runtime.

        Args:
            states: Current environment states
            episode_rewards: Episode reward accumulator

        Returns:
            New states after the warmup steps
        """
        cfg = self.config

        # Use pre-allocated buffers (same as main training loop)
        states_buf = self._states_buf
        actions_buf = self._actions_buf
        log_probs_buf = self._log_probs_buf
        rewards_buf = self._rewards_buf
        dones_buf = self._dones_buf
        terminated_buf = self._terminated_buf
        values_buf = self._values_buf

        # Initial inference to get first actions (same as main loop)
        potentials = self.reward_computer.get_initial_potentials(states)
        critic_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None
        with torch.no_grad():
            with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                logits, features = self.policy.forward_with_features(states.float())
                dist = Categorical(logits=logits, validate_args=False)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.value_head(features, critic_aux).squeeze(-1)

        for step in range(cfg.steps_per_env):
            with torch.no_grad():
                # Use compiled rollout step if available (triggers CUDA graph capture)
                if self._compiled_rollout_step is not None:
                    torch.compiler.cudagraph_mark_step_begin()
                    (
                        next_states, next_actions, next_log_probs, next_values,
                        next_potentials, logits,
                        _current_states, _shaped_rewards, _dones, _terminated, _critic_aux,
                        _finished_episode_rewards
                    ) = self._compiled_rollout_step(
                        actions, log_probs, values, states, potentials
                    )
                    # Warmup doesn't need to write buffers, just exercise the compiled path

                    # Clone outputs to prevent CUDA graph buffer reuse issues
                    states = next_states.clone()
                    actions = next_actions.clone()
                    log_probs = next_log_probs.clone()
                    values = next_values.clone()
                    potentials = next_potentials.clone()
                else:
                    # Fallback: eager mode
                    critic_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None
                    with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                        logits, features = self.policy.forward_with_features(states.float())
                        dist = Categorical(logits=logits, validate_args=False)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)
                        values = self.value_head(features, critic_aux).squeeze(-1)

                    current_states = states.clone()
                    next_states, eating_info, terminated, truncated = self.vec_env.step(actions)
                    dones = terminated | truncated
                    shaped_rewards = self.reward_computer.compute(eating_info, current_states, next_states, terminated)

                    # Store in pre-allocated tensor buffers
                    states_buf[step] = current_states
                    actions_buf[step] = actions
                    log_probs_buf[step] = log_probs.detach()
                    rewards_buf[step] = shaped_rewards
                    dones_buf[step] = dones
                    terminated_buf[step] = terminated
                    values_buf[step] = values
                    if self._aux_buf is not None:
                        self._aux_buf[step] = critic_aux

                    episode_rewards += shaped_rewards
                    episode_rewards *= (~dones)
                    states = next_states

        # Bootstrap value
        with torch.no_grad():
            states_t = states.float()
            _, features = self.policy.forward_with_features(states_t)
            bootstrap_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None
            next_value = self.value_head(features, bootstrap_aux).squeeze(-1)

        # Compute GAE (pass tensors directly, no stacking needed)
        # Use terminated_buf - only zero bootstrap on true death
        gae_fn = self._compiled_gae if self._compiled_gae is not None else compute_gae
        advantages, returns = gae_fn(
            rewards_buf, values_buf, terminated_buf,
            next_value, cfg.gamma, cfg.gae_lambda, device=self.device
        )

        # Update PopArt statistics for value normalization
        # Pass optimizer so PopArt can rescale momentum buffers after weight adjustment
        if hasattr(self.value_head, 'update_stats'):
            self.value_head.update_stats(returns.flatten(), self.optimizer)

        # PPO update (this triggers backward pass lazy init)
        # Reshape pre-allocated buffers (no torch.cat needed)
        batch_size = cfg.steps_per_env * cfg.n_envs
        all_states = states_buf.reshape(batch_size, -1, self.vec_env.view_size, self.vec_env.view_size).float()
        all_actions = actions_buf.flatten()
        all_log_probs = log_probs_buf.flatten()
        all_values = values_buf.flatten()
        all_returns = returns.flatten()
        all_advantages = advantages.flatten()

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Use compiled version if available (triggers JIT compilation during warmup)
        # IMPORTANT: Must pass aux_inputs with same shape as training to avoid recompilation
        all_aux = self._aux_buf.reshape(batch_size, -1) if self._aux_buf is not None else None
        ppo_fn = self._compiled_ppo_update if self._compiled_ppo_update is not None else ppo_update
        ppo_fn(
            self.policy, self.value_head, self.optimizer,
            all_states, all_actions, all_log_probs,
            all_returns, all_advantages, all_values,
            self.device,
            eps_clip=cfg.eps_clip,
            k_epochs=cfg.k_epochs,
            entropy_coef=cfg.entropy_initial,  # Use initial entropy for warmup
            value_coef=cfg.value_coef,
            n_minibatches=cfg.n_minibatches,
            scaler=self.scaler,
            aux_inputs=all_aux,
            entropy_floor=cfg.entropy_floor,  # Floor active during warmup
            entropy_floor_penalty=cfg.entropy_floor_penalty,
        )

        # Sync to ensure all kernels complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        return states

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Summary dict with training results
        """
        self._setup()

        try:
            self._training_loop()
        except KeyboardInterrupt:
            self._aborted = True
            print(f"\n[{self.config.mode}] Training interrupted by user")
        except Exception as e:
            self._aborted = True
            print(f"\n[{self.config.mode}] Training failed: {e}")
            self.cleanup()
            raise

        # Check if abort was requested globally (e.g., by signal handler)
        if is_abort_requested():
            self._aborted = True

        return self._finalize()

    def cleanup(self):
        """
        Release all resources.

        Call this after training completes or on abort to ensure
        GPU memory, threads, and file handles are properly released.
        """
        # Stop GPU monitor first
        if self.gpu_monitor is not None:
            try:
                self.gpu_monitor.stop()
            except Exception as e:
                logger.debug(f"Cleanup: gpu_monitor.stop() failed: {e}")
            self.gpu_monitor = None

        # Stop bookkeeper before async_logger (bookkeeper submits to async_logger)
        if hasattr(self, 'bookkeeper') and self.bookkeeper is not None:
            try:
                self.bookkeeper.stop()
            except Exception as e:
                logger.debug(f"Cleanup: bookkeeper.stop() failed: {e}")
            self.bookkeeper = None

        # Shutdown async logger (flushes any pending logs)
        if self.async_logger:
            try:
                self.async_logger.shutdown(timeout=2.0)
            except Exception as e:
                logger.debug(f"Cleanup: async_logger.shutdown() failed: {e}")
            self.async_logger = None

        # Close TensorBoard writer
        if self.tb_writer:
            try:
                self.tb_writer.close()
            except Exception as e:
                logger.debug(f"Cleanup: tb_writer.close() failed: {e}")
            self.tb_writer = None

        # Release environment (frees GPU tensors)
        if self.vec_env:
            try:
                # VecEnv doesn't have an explicit close(), but we can
                # release references to allow garbage collection
                del self.vec_env
            except Exception as e:
                logger.debug(f"Cleanup: vec_env deletion failed: {e}")
            self.vec_env = None

        # Release model tensors
        # Note: Can't use truthiness check on torch.compile wrapped modules
        # (they don't support __len__), so check against None explicitly
        if self.policy is not None:
            del self.policy
            self.policy = None
        if self.value_head is not None:
            del self.value_head
            self.value_head = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None

        # Clear GPU cache
        if self.device and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"Cleanup: torch.cuda.empty_cache() failed: {e}")
    
    def _setup(self):
        """Initialize environment, networks, and logging."""
        cfg = self.config

        # Reproducibility: set all random seeds
        self.seed = set_seed(
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            verbose=False
        )

        v = cfg.hyper_verbose
        _vprint("_setup() starting", v)

        print(f"\n[PPO] Starting training: {cfg.mode}")
        print(f"   Device: {self.device}, Envs: {cfg.n_envs}, Seed: {self.seed}")
        if is_tpu(self.device):
            print("   Note: TPU uses XLA - first step compiles the graph (may take a few minutes)")
        
        # Apply hardware optimizations
        _vprint("Applying system optimizations...", v)
        apply_system_optimizations(self.device, verbose=True)
        
        # Load configs
        _vprint("Loading configs...", v)
        sim_config = get_simulation_config()
        train_cfg = get_training_config()
        
        # Observation spec
        _vprint("Creating observation spec...", v)
        self.spec = sim_config['get_observation_spec'](cfg.mode)

        # Action space (pluggable - stored for serialization)
        self.action_space = create_action_space(
            cfg.action_space_type,
            max_move_distance=cfg.max_move_distance
        )
        n_actions = self.action_space.n_outputs
        
        # Environment - ALWAYS use GPU-native TorchVecEnv
        _vprint("Creating TorchVecEnv...", v)
        self.vec_env = create_torch_vec_env(n_envs=cfg.n_envs, obs_spec=self.spec, device=self.device)
        env_type = "TorchVecEnv (TPU-native)" if is_tpu(self.device) else "TorchVecEnv (GPU-native)"
        print(f"   Env: {env_type}")
        _vprint("Environment created", v)
        print(f"   View: {self.vec_env.view_size}x{self.vec_env.view_size}, Channels: {self.vec_env.n_channels}")

        # Compile environment step for better GPU utilization (reduces CPU dispatch overhead)
        if cfg.compile_env:
            _vprint("Compiling environment step...", v)
            self.vec_env.compile_step(mode=cfg.compile_mode, fullgraph=True)
            print(f"   Env compile: {cfg.compile_mode} (fullgraph=True)")
        
        # Networks
        _vprint("Creating networks...", v)
        self.policy = create_brain(cfg.brain_type, self.spec, output_size=n_actions).to(self.device)

        # Value head - configurable between simple and PopArt
        # Privileged critic: value head sees ground truth view + density that policy may not see
        # For proxy modes, this gives the critic info the policy can't access
        num_aux = self.vec_env.num_critic_aux if cfg.privileged_critic else 0
        if cfg.value_head_type == 'popart':
            self.value_head = PopArtValueHead(
                input_size=self.policy.hidden_size,
                num_aux_inputs=num_aux,
                beta_min=cfg.popart_beta_min,
                rescale_weights=cfg.popart_rescale_weights,
            ).to(self.device)
        elif cfg.value_head_type == 'simple':
            self.value_head = ValueHead(
                input_size=self.policy.hidden_size,
                num_aux_inputs=num_aux
            ).to(self.device)
        else:
            raise ValueError(f"Unknown value_head_type: {cfg.value_head_type}. Use 'simple' or 'popart'.")
        self._num_aux_inputs = num_aux
        # Store architecture info before potential torch.compile (for serialization)
        self._policy_arch_info = self.policy.get_architecture_info()
        _vprint("Networks created and moved to device", v)

        # AMP
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(enabled=cfg.use_amp) if cfg.use_amp else None

        # Compiled function placeholders - actual compilation happens after
        # all components (including reward_computer) are created
        self._compiled_inference = None
        self._compiled_rollout_step = None
        self._compiled_ppo_update = None
        self._compiled_gae = None

        # Pre-allocated constants for BUFFER_FLATTEN (avoids tensor creation each update)
        self._inf_const = torch.tensor(float('inf'), device=self.device)
        self._neg_inf_const = torch.tensor(float('-inf'), device=self.device)

        # Optimizer - use fused=True on CUDA to eliminate .item() sync overhead
        # Fused runs entirely on GPU, avoiding 384 CPU round-trips per update
        use_fused = self.device.type == 'cuda'
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr,
            fused=use_fused
        )

        # Reward computer - unified class handles both numpy and torch
        # Shaping is now handled internally by each RewardComputer subclass
        self.reward_computer = RewardComputer.create(
            cfg.mode, self.spec, sim_config,
            gamma=cfg.gamma,
            device=self.device
        )

        # Pre-allocate rollout buffers (eliminates torch.cat burstiness)
        # These are GPU tensors that get overwritten each update cycle
        # MUST be allocated before compile block since closures capture these references
        obs_shape = (cfg.steps_per_env, cfg.n_envs,
                     self.vec_env.n_channels, self.vec_env.view_size, self.vec_env.view_size)
        scalar_shape = (cfg.steps_per_env, cfg.n_envs)

        self._states_buf = torch.zeros(obs_shape, device=self.device, dtype=torch.uint8)
        self._actions_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.long)
        self._log_probs_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)
        self._rewards_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)
        self._dones_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.bool)
        self._terminated_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.bool)
        self._values_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)

        # Privileged critic: auxiliary info buffer (density + ground truth view per step)
        if self._num_aux_inputs > 0:
            aux_shape = (cfg.steps_per_env, cfg.n_envs, self._num_aux_inputs)
            self._aux_buf = torch.zeros(aux_shape, device=self.device, dtype=torch.float32)
        else:
            self._aux_buf = None

        # Episode tracking buffers (for async logging)
        self._finished_dones_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.bool)
        self._finished_rewards_buf = torch.zeros(scalar_shape, device=self.device, dtype=torch.float32)

        # Note: step_idx tensor removed - buffer writes now use Python int index
        # outside the compiled function to avoid implicit .item() calls

        # Episode rewards accumulator lives in TorchVecEnv (nn.Module buffer for CUDA graph stability)
        # Accessed via vec_env.episode_rewards

        # Compile models for extra speed if torch.compile is available (PyTorch 2.0+)
        # Use lock to serialize compilation - Dynamo's global state is not thread-safe
        # Skip on TPU - XLA uses its own JIT compilation
        # Note: Actual warmup happens in _training_loop via _run_warmup_update()
        #
        # Key insight: Compiling ENV_STEP + REWARD_SHAPE + INFERENCE into ONE function
        # eliminates graph boundaries between all three operations. This fuses many
        # small kernels into fewer large ones, reducing kernel launch overhead.
        with get_compile_lock():
            if cfg.compile_models and hasattr(torch, 'compile') and not is_tpu(self.device):
                try:
                    compile_mode = cfg.compile_mode

                    # Capture references for closure - all must exist before this point
                    policy = self.policy
                    value_head = self.value_head
                    vec_env = self.vec_env
                    reward_computer = self.reward_computer
                    device_type = self.device_type
                    use_amp = cfg.use_amp
                    privileged_critic = cfg.privileged_critic

                    # Capture buffer references for fused buffer storage
                    states_buf = self._states_buf
                    actions_buf = self._actions_buf
                    log_probs_buf = self._log_probs_buf
                    rewards_buf = self._rewards_buf
                    dones_buf = self._dones_buf
                    terminated_buf = self._terminated_buf
                    values_buf = self._values_buf
                    aux_buf = self._aux_buf
                    finished_dones_buf = self._finished_dones_buf
                    finished_rewards_buf = self._finished_rewards_buf
                    # NOTE: episode_rewards must be accessed through vec_env (not captured)
                    # to maintain nn.Module buffer semantics for CUDA graph compatibility
                    # NOTE: Buffer writes done outside compiled function with Python int index

                    # ============================================================
                    # FUSED ROLLOUT STEP: ENV + REWARD + INFERENCE + BUFFER + TRACK
                    # ============================================================
                    # All operations compiled into ONE graph for maximum fusion.
                    # This eliminates CPU dispatch overhead between operations.
                    @torch.compile(mode=compile_mode)
                    def compiled_rollout_step(actions, log_probs, values, states, potentials):
                        """
                        Fully fused rollout step with buffer storage and episode tracking.

                        Args:
                            actions: Current actions to execute (also stored to buffer)
                            log_probs: Current log probs (stored to buffer)
                            values: Current values (stored to buffer)
                            states: Current observations (before step)
                            potentials: Current potential values for reward shaping

                        Returns:
                            Tuple for next iteration:
                            - next_states, next_actions, next_log_probs, next_values
                            - next_potentials, logits (for action prob logging)
                        """
                        # Snapshot current states before env mutates them
                        current_states = states.clone()

                        # ENV_STEP
                        next_states, eating_info, terminated, truncated = vec_env.step(actions)
                        dones = terminated | truncated

                        # Get critic aux (density + privileged view) for value function
                        critic_aux = vec_env.get_critic_aux() if privileged_critic else None

                        # REWARD_SHAPE (stateless)
                        shaped_rewards, next_potentials = reward_computer.compute_stateless(
                            eating_info, current_states, next_states, terminated, potentials
                        )

                        # INFERENCE (for next step)
                        with autocast(device_type=device_type, enabled=use_amp):
                            logits, features = policy.forward_with_features(next_states)
                            dist = Categorical(logits=logits, validate_args=False)
                            next_actions = dist.sample()
                            next_log_probs = dist.log_prob(next_actions)
                            next_values = value_head(features, critic_aux).squeeze(-1)

                        # EPISODE_TRACK - accumulate rewards, reset on done
                        # Access through vec_env (not captured) to maintain nn.Module buffer semantics
                        vec_env.episode_rewards.add_(shaped_rewards)
                        # Capture episode rewards BEFORE reset (for logging finished episodes)
                        finished_episode_rewards = vec_env.episode_rewards.clone()
                        vec_env.episode_rewards.mul_(~dones)  # Reset for done agents

                        # Return all values needed for buffer writes (done outside with Python int index)
                        # NOTE: Buffer writes with tensor indices cause implicit .item() calls,
                        # breaking the graph into multiple regions. We return values instead.
                        return (
                            next_states, next_actions, next_log_probs, next_values,
                            next_potentials, logits,
                            # Additional returns for buffer storage (written outside compiled function)
                            current_states, shaped_rewards, dones, terminated, critic_aux,
                            finished_episode_rewards  # Pre-reset rewards for logging
                        )

                    self._compiled_rollout_step = compiled_rollout_step

                    # NOTE: We intentionally do NOT compile a separate inference function.
                    # Having two compiled functions (compiled_rollout_step + compiled_inference)
                    # that share the same model weights causes segfaults on ROCm when weights
                    # are modified between calls (PPO update). The prefetch uses eager mode,
                    # which is fine since it's only 1 call per update vs 128 for the rollout.

                    # Compile PPO update function
                    self._compiled_ppo_update = torch.compile(ppo_update, mode=compile_mode)

                    # Compile GAE computation (27x speedup from loop optimization)
                    # Use reduce-overhead for better CUDA graph capture of the loop
                    self._compiled_gae = torch.compile(compute_gae, mode=compile_mode)

                    print(f"   [JIT] torch.compile enabled ({compile_mode}) - fused rollout step + GAE")
                except RuntimeError as e:
                    if "FX" in str(e) or "dynamo" in str(e).lower():
                        print(f"   [JIT] Warning: torch.compile failed ({e}). Using eager mode.")
                    else:
                        raise e

        # TensorBoard logging (always enabled, unified logging target)
        self.tb_writer = None
        if cfg.log_to_file and not cfg.benchmark_mode:
            try:
                from torch.utils.tensorboard import SummaryWriter
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tb_dir = os.path.join(cfg.log_dir, f"{cfg.mode}_{timestamp}")
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                self.tb_log_dir = tb_dir
                print(f"   Logging: {tb_dir}")
            except ImportError:
                print("   Warning: TensorBoard not available (install tensorboard package)")
        
        # Profiler (disable with --no-profile to remove GPU sync overhead)
        self.profiler = Profiler(self.device, enabled=cfg.profile_enabled)
        
        # Async logger - handles all I/O in background thread to avoid GPU stalls
        self.async_logger = AsyncLogger(
            tb_writer=self.tb_writer,
            dashboard=self.dashboard,
            mode=cfg.mode
        )
        self.async_logger.start()

        # Background bookkeeper - processes metrics in parallel with GPU work
        # Uses double-buffered pinned memory for zero-copy submission
        from goodharts.behaviors.action_space import num_actions
        n_actions = num_actions(cfg.max_move_distance)
        buffer_size = N_SCALAR_METRICS + n_actions  # Schema metrics + action probs
        self.bookkeeper = BackgroundBookkeeper(
            buffer_size=buffer_size,
            async_logger=self.async_logger,
            device=self.device,
        )
        self.bookkeeper.start()
        
        # Curriculum settings - inform VecEnv of ranges (each env randomizes on reset)
        self.min_food = train_cfg.get('min_food', 50)
        self.max_food = train_cfg.get('max_food', 200)
        self.min_poison = train_cfg.get('min_poison', 20)
        self.max_poison = train_cfg.get('max_poison', 100)
        self.vec_env.set_curriculum_ranges(
            self.min_food, self.max_food, 
            self.min_poison, self.max_poison
        )
        # Force reset to apply new ranges
        self.vec_env.reset()
        
        # Checkpoint settings
        self.checkpoint_interval = train_cfg.get('checkpoint_interval', 0)
        self.checkpoint_dir = os.path.dirname(cfg.output_path) or 'models'

        # Async metrics pipeline: GPU transfers to pinned memory while continuing work
        # We process PREVIOUS update's metrics while GPU runs NEXT update
        # 18 = 6 episode aggregates + 4 PPO metrics + 8 action probs (max)
        self._metrics_pinned = torch.zeros(32, dtype=torch.float32, pin_memory=True)
        self._metrics_event = None  # CUDA event for pending transfer
        self._pending_log_data = None  # Metadata for pending metrics

        print(f"   Brain: {cfg.brain_type} (hidden={self.policy.hidden_size})")
        print(f"   Value head: {cfg.value_head_type}")
        print(f"   AMP: {'Enabled' if cfg.use_amp else 'Disabled'}")
    
    def _training_loop(self):
        """Main training loop."""
        cfg = self.config
        v = cfg.hyper_verbose  # Verbose debug mode

        # Use pre-allocated GPU tensor buffers (from _setup)
        # These eliminate torch.cat overhead that caused GPU burstiness
        # For non-compiled path, create local refs; compiled path uses closures
        states_buf = self._states_buf
        actions_buf = self._actions_buf
        log_probs_buf = self._log_probs_buf
        rewards_buf = self._rewards_buf
        dones_buf = self._dones_buf
        terminated_buf = self._terminated_buf
        values_buf = self._values_buf
        finished_dones_buf = self._finished_dones_buf
        finished_rewards_buf = self._finished_rewards_buf

        # Step index for buffer writes (Python int - no GPU sync needed)
        step_in_buffer = 0

        # Episode rewards accumulator (nn.Module buffer in TorchVecEnv for CUDA graph stability)
        episode_rewards = self.vec_env.episode_rewards
        episode_rewards.zero_()

        # Initial state
        _vprint("Resetting environment for initial state...", v)
        states = self.vec_env.reset()
        _vprint("Environment reset done", v)

        # Debug: verify observation encoding (once at startup)
        if v:
            _vprint(f"Observation mode: is_proxy_mode={self.vec_env.is_proxy_mode}", v)
            _vprint(f"Channel names: {self.vec_env.channel_names}", v)
            sample_obs = states[0]  # First agent's observation
            _vprint(f"Observation shape: {sample_obs.shape}", v)
            _vprint(f"Observation range: min={sample_obs.min():.3f}, max={sample_obs.max():.3f}", v)
            if sample_obs.max() > 0:
                unique_vals = sample_obs.unique().tolist()
                _vprint(f"Unique values in obs: {unique_vals[:10]}", v)

        # Initialize reward computer
        _vprint("Initializing reward computer...", v)
        self.reward_computer.initialize(states)  # stays on GPU
        _vprint("Training loop ready to start", v)

        # === WARMUP UPDATE (lazy init) ===
        # Run one full update cycle to trigger all lazy initialization:
        # - cuDNN/MIOpen algorithm selection
        # - Autograd graph construction
        # - Any remaining JIT compilation
        # This is discarded; real training starts fresh after.
        # For sequential/parallel training, warmup only runs once.
        should_warmup = not cfg.skip_warmup

        with get_warmup_lock():
            if check_warmup_done():
                should_warmup = False
                print(f"   [Warmup] Skipped (already done)", flush=True)

        if should_warmup:
            print(f"   [Warmup] Running warmup update (lazy init)...", flush=True)
            warmup_start = time.perf_counter()

            # Save model state before warmup (will be restored after)
            policy_state = {k: v.clone() for k, v in self.policy.state_dict().items()}
            value_head_state = {k: v.clone() for k, v in self.value_head.state_dict().items()}
            optimizer_state = self.optimizer.state_dict()

            # Run warmup (modifies model, but we'll restore)
            states = self._run_warmup_update(states, episode_rewards)

            # Restore model state (discard warmup training)
            self.policy.load_state_dict(policy_state)
            self.value_head.load_state_dict(value_head_state)
            self.optimizer.load_state_dict(optimizer_state)

            warmup_elapsed = time.perf_counter() - warmup_start
            print(f"   [Warmup] Complete ({warmup_elapsed:.1f}s) - weights restored", flush=True)

            # Mark warmup as done globally
            mark_warmup_done()

            # Reset environment state for clean start
            states = self.vec_env.reset()
            self.reward_computer.initialize(states)
            episode_rewards.zero_()

        # Start GPU monitor after warmup (if enabled)
        if cfg.gpu_log_interval_ms > 0:
            self.gpu_monitor = GPUMonitor(
                interval_ms=cfg.gpu_log_interval_ms,
                output_path="gpu_utilization.csv"
            )
            self.gpu_monitor.start()

        self.start_time = time.perf_counter()

        # Rolling window for sps calculation (all updates valid post-warmup)
        sps_window = []  # (steps, time) pairs for last 4 updates
        last_update_time = self.start_time
        last_update_steps = 0

        # ============================================================
        # STEP-THEN-INFER PATTERN
        # ============================================================
        # Unlike traditional Infer-Then-Step, we run inference AFTER env.step()
        # to get next step's actions. This makes ENV_STEP → REWARD_SHAPE → INFERENCE
        # adjacent and fusable into ONE compiled graph.
        #
        # Initial inference: get first actions before entering the loop.
        # After PPO updates weights, needs_initial_inference triggers fresh inference.
        # ============================================================

        # Initial potentials for reward shaping
        potentials = self.reward_computer.get_initial_potentials(states)

        # Initial inference to get first actions (eager mode - only runs once)
        critic_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None
        with torch.no_grad():
            with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                logits, features = self.policy.forward_with_features(states.float())
                dist = Categorical(logits=logits, validate_args=False)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.value_head(features, critic_aux).squeeze(-1)
        last_logits = logits

        # Flag to trigger fresh inference after PPO update
        needs_initial_inference = False

        while self.total_steps < cfg.total_timesteps:
            self.profiler.start()
            
            # Check for abort (only at start of each UPDATE, not every step)
            # This reduces multiprocessing Event checks from 128x to 1x per update
            if step_in_buffer == 0:
                if is_abort_requested():
                    print(f"\n[{cfg.mode}] Abort signal received")
                    break
                # Check stop signal via dashboard (if available)
                if self.dashboard and hasattr(self.dashboard, 'should_stop') and self.dashboard.should_stop():
                    print(f"\n[{cfg.mode}] Dashboard stop requested")
                    break
            
            if v:
                _vprint(f"Step {self.total_steps}: collecting experience...", v)

            # ============================================================
            # STEP-THEN-INFER: Use current actions, then infer next ones
            # ============================================================
            # At this point we have: states, actions, log_probs, values, potentials
            # (from initial inference or previous iteration's inference)

            # After PPO update, we need fresh inference with updated weights
            # This runs once per rollout (~0.8% overhead vs prefetch complexity)
            if needs_initial_inference:
                with torch.no_grad():
                    critic_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None
                    with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                        logits, features = self.policy.forward_with_features(states.float())
                        dist = Categorical(logits=logits, validate_args=False)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)
                        values = self.value_head(features, critic_aux).squeeze(-1)
                    potentials = self.reward_computer.get_initial_potentials(states)
                needs_initial_inference = False

            # ============================================================
            # FUSED ROLLOUT STEP: ENV + REWARD + INFERENCE + BUFFER + TRACK
            # ============================================================
            # All operations compiled into ONE graph for maximum fusion.
            with torch.no_grad():
                if self._compiled_rollout_step is not None:
                    with record_function("ROLLOUT_STEP"):
                        torch.compiler.cudagraph_mark_step_begin()
                        # Compiled function returns values; we write buffers with Python int index
                        (
                            next_states, next_actions, next_log_probs, next_values,
                            next_potentials, logits,
                            current_states, shaped_rewards, dones, terminated, critic_aux,
                            finished_episode_rewards  # Pre-reset rewards for logging
                        ) = self._compiled_rollout_step(
                            actions, log_probs, values, states, potentials
                        )

                    # BUFFER_STORE - use Python int index (no GPU sync)
                    # Done outside compiled function to avoid .item() calls on tensor indices
                    step_i = step_in_buffer
                    states_buf[step_i] = current_states
                    actions_buf[step_i] = actions
                    log_probs_buf[step_i] = log_probs
                    rewards_buf[step_i] = shaped_rewards
                    dones_buf[step_i] = dones
                    terminated_buf[step_i] = terminated
                    values_buf[step_i] = values
                    if self._aux_buf is not None:
                        self._aux_buf[step_i] = critic_aux
                    finished_dones_buf[step_i] = dones
                    finished_rewards_buf[step_i] = finished_episode_rewards  # Use pre-reset value

                    step_in_buffer += 1
                else:
                    # Fallback to separate calls (eager mode)
                    with record_function("ENV_STEP"):
                        current_states = states.clone()
                        next_states, eating_info, terminated, truncated = self.vec_env.step(actions)
                        dones = terminated | truncated

                    with record_function("REWARD_SHAPE"):
                        shaped_rewards, next_potentials = self.reward_computer.compute_stateless(
                            eating_info, current_states, next_states, terminated, potentials
                        )

                    next_critic_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None

                    with record_function("INFERENCE"):
                        with autocast(device_type=self.device_type, enabled=cfg.use_amp):
                            logits, features = self.policy.forward_with_features(next_states.float())
                            dist = Categorical(logits=logits, validate_args=False)
                            next_actions = dist.sample()
                            next_log_probs = dist.log_prob(next_actions)
                            next_values = self.value_head(features, next_critic_aux).squeeze(-1)

                    # EPISODE_TRACK (eager path only) - must happen BEFORE buffer store
                    # so finished_episode_rewards captures the final value before reset
                    with record_function("EPISODE_TRACK"):
                        episode_rewards += shaped_rewards
                        finished_episode_rewards = episode_rewards.clone()  # Capture before reset
                        episode_rewards *= (~dones)

                    # BUFFER_STORE (eager path only - compiled path does this inside)
                    with record_function("BUFFER_STORE"):
                        step_i = step_in_buffer  # Use Python int - no GPU sync
                        states_buf[step_i] = current_states
                        actions_buf[step_i] = actions
                        log_probs_buf[step_i] = log_probs.detach()
                        rewards_buf[step_i] = shaped_rewards
                        dones_buf[step_i] = dones
                        terminated_buf[step_i] = terminated
                        values_buf[step_i] = values
                        if self._aux_buf is not None:
                            self._aux_buf[step_i] = next_critic_aux
                        finished_dones_buf[step_i] = dones
                        finished_rewards_buf[step_i] = finished_episode_rewards  # Use pre-reset value

                    step_in_buffer += 1

            # Store logits for action_probs logging
            last_logits = logits
            self.profiler.tick("Inference")
            self.total_steps += cfg.n_envs

            # Carry forward for next iteration
            # Clone outputs from compiled function to prevent CUDA graph buffer reuse issues
            # (CUDA graphs reuse output memory; without clone, next call overwrites these)
            if self._compiled_rollout_step is not None:
                states = next_states.clone()
                actions = next_actions.clone()
                log_probs = next_log_probs.clone()
                values = next_values.clone()
                potentials = next_potentials.clone()
            else:
                states = next_states
                actions = next_actions
                log_probs = next_log_probs
                values = next_values
                potentials = next_potentials

            # PPO Update (when buffer is full)
            if step_in_buffer >= cfg.steps_per_env:
                self.profiler.tick("Collection")

                # Get bootstrap value (keep on GPU)
                with record_function("GAE_COMPUTE"):
                    with torch.no_grad():
                        states_t = states.float()
                        _, features = self.policy.forward_with_features(states_t)
                        # Get current critic aux for bootstrap value (privileged critic)
                        bootstrap_aux = self.vec_env.get_critic_aux() if self._aux_buf is not None else None
                        next_value = self.value_head(features, bootstrap_aux).squeeze(-1)

                    # Compute GAE (pass pre-allocated tensors directly, no stacking)
                    # Use terminated_buf (not dones_buf) - only zero bootstrap on true death
                    gae_fn = self._compiled_gae if self._compiled_gae is not None else compute_gae
                    advantages, returns = gae_fn(
                        rewards_buf, values_buf, terminated_buf,
                        next_value, cfg.gamma, cfg.gae_lambda, device=self.device
                    )

                    # Update PopArt statistics for value normalization
                    # Pass optimizer so PopArt can rescale momentum buffers after weight adjustment
                    if hasattr(self.value_head, 'update_stats'):
                        self.value_head.update_stats(returns.flatten(), self.optimizer)

                self.profiler.tick("GAE Calc")

                # Debug: sample reward statistics (verbose mode only)
                if v and self.update_count % 50 == 0:
                    _vprint(f"Reward buffer stats: mean={rewards_buf.mean():.4f}, std={rewards_buf.std():.4f}, "
                           f"min={rewards_buf.min():.4f}, max={rewards_buf.max():.4f}", v)
                    _vprint(f"Value buffer stats: mean={values_buf.mean():.4f}, std={values_buf.std():.4f}", v)
                    _vprint(f"Returns stats: mean={returns.mean():.4f}, std={returns.std():.4f}", v)

                # EPISODE LOGGING - Fixed-size GPU aggregates, NO sync until after PPO
                with record_function("BUFFER_FLATTEN"):
                    # Pre-allocated buffers are already (steps, envs) tensors
                    all_step_dones = finished_dones_buf  # Already (steps, envs)
                    all_step_rewards = finished_rewards_buf  # Already (steps, envs)

                    # Masked aggregates - all ops stay on GPU
                    # Use pre-allocated inf constants (avoids tensor creation overhead)
                    done_mask = all_step_dones.float()

                    # Food/poison: sum for envs that finished at least once this update
                    any_done_per_env = all_step_dones.any(dim=0)  # (envs,)

                    # Episode aggregates - order must match metrics_keys in sync section
                    # Use pre-allocated constants and stack into tensor
                    episode_agg_tensor = torch.stack([
                        done_mask.sum(),
                        (all_step_rewards * done_mask).sum(),
                        torch.where(all_step_dones, all_step_rewards, self._inf_const).min(),
                        torch.where(all_step_dones, all_step_rewards, self._neg_inf_const).max(),
                        (self.vec_env.last_episode_food * any_done_per_env).sum().float(),
                        (self.vec_env.last_episode_poison * any_done_per_env).sum().float(),
                    ])

                    # Reshape pre-allocated buffers (no torch.cat needed!)
                    # states_buf is (steps, envs, C, H, W) -> flatten to (steps*envs, C, H, W)
                    batch_size = cfg.steps_per_env * cfg.n_envs
                    all_states = states_buf.reshape(batch_size, -1, self.vec_env.view_size, self.vec_env.view_size).float()
                    all_actions = actions_buf.flatten()
                    all_log_probs = log_probs_buf.flatten()
                    all_old_values = values_buf.flatten()

                    # Privileged critic: flatten aux buffer for PPO update
                    all_aux = None
                    if self._aux_buf is not None:
                        all_aux = self._aux_buf.reshape(batch_size, -1)

                    # Returns and advantages from compute_gae
                    all_returns = returns.flatten()
                    all_advantages = advantages.flatten()

                    if v:
                        _vprint(f"   [DEBUG] Rewards: mean={rewards_buf.mean():.4f}, std={rewards_buf.std():.4f}", v)
                        _vprint(f"   [DEBUG] Advantages: mean={all_advantages.mean():.4f}, std={all_advantages.std():.4f}", v)
                        _vprint(f"   [DEBUG] Returns: mean={all_returns.mean():.4f}, std={all_returns.std():.4f}", v)

                    # Normalize advantages
                    all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

                # Compute dynamic entropy coefficient and floor
                # Both decay continuously from initial to final over full training
                progress = self.total_steps / cfg.total_timesteps

                # Entropy coef: decays over decay_fraction, then stays at final
                coef_progress = min(1.0, progress / cfg.entropy_decay_fraction)
                current_entropy_coef = (
                    cfg.entropy_initial * (1 - coef_progress) +
                    cfg.entropy_final * coef_progress
                )

                # Entropy floor: constant during learning phase, then decays
                # Phase 1 (0 to decay_fraction): constant floor for stability while learning
                # Phase 2 (decay_fraction to 1): floor decays to 0, allowing determinism
                # Special case: decay_fraction >= 1.0 means no decay (constant floor)
                if cfg.entropy_decay_fraction >= 1.0 or progress < cfg.entropy_decay_fraction:
                    current_entropy_floor = cfg.entropy_floor
                else:
                    # Linear decay from entropy_floor to 0 over remaining training
                    floor_progress = (progress - cfg.entropy_decay_fraction) / (1 - cfg.entropy_decay_fraction)
                    current_entropy_floor = cfg.entropy_floor * (1 - floor_progress)

                # LR decay: reduce learning rate over training for fine-tuning
                if cfg.lr_decay:
                    current_lr = cfg.lr * (1 - progress) + cfg.lr_final * progress
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr

                # Clip decay: tighten trust region over training to reduce oscillation
                current_eps_clip = cfg.eps_clip * (1 - progress) + cfg.eps_clip_final * progress

                # PPO update - tensors go in directly, no CPU sync needed
                # Use compiled version when available (~9% faster)
                with record_function("PPO_UPDATE"):
                    ppo_fn = self._compiled_ppo_update if self._compiled_ppo_update is not None else ppo_update
                    policy_loss, value_loss, entropy, explained_var = ppo_fn(
                        self.policy, self.value_head, self.optimizer,
                        all_states, all_actions, all_log_probs,
                        all_returns, all_advantages, all_old_values,
                        self.device,
                        eps_clip=current_eps_clip,
                        k_epochs=cfg.k_epochs,
                        entropy_coef=current_entropy_coef,
                        value_coef=cfg.value_coef,
                        n_minibatches=cfg.n_minibatches,
                        scaler=self.scaler,
                        verbose=True,
                        aux_inputs=all_aux,
                        entropy_floor=current_entropy_floor,
                        entropy_floor_penalty=cfg.entropy_floor_penalty,
                    )
                # Clear progress line
                print(" " * 40, end='\r')

                # Sync device only for TPU (forces XLA execution)
                # CUDA profiling is now async - sync happens in background thread
                if is_tpu(self.device):
                    sync_device(self.device)
                self.profiler.tick("PPO Update")

                self.update_count += 1

                # External profiler callback (for --profile-trace)
                if self._profiler_callback is not None:
                    if not self._profiler_callback(self.update_count):
                        # Callback returned False - stop training early
                        break

                # Reset buffer index (buffers are pre-allocated, just overwrite)
                step_in_buffer = 0

                # Run validation if needed (before metrics submission)
                val_metrics = None
                if cfg.validation_interval > 0 and self.update_count % cfg.validation_interval == 0:
                    val_metrics = self._run_validation_episodes()
                    if cfg.validation_mode == "training":
                        states = self.vec_env.reset()
                        potentials = self.reward_computer.get_initial_potentials(states)
                        episode_rewards.zero_()

                # Signal that next rollout needs fresh inference with updated weights
                needs_initial_inference = True

                # Save logits for metrics packing (action probs from this update)
                prev_logits = last_logits

                # ============================================================
                # METRICS PACKING
                # ============================================================

                # Pack metrics tensor (GPU work - fast)
                # Use prev_logits (from last step of previous update) for action probs
                action_probs_gpu = F.softmax(prev_logits, dim=1)[0]
                metrics_values = torch.cat([
                    episode_agg_tensor,           # n_episodes, reward_sum/min/max, food/poison
                    policy_loss.unsqueeze(0),
                    value_loss.unsqueeze(0),
                    entropy.unsqueeze(0),
                    explained_var.unsqueeze(0),
                    action_probs_gpu,             # Variable length tail
                ])

                # ZERO-BLOCK SUBMISSION: Send metrics to background thread
                # Main thread continues immediately, GPU keeps running
                # Background thread handles: sync, unpack, SPS calc, LogPayload, async_logger
                #
                # Pass profiler events instead of summary to avoid sync on main thread!
                # The background thread computes summary after its DMA sync completes.
                profiler_events = list(self.profiler._pending_events) if cfg.profile_enabled else []
                self.bookkeeper.submit(
                    gpu_tensor=metrics_values,
                    update_count=self.update_count,
                    total_steps=self.total_steps,
                    mode=cfg.mode,
                    profiler_events=profiler_events,
                    profiler_enabled=cfg.profile_enabled,
                    validation_metrics=val_metrics,
                    update_start_time=last_update_time,
                    prev_update_steps=last_update_steps,
                    training_start_time=self.start_time,
                )

                # Reset profiler for next update (clears pending events)
                self.profiler.reset()

                # Update timing reference for NEXT submission
                last_update_time = time.perf_counter()
                last_update_steps = self.total_steps
                
                # Checkpoint
                if self.checkpoint_interval > 0 and self.update_count % self.checkpoint_interval == 0:
                    self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save training checkpoint with full architecture metadata."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.config.mode}_u{self.update_count}_s{self.total_steps}.pth"
        )
        torch.save({
            # Brain metadata (for architecture-agnostic loading)
            'brain_type': self.config.brain_type,
            'architecture': self._policy_arch_info,
            'action_space': self.action_space.get_config(),
            'mode': self.config.mode,
            'seed': self.seed,
            # Training state
            'update': self.update_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            # Model weights (cleaned of torch.compile prefixes)
            'state_dict': _clean_state_dict(self.policy.state_dict()),
            'value_head_state_dict': _clean_state_dict(self.value_head.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}")

    def _process_pending_metrics(self):
        """
        Process metrics from PREVIOUS update (if any).

        Called at the start of each update cycle. By now, the async transfer
        that started at the end of the previous update has long since completed
        (transfer takes ~8ms, update takes ~800ms).
        """
        if self._metrics_event is None:
            return  # No pending metrics (first update)

        # Sync transfer (should be instant - finished 800ms ago)
        self._metrics_event.synchronize()

        # Unpack using schema (single source of truth)
        pending = self._pending_log_data
        cpu_array = self._metrics_pinned[:pending.n_metrics].numpy()
        m = unpack_metrics(cpu_array)

        # Update running totals (one update behind, nobody notices)
        if m['n_episodes'] > 0:
            if m['reward_max'] > self.best_reward:
                self.best_reward = m['reward_max']
            self.episode_count += m['n_episodes']

        # Prepare episode stats
        ep_stats = None
        if m['n_episodes'] > 0:
            ep_stats = {
                'reward': m['reward_sum'] / m['n_episodes'],
                'food': m['food_sum'] / m['n_episodes'],
                'poison': m['poison_sum'] / m['n_episodes'],
            }

        # Queue to async logger
        log_payload = LogPayload(
            policy_loss=m['policy_loss'],
            value_loss=m['value_loss'],
            entropy=m['entropy'],
            explained_var=m['explained_var'],
            action_probs=m['action_probs'],
            update_count=pending.update_count,
            total_steps=pending.total_steps,
            best_reward=self.best_reward,
            mode=self.config.mode,
            episode_stats=ep_stats,
            profiler_summary=pending.profiler_summary,
            sps_instant=pending.sps_instant,
            sps_rolling=pending.sps_rolling,
            sps_global=pending.sps_global,
            validation_metrics=pending.validation_metrics,
            episodes_count=m['n_episodes'],
            reward_sum=m['reward_sum'],
            reward_min=m['reward_min'] if m['n_episodes'] > 0 else 0.0,
            reward_max=m['reward_max'] if m['n_episodes'] > 0 else 0.0,
            food_sum=m['food_sum'],
            poison_sum=m['poison_sum'],
        )
        self.async_logger.log_update(log_payload)

        # Clear pending state
        self._metrics_event = None
        self._pending_log_data = None

    def _start_metrics_transfer(
        self,
        metrics_values: torch.Tensor,
        update_count: int,
        total_steps: int,
        sps_instant: float,
        sps_rolling: float,
        sps_global: float,
        profiler_summary: str,
        validation_metrics: Optional[dict],
    ):
        """
        Start async transfer of metrics to CPU.

        GPU continues immediately to next update. Transfer completes in background.
        Processing happens at start of NEXT update (or at training end).

        Metrics must be packed according to METRICS_SCHEMA order.
        """
        n_metrics = len(metrics_values)

        # Copy to pinned buffer (non-blocking DMA transfer)
        self._metrics_pinned[:n_metrics].copy_(metrics_values, non_blocking=True)

        # Record event so we know when transfer completes
        self._metrics_event = torch.cuda.Event()
        self._metrics_event.record()

        # Store context for later processing
        self._pending_log_data = PendingMetrics(
            n_metrics=n_metrics,
            update_count=update_count,
            total_steps=total_steps,
            sps_instant=sps_instant,
            sps_rolling=sps_rolling,
            sps_global=sps_global,
            profiler_summary=profiler_summary,
            validation_metrics=validation_metrics,
        )

    def _run_validation_episodes(self) -> dict:
        """
        Run deterministic validation episodes.
        
        Returns metrics dict with mean reward, food, and poison.
        Uses argmax for action selection (no exploration).
        """
        cfg = self.config
        
        # Select environment based on mode
        if cfg.validation_mode == "fixed":
            # Create temporary env with fixed density
            val_env = create_torch_vec_env(
                n_envs=cfg.n_envs, 
                obs_spec=self.spec, 
                device=self.device
            )
            val_env.set_curriculum_ranges(
                cfg.validation_food, cfg.validation_food,
                cfg.validation_poison, cfg.validation_poison
            )
            val_env.reset()
        else:
            # Use training env directly (faster, same randomization)
            val_env = self.vec_env
        
        # Run episodes
        total_reward = 0.0
        total_food = 0
        total_poison = 0
        episodes_done = 0
        
        states = val_env.reset()
        episode_rewards = torch.zeros(cfg.n_envs, device=self.device)
        
        # Run until we have enough completed episodes
        max_steps = cfg.n_envs * 600  # Safety limit
        steps = 0
        
        with torch.no_grad():
            while episodes_done < cfg.validation_episodes and steps < max_steps:
                states_t = states.float()

                # Deterministic action selection (argmax instead of sample)
                logits = self.policy(states_t)
                actions = logits.argmax(dim=-1)

                next_states, eating_info, terminated, truncated = val_env.step(actions)
                dones = terminated | truncated
                shaped_rewards = self.reward_computer.compute(eating_info, states, next_states, terminated)
                episode_rewards += shaped_rewards

                # Check for completed episodes
                done_mask = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_mask.numel() > 0:
                    for idx in done_mask:
                        if episodes_done >= cfg.validation_episodes:
                            break
                        total_reward += episode_rewards[idx].item()
                        total_food += val_env.last_episode_food[idx].item()
                        total_poison += val_env.last_episode_poison[idx].item()
                        episodes_done += 1
                        episode_rewards[idx] = 0.0
                
                states = next_states
                steps += cfg.n_envs
        
        # Compute means
        n = max(episodes_done, 1)
        metrics = {
            'reward': total_reward / n,
            'food': total_food / n,
            'poison': total_poison / n,
            'episodes': episodes_done,
            'mode': cfg.validation_mode,
        }
        
        # Print validation results
        print(f"   [Validation] reward={metrics['reward']:.2f}, "
              f"food={metrics['food']:.1f}, poison={metrics['poison']:.1f} "
              f"({episodes_done} episodes, {cfg.validation_mode} env)")
        
        return metrics
    
    
    def _finalize(self) -> dict:
        """Save final model and return summary."""
        cfg = self.config

        # Let bookkeeper finish processing any remaining metrics
        # (It processes in background, so give it a moment to catch up)
        if hasattr(self, 'bookkeeper') and self.bookkeeper is not None:
            time.sleep(0.1)  # Brief pause for background thread
            best_reward = self.bookkeeper.best_reward
        else:
            best_reward = self.best_reward

        # Calculate throughput (handle case where start_time wasn't set)
        elapsed = 0.0
        throughput = 0.0
        if hasattr(self, 'start_time'):
            elapsed = time.perf_counter() - self.start_time
            throughput = self.total_steps / elapsed if elapsed > 0 else 0

        # Handle aborted training
        if self._aborted:
            print(f"\n[{cfg.mode}] Training aborted after {self.total_steps:,} steps")
            # Still try to flush logs
            if self.async_logger:
                self.async_logger.shutdown(timeout=2.0)
            if self.dashboard:
                self.dashboard.update(cfg.mode, 'aborted', None)
            # Full cleanup
            self.cleanup()
            return {
                'mode': cfg.mode,
                'total_steps': self.total_steps,
                'elapsed_seconds': elapsed,
                'throughput': throughput,
                'best_reward': best_reward,
                'model_path': None,
                'aborted': True,
            }

        # Normal completion path
        # Shutdown async logger first - flushes all queued logs
        if self.async_logger:
            self.async_logger.shutdown()

        if cfg.benchmark_mode:
            # Benchmark mode: just report throughput, don't save
            print(f"\n{'='*60}")
            print(f"BENCHMARK RESULTS ({cfg.mode})")
            print(f"{'='*60}")
            print(f"  Total steps:    {self.total_steps:,}")
            print(f"  Elapsed time:   {elapsed:.2f}s")
            print(f"  Throughput:     {throughput:,.0f} steps/sec")
            print(f"  Best reward:    {best_reward:.1f}")
            print(f"{'='*60}")
        else:
            # Save with full metadata for architecture-agnostic loading
            checkpoint = {
                'brain_type': cfg.brain_type,
                'architecture': self._policy_arch_info,
                'action_space': self.action_space.get_config(),
                'state_dict': _clean_state_dict(self.policy.state_dict()),
                'mode': cfg.mode,
                'training_steps': self.total_steps,
                'seed': self.seed,
            }
            torch.save(checkpoint, cfg.output_path)
            print(f"\n[PPO] Training complete: {cfg.output_path} (seed={self.seed})")

        # Close TensorBoard writer to flush all data
        if self.tb_writer:
            self.tb_writer.close()

        if self.dashboard:
            self.dashboard.update(cfg.mode, 'finished', None)

        # Full cleanup after normal completion too
        self.cleanup()

        return {
            'mode': cfg.mode,
            'total_steps': self.total_steps,
            'elapsed_seconds': elapsed,
            'throughput': throughput,
            'best_reward': best_reward,
            'model_path': None if cfg.benchmark_mode else cfg.output_path,
            'aborted': False,
        }
