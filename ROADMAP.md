# Goodhart's Law Simulation - Future Enhancements

## Core Thesis
We want EMERGENCE, not just optimization failure. The current demo shows proxies fail,
but the "a-ha" moment should be watching agents *discover* deceptive strategies themselves.

---

## Phase 1: Better Measurement & Visualization ✅
- [x] Track WHY agents die (starvation vs poison) - death stats
- [x] Per-agent energy over time charts (dynamic by behavior type)
- [x] Heatmaps with RadioButton filtering by agent type
- [x] Death statistics as stacked bar charts

## Phase 2: Learned Behaviors (CNN/RL) ✅
- [x] BaseCNN architecture with dynamic channel input
- [x] LearnedBehavior with presets (ground_truth, ground_truth_handhold, proxy, ground_truth_blinded)
- [x] PPO with GAE-Lambda (GPU-native vectorized)
- [x] Multi-mode training dashboard
- [x] Structured logging (CSV/JSON)
- [x] Model verification suite
- [x] Saliency visualization (gradient-based interpretability)
- [x] **Run full comparison**: ground-truth vs proxy-trained agents (see README results table)
- [x] Statistical validation that proxy agents die more from poison (69x death rate, 56.1% efficiency gap)

## Phase 2.5: Code Quality ✅
- [x] TOML configuration system with defaults
- [x] Auto-discovery behavior registry
- [x] Centralized device selection (CPU/CUDA/ROCm)
- [x] Comprehensive test suite
- [x] Vectorized environment (VecEnv with shared_grid mode)
- [x] Documentation refresh (README, training docs)

---

## Phase 3: Research Directions (Architectural Redesign Required)

The following experiments would extend the Goodhart demonstration to more complex failure modes, but would require **significant architectural changes** to TorchVecEnv. The current vectorized environment assumes independent agents with no inter-agent communication or shared state manipulation.

### Why This Matters
The current demo shows Goodhart failure in single-agent optimization. Multi-agent dynamics would demonstrate:
- How proxy metrics fail under competition
- Whether agents can learn to game metrics (not just optimize them)
- Emergent deception as a convergent strategy

### Potential Experiments (require new architecture)
- Multi-agent signaling dynamics (agents need shared observation space)
- Resource competition under scarcity (agents need to see each other)
- Proxy gaming: can agents CREATE high-proxy cells? (agents need world-modification)
- "Inspector" vs "deceiver" co-evolution (requires population-level training)

### Incremental Extensions (possible with current architecture)
- [ ] **Temporal state**: Feed step count / episode progress (aux MLP head)
- [ ] **Noisy vision**: Gaussian noise at view edges (uncertainty modeling)
- [ ] **Recurrent agents**: LSTM/GRU for memory
- [ ] **Auxiliary scalar inputs**: Energy level, age, etc.

These single-agent extensions don't require architectural changes and could strengthen the demonstration.

---

## Phase 4: Publication-Ready ✅
- [x] Comprehensive README with AI safety connection
- [x] Reproducible experiments with seeds and config files (TOML config, deterministic seeds, run_full_evaluation.sh)
- [x] Statistical analysis across many runs (multi-run aggregation with CIs in evaluation)
- [ ] Diagrams showing information asymmetry (proxy vs ground truth)
- [x] Connection to real-world AI alignment failures (YouTube, CTR, test scores in README)

---

## Technical Debt

### High Priority
- [x] Add validation episodes to RL training (periodic eval without exploration)
- [?] Make dropout/weight_decay configurable in TOML (Likely unnecessary)

### Performance Optimizations
- [ ] Staggered multi-mode training for better GPU utilization
      (Currently GPU pauses briefly during CPU work between modes.
       Staggered starts could keep GPU more consistently busy.)
- [ ] Profile memory usage for large n_envs
- [x] torch.compile() for model optimization (implemented with async warmup)

### Code Quality
- [ ] Consistent error handling across modules
- [ ] More granular logging levels
- [ ] CLI help text improvements

---

## Wild Ideas (Someday)
- Agents that can "lie" about what they found
- Evolving the proxy function itself
- Meta-learning: agents learn what proxy to use
- Hierarchical agents (managers who set proxies, workers who optimize)
- Self-play between proxy-gamers and inspectors

---

## Moonshot (we did it anyway)
- [x] Reimplement environment on GPU (JAX/Warp) for massive parallelization
      (Current bottleneck is Python/NumPy for env, GPU for neural net.
       Full GPU would unlock 100k+ sps.)
       Proceeded to do this anyway, the perf gain was actually worth it (10x+!)
