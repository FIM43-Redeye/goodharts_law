# GPU-First Analysis Design

This directory contains utilities for training analysis that follow a strict
GPU-first philosophy.

## The Problem

Traditional training analysis looks like this:

```python
# BAD: Transfers data to CPU every step
for step in training:
    episode_data = get_episodes()  # GPU -> CPU transfer
    episodes_list.append(episode_data)  # Python list allocation

# Then analyze on CPU
mean_reward = np.mean([e.reward for e in episodes_list])
```

This causes:
- **Serialization overhead**: Converting GPU tensors to Python objects
- **GIL contention**: Python's Global Interpreter Lock blocks during serialization
- **Memory copies**: Each transfer stalls the training pipeline
- **Jitter**: Periodic slowdowns (we saw 28k -> 11k sps every 4th update)

## The Solution: GPU-Side Aggregation

All per-step/per-episode statistics should be computed ON THE GPU using
PyTorch's parallel reduction operations. Only final aggregated results
are transferred to CPU.

```python
# GOOD: Aggregate on GPU, transfer once
episode_rewards = collected_rewards_tensor  # Already on GPU

# Single CUDA kernel computes all stats
agg = torch.stack([
    episode_rewards.sum(),
    episode_rewards.min(),
    episode_rewards.max(),
])

# One transfer: 3 floats instead of thousands of episodes
cpu_stats = agg.cpu().numpy()
```

## Available Reduction Operations

PyTorch provides these GPU-accelerated aggregations:

| Operation | Function | Use Case |
|-----------|----------|----------|
| Sum | `tensor.sum()` | Total rewards, counts |
| Mean | `tensor.mean()` | Average metrics |
| Min/Max | `tensor.min()`, `tensor.max()` | Range tracking |
| Std | `tensor.std()` | Variance analysis |
| Histogram | `torch.histc()` | Distributions |
| Percentile | `torch.quantile()` | Median, quartiles |
| Argmax | `tensor.argmax()` | Best episode index |

## Using AnalysisReceiver

The `AnalysisReceiver` class provides a generic interface for receiving
GPU-computed analysis results at the end of training:

```python
from goodharts.training.analysis import AnalysisReceiver

# During training: compute on GPU
reward_stats = torch.stack([rewards.mean(), rewards.std()])
action_distribution = action_counts.float() / action_counts.sum()

# At end: transfer and log
receiver = AnalysisReceiver(output_dir="logs")
receiver.receive(
    mode="ground_truth",
    reward_stats=reward_stats.cpu().numpy(),
    action_dist=action_distribution.cpu().numpy(),
    custom_metrics={"episodes": total_episodes},
)
```

## When to Transfer to CPU

Transfer to CPU only when:
1. **Training is complete** - Final summary statistics
2. **Logging to file** - After aggregation is done
3. **User display** - Progress bars, dashboards (sparingly)

Never transfer:
- Raw episode data
- Per-step intermediate values
- Large arrays that could be reduced first

## Performance Impact

With GPU-first analysis:
- **Before**: 28k sps with periodic drops to 11k sps (every 4th update)
- **After**: Steady 28.5-28.9k sps, no jitter

The difference: transferring 5 floats per update vs. thousands of episode records.
