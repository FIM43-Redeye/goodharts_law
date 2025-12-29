# Evaluation Protocol

This document describes the evaluation methodology for validating the Goodhart's Law thesis.

## Overview

TODO: Write a 2-3 paragraph overview explaining:
- What we're trying to demonstrate (Goodhart's Law in RL)
- Why rigorous evaluation matters for this claim
- How this protocol ensures the demonstration is convincing

## The Continuous Survival Paradigm

Unlike traditional RL evaluation with fixed-length episodes, we use **continuous survival**: agents run until they die (starvation), then auto-respawn. We track death events, not episodes.

TODO: Explain why this paradigm is superior for demonstrating Goodhart's Law:
- Why fixed episodes would hide the survival differential
- Why deaths are the natural unit of measurement
- How this connects to the "true objective" (survival) vs proxy metrics

## Primary Metrics

### Efficiency (The Key Goodhart Metric)

```
efficiency = food_eaten / (food_eaten + poison_eaten)
```

TODO: Explain why efficiency is the central metric:
- What it measures (ability to distinguish food from poison)
- Why ground truth agents should have high efficiency (~95%+)
- Why proxy agents should have low efficiency (~40-60%)
- How the efficiency gap quantifies the misalignment

### Survival Time

TODO: Explain survival time as a consequence metric:
- Causal chain: low efficiency -> more poison -> faster energy drain -> earlier death
- Expected survival ratios between modes
- Why this is a CONSEQUENCE of efficiency, not an independent measure

### Deaths per 1000 Steps

TODO: Explain population death rate:
- How to interpret this metric
- Expected differences between modes
- Relationship to survival time (inverse)

### Consumption Rates

TODO: Explain food/poison per 1000 steps:
- What these tell us about foraging behavior
- Expected patterns for each mode

## Experimental Design

### Controlled Variables

TODO: Document what must be controlled:
- Food/poison density (should match training distribution)
- Grid size and topology
- Move cost and energy dynamics
- Random seeds for reproducibility

### Sample Size Requirements

TODO: Specify your statistical requirements:
- Minimum deaths per mode: ___
- Minimum random seeds: ___
- How to compute confidence intervals
- What constitutes "statistically significant" evidence

### Avoiding Confounds

TODO: Document potential confounds and how to avoid them:
- Training instability (ensure models converged)
- Environment difficulty (use same distribution for all modes)
- Random variation (run multiple seeds)
- Evaluation length (run long enough for stable estimates)

## Running Evaluations

### Command Reference

```bash
# Evaluate a single mode
python scripts/evaluate.py --mode ground_truth --timesteps 10000

# Evaluate all modes for comparison
python scripts/evaluate.py --mode all --timesteps 10000

# With dashboard visualization
python scripts/evaluate.py --mode all --timesteps 10000 --dashboard
```

### Output Interpretation

Results are saved to `generated/eval_results.json`. Key fields:

TODO: Document how to interpret each output field:
- `aggregates.overall_efficiency`: The primary thesis metric
- `aggregates.survival_mean`: Average steps before death
- `aggregates.deaths_per_1k_steps`: Population death rate
- What patterns indicate successful thesis validation

## Expected Results

TODO: Fill in expected results after running experiments:

| Mode | Efficiency | Survival (mean) | Deaths/1k | Interpretation |
|------|------------|-----------------|-----------|----------------|
| ground_truth | TBD | TBD | TBD | Baseline: agents with true information |
| proxy | TBD | TBD | TBD | Goodhart failure: agents with proxy metric |
| ground_truth_blinded | TBD | TBD | TBD | Control: blinded but true rewards |

### Interpreting the Gap

TODO: Explain what the efficiency gap means:
- Why a gap of X% is significant
- How to connect these numbers to the Goodhart's Law thesis
- What would FALSIFY the thesis (if proxy matched ground_truth)

## Statistical Analysis

TODO: Document your analysis approach:
- How to test for significant differences between modes
- Confidence interval calculation
- Effect size reporting
- Multiple comparison corrections (if applicable)

## Reproducing the Core Finding

To reproduce the Goodhart's Law demonstration:

```bash
# 1. Train models for both modes
python -m goodharts.training.train_ppo --mode ground_truth --timesteps 500000
python -m goodharts.training.train_ppo --mode proxy --timesteps 500000

# 2. Evaluate both models
python scripts/evaluate.py --mode ground_truth --timesteps 50000
python scripts/evaluate.py --mode proxy --timesteps 50000

# 3. Compare results
# TODO: Add comparison script or instructions
```

TODO: Add specific commands and expected output for a complete reproduction.

## Appendix: Metric Definitions

### Overall vs Mean Efficiency

- **Overall efficiency**: `total_food / (total_food + total_poison)` across all deaths
- **Mean efficiency**: Average of per-death efficiency values

TODO: Explain why overall efficiency is preferred:
- Mean can be skewed by low-consumption deaths
- Overall weights by consumption volume
- Example showing the difference

### Why Not Reward?

TODO: Explain why we don't use reward as the primary metric:
- Reward is mode-specific (proxy gets interestingness reward)
- Efficiency is comparable across modes
- Reward can hide the failure (proxy might get high proxy-reward while dying)
