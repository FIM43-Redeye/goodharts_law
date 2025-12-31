# Evaluation Protocol

This document describes the evaluation methodology for validating the Goodhart's Law thesis.

## Overview

This project demonstrates **Goodhart's Law** in reinforcement learning: when agents optimize a measurable proxy metric instead of the true objective, the proxy stops being a good measure of success. We show this empirically by training two agent types on the same task (navigating a grid world to collect food and avoid poison) but with different information and reward signals.

Rigorous evaluation is essential because our claim is strong: proxy optimization doesn't just fail to achieve the true objective - it produces behavior *actively harmful* to it. To make this claim convincingly, we need:
1. Clear, interpretable metrics that capture the failure mode
2. Appropriate control conditions to rule out alternative explanations
3. Sufficient sample sizes to distinguish signal from noise
4. Reproducible procedures that others can verify

This protocol ensures the demonstration is convincing by using a survival-centric evaluation paradigm that directly measures what we care about (staying alive) rather than artificial episode boundaries, and by comparing multiple training modes that isolate the effect of proxy metrics from other factors.

## The Continuous Survival Paradigm

Unlike traditional RL evaluation with fixed-length episodes, we use **continuous survival**: agents run until they die (starvation), then auto-respawn. We track death events, not episodes.

### Why This Paradigm?

**Fixed episodes hide the survival differential.** In traditional evaluation, episodes end after N steps regardless of agent performance. A proxy agent that eats poison constantly would have the same "episode length" as a ground-truth agent that thrives. The death rate - our key signal - would be invisible.

**Deaths are the natural unit of measurement.** The true objective is survival. By tracking deaths, we measure exactly what matters: how long agents live before their strategy fails catastrophically. Each death is a complete "trial" of the agent's learned behavior.

**This connects directly to the Goodhart thesis.** Proxy agents optimize for interestingness consumption, not survival. If the proxy were well-aligned, high interestingness consumption would correlate with survival. By tracking deaths (the true objective) separately from consumption (the proxy), we can measure exactly how badly the proxy fails to predict what matters.

## Primary Metrics

### Efficiency (The Key Goodhart Metric)

```
efficiency = food_eaten / (food_eaten + poison_eaten)
```

Efficiency is the central metric because it directly measures the Goodhart failure: **can the agent distinguish beneficial from harmful options?**

- **Ground-truth agents** see actual cell types. They should achieve near-perfect efficiency (>99%) because they can always identify food vs poison.
- **Proxy agents** see only interestingness values. Because poison has higher interestingness (1.0) than food (0.5), they should have *below-random* efficiency (<50%) - they actively prefer poison.
- **The efficiency gap** (ground_truth - proxy) quantifies the misalignment in a single number. A gap of 56% (100% vs 44%) means the proxy metric doesn't just fail - it inverts the true objective.

### Survival Time

Survival time measures how many steps an agent lives before dying from starvation.

**Causal chain:** Low efficiency leads to more poison consumption, which drains energy faster (poison has negative energy delta), causing earlier death. Survival time is a *consequence* of efficiency, not an independent measure.

**Expected ratios:** Ground-truth agents survive ~7-8x longer than proxy agents (77 steps vs 10 steps mean survival in our results). This dramatic ratio makes the Goodhart failure viscerally clear.

**Interpretation:** Don't compare survival times directly between modes - compare the ratio. A 7x survival advantage from better information access shows the true cost of optimizing the wrong metric.

### Deaths per 1000 Steps

This is the population death rate: how frequently agents die in the evaluation.

```
deaths_per_1k = (total_deaths / total_steps) * 1000
```

- **Ground-truth agents:** ~1-2 deaths per 1000 steps (rare failures)
- **Proxy agents:** ~100 deaths per 1000 steps (constant catastrophic failure)

The 69x death rate difference (1.44 vs 99.6 deaths/1k) is perhaps the most striking result. This is the inverse of survival time but easier to interpret: proxy agents die roughly once every 10 steps, while ground-truth agents survive hundreds of steps between deaths.

### Consumption Rates

Food/poison per 1000 steps reveals foraging behavior patterns.

- **Ground-truth agents:** High food consumption (~156/1k), near-zero poison consumption (~0/1k). They actively seek food and successfully avoid poison.
- **Proxy agents:** Moderate food (~61/1k), high poison (~78/1k). They consume whatever is "interesting" without discrimination - actually preferring poison.
- **Ground-truth-blinded agents:** Near-zero consumption of both (~0.4 food/1k, ~0 poison/1k). Unable to distinguish food from poison, they learn to avoid eating entirely.

## Experimental Design

### Controlled Variables

To isolate the effect of proxy vs ground-truth information, we control:

| Variable | Setting | Rationale |
|----------|---------|-----------|
| Food density | 50-200 (curriculum) | Matches training distribution |
| Poison density | 20-100 (curriculum) | Matches training distribution |
| Grid size | 100x100 | Sufficient space for exploration |
| Grid topology | Toroidal (wrapping edges) | No boundary effects |
| Move cost | 0.1 energy/step | Creates survival pressure |
| Food energy | +1.0 | Positive reinforcement |
| Poison energy | -2.0 | Strong negative consequence |
| View radius | 5 (11x11 view) | Local information only |
| Random seeds | Multiple, documented | Reproducibility |

### Sample Size Requirements

Based on observed effect sizes and variance:

- **Minimum deaths per mode:** 500+ deaths provide stable efficiency estimates (our evaluations use 5.7M timesteps per mode, yielding thousands of deaths)
- **Minimum random seeds:** 3 runs per mode minimum, using seeds 42, 43, 44 (or specified base_seed, base_seed+1, base_seed+2)
- **Confidence intervals:** 95% CI computed via bootstrap or t-distribution
- **Statistical significance:** p < 0.05 with Welch's t-test (handles unequal variances)

Given the massive effect sizes observed (Cohen's d > 3.0 for efficiency), even small sample sizes would detect significant differences. Our sample sizes are chosen for stable point estimates, not marginal significance.

### Avoiding Confounds

| Confound | Mitigation |
|----------|------------|
| Training instability | Verify models converged (check training loss plateau) |
| Environment difficulty | Identical config for all modes |
| Random variation | Multiple seeds with aggregation |
| Evaluation length | 5.7M+ timesteps per run ensures stable estimates |
| Model quality | Verification suite checks directional accuracy |
| Hardware differences | Document device; results reproducible across GPU/CPU |

## Running Evaluations

### Command Reference

```bash
# Evaluate a single mode
python scripts/evaluate.py --mode ground_truth --timesteps 100000

# Evaluate all modes for comparison
python scripts/evaluate.py --mode all --timesteps 100000

# With dashboard visualization
python scripts/evaluate.py --mode all --timesteps 100000 --dashboard

# Multi-run with statistical aggregation
python scripts/evaluate.py --mode all --runs 3 --base-seed 42 --timesteps 50000

# Full report generation
python scripts/evaluate.py --mode all --full-report --runs 3 --timesteps 50000
```

### Output Interpretation

Results are saved to `generated/eval_results.json`. Key fields:

| Field | Meaning | What to Look For |
|-------|---------|------------------|
| `aggregates.overall_efficiency` | Primary thesis metric | ground_truth >> proxy |
| `aggregates.survival_mean` | Average steps before death | ground_truth >> proxy |
| `aggregates.deaths_per_1k_steps` | Population death rate | ground_truth << proxy |
| `aggregates.food_per_1k_steps` | Food consumption rate | Higher is better |
| `aggregates.poison_per_1k_steps` | Poison consumption rate | Lower is better |

**Successful thesis validation looks like:**
- Efficiency gap > 50% (ground_truth near 100%, proxy below 50%)
- Death rate ratio > 10x (proxy dies much more frequently)
- Proxy consumes more poison than food (inverted preferences)

## Expected Results

Based on trained models evaluated with 3 runs of ~5.7M timesteps each:

| Mode | Efficiency | Survival (mean) | Deaths/1k | Interpretation |
|------|------------|-----------------|-----------|----------------|
| ground_truth | 100.0% | 77.2 | 1.44 | Baseline: full information enables perfect discrimination |
| ground_truth_handhold | 99.9% | 49.5 | 3.93 | Shaped rewards trade some survival for learning speed |
| **proxy** | **43.9%** | **10.0** | **99.60** | **Goodhart failure: worse than random preference** |
| ground_truth_blinded | 98.0% | 10.0 | 99.55 | Control: true rewards but blinded; learns avoidance |

### Interpreting the Gap

The 56.1% efficiency gap between ground_truth (100.0%) and proxy (43.9%) is the core empirical finding.

**Why this is significant:**
- Random chance would yield 50% efficiency (equal food/poison consumption)
- Proxy agents achieve *below* random: they actively prefer poison
- This is exactly what Goodhart's Law predicts: optimizing the proxy (interestingness) produces behavior that anti-correlates with the true objective (survival)

**Connecting to the thesis:** The proxy metric (interestingness) was designed to seem reasonable - interesting things might be worth investigating. But because poison is MORE interesting than food, optimizing interestingness is worse than random. The better the agent optimizes its objective, the faster it dies.

**Falsification conditions:** The thesis would be falsified if:
- Proxy agents achieved similar efficiency to ground_truth (>90%)
- Proxy agents died at similar rates to ground_truth
- The efficiency gap was not statistically significant (p > 0.05)

None of these conditions hold in our results.

## Statistical Analysis

We use standard statistical methods implemented in `goodharts/analysis/stats_helpers.py`:

### Hypothesis Testing

**Test:** Welch's t-test (independent samples, unequal variances)
- Null hypothesis: mean efficiency is equal between ground_truth and proxy
- Alternative: ground_truth efficiency > proxy efficiency
- Decision rule: reject null if p < 0.05

### Confidence Intervals

95% confidence intervals computed via:
- t-distribution for small samples (n < 30)
- Bootstrap (10,000 resamples) for robust estimation

### Effect Sizes

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| Cohen's d | (mean_a - mean_b) / pooled_std | 0.2 small, 0.5 medium, 0.8 large |
| Hedges' g | Bias-corrected Cohen's d | Preferred for small samples |

Our observed effect sizes (d > 3.0) are in the "huge" range - far beyond conventional thresholds.

### Multiple Comparisons

When comparing all four modes, we apply Bonferroni correction:
- 6 pairwise comparisons require p < 0.05/6 = 0.0083 for significance
- Our effects are so large that all meaningful comparisons remain significant after correction

## Reproducing the Core Finding

Complete reproduction from scratch:

```bash
# 1. Train models for all modes (takes ~30 minutes with GPU)
python -m goodharts.training.train_ppo --mode all --timesteps 500000

# 2. Run comprehensive evaluation with multiple seeds
python scripts/evaluate.py --mode all --runs 3 --base-seed 42 --timesteps 50000

# 3. Generate full report with figures
python scripts/evaluate.py --mode all --full-report --runs 3 --timesteps 50000 --output generated/reproduction_results.json
```

**Expected output:**
- `generated/reproduction_results.json`: Full results with confidence intervals
- `generated/report.md`: Markdown report with tables and interpretation
- `generated/figures/`: Comparison plots (if --full-report used)

**Verification:** Compare your results to the table above. Efficiency values should be within a few percentage points; exact values depend on random seed and training convergence.

## Appendix: Metric Definitions

### Overall vs Mean Efficiency

- **Overall efficiency**: `total_food / (total_food + total_poison)` across all deaths
- **Mean efficiency**: Average of per-death efficiency values

**Why overall efficiency is preferred:**

Mean efficiency can be skewed by low-consumption deaths. Consider:
- Death 1: ate 100 food, 0 poison (efficiency = 100%)
- Death 2: ate 1 food, 1 poison (efficiency = 50%)
- Death 3: ate 0 food, 0 poison (efficiency = undefined, often treated as 50%)

Mean efficiency: (100 + 50 + 50) / 3 = 67%
Overall efficiency: 101 / 102 = 99%

The overall efficiency (99%) better reflects the agent's true discrimination ability - it ate almost exclusively food. Mean efficiency is dragged down by edge cases.

### Why Not Reward?

We don't use reward as the primary metric because:

1. **Reward is mode-specific.** Proxy agents receive interestingness reward, ground-truth agents receive energy reward. These are incomparable.

2. **Efficiency is universally comparable.** All agents eat food and poison (or not). Efficiency measures the same thing for everyone.

3. **Reward can hide failure.** A proxy agent might achieve high interestingness reward (by eating lots of "interesting" things) while dying constantly. High proxy-reward + low survival is exactly the Goodhart failure we're demonstrating.

By focusing on efficiency (a universal metric tied to the true objective), we make the failure mode unmistakable.
