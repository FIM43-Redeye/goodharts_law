"""
Power analysis for Goodhart's Law experiments.

Provides functions to determine required sample sizes for detecting
efficiency differences between ground_truth and proxy agents.

In the continuous survival paradigm, the unit of analysis is a "death".
Each death produces an efficiency measurement (food / total consumed).
This module helps answer: "How many deaths do we need to detect
a meaningful Goodhart effect with statistical confidence?"

Typical effect sizes in Goodhart experiments:
- d = 1.0-2.0: Moderate proxy failure (50-70% efficiency vs 90%+)
- d = 2.0-3.0: Severe proxy failure (30-50% efficiency vs 90%+)
- d > 3.0: Catastrophic failure (proxy near random chance)

For reference: random selection would yield ~50% efficiency
(assuming equal food and poison), while optimal behavior approaches 100%.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class PowerAnalysisResult:
    """
    Results from power analysis calculation.

    Provides guidance on sample size requirements and practical
    estimates for experimental planning.
    """
    # Input parameters
    effect_size: float  # Expected Cohen's d
    alpha: float        # Significance level (Type I error rate)
    power: float        # Target power (1 - Type II error rate)

    # Computed requirements
    n_per_group: int           # Deaths needed per mode
    total_n: int               # Total deaths across both modes

    # Practical guidance (based on typical death rates)
    estimated_timesteps: int   # Rough timestep estimate per mode
    deaths_per_1k_assumed: float  # Assumed death rate used for estimate

    # Context
    interpretation: str        # Human-readable guidance

    def format_guidance(self) -> str:
        """Format complete guidance message."""
        return (
            f"Power Analysis Results\n"
            f"{'='*50}\n"
            f"To detect effect size d = {self.effect_size:.2f}\n"
            f"with {self.power:.0%} power at alpha = {self.alpha}\n"
            f"\n"
            f"Required sample size:\n"
            f"  - {self.n_per_group:,} deaths per mode\n"
            f"  - {self.total_n:,} total deaths\n"
            f"\n"
            f"Estimated experiment duration:\n"
            f"  - ~{self.estimated_timesteps:,} timesteps per mode\n"
            f"  - (assuming {self.deaths_per_1k_assumed:.1f} deaths/1k steps)\n"
            f"\n"
            f"Interpretation:\n"
            f"  {self.interpretation}"
        )


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> int:
    """
    Calculate required sample size per group for two-sample t-test.

    Uses the standard power analysis formula based on non-central t-distribution.
    For the Goodhart experiment, each "sample" is a death event from one mode.

    Args:
        effect_size: Expected Cohen's d (difference in means / pooled std)
        alpha: Significance level (default 0.05)
        power: Target power, 1 - beta (default 0.80)
        two_tailed: Use two-tailed test (default True, more conservative)

    Returns:
        Required n per group (number of deaths per mode)

    Example:
        >>> required_sample_size(effect_size=1.5, power=0.80)
        15  # Need 15 deaths per mode
    """
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")

    # For two-sample t-test with equal n:
    # n = 2 * ((z_alpha + z_beta) / d)^2
    # where z_alpha is critical value for alpha, z_beta for beta=1-power

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    # Round up to ensure adequate power
    return int(np.ceil(n))


def achieved_power(
    n_a: int,
    n_b: int,
    effect_size: float,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Calculate achieved power given sample sizes.

    Useful for post-hoc power analysis of completed experiments.
    Note: Post-hoc power analysis is controversial; this is primarily
    for planning future experiments based on pilot data.

    Args:
        n_a: Sample size group A (deaths from ground_truth)
        n_b: Sample size group B (deaths from proxy)
        effect_size: Observed or assumed Cohen's d
        alpha: Significance level
        two_tailed: Two-tailed test

    Returns:
        Achieved power (0-1)

    Example:
        >>> achieved_power(n_a=50, n_b=50, effect_size=1.5)
        0.99  # 99% power with these sample sizes
    """
    if n_a <= 1 or n_b <= 1:
        return 0.0
    if effect_size <= 0:
        return alpha  # Power = alpha when no effect

    # Harmonic mean for unequal sample sizes
    n_harmonic = 2 * n_a * n_b / (n_a + n_b)

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_harmonic / 2)

    # Degrees of freedom (approximate for Welch's t)
    df = n_a + n_b - 2

    # Critical t-value
    if two_tailed:
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)

    # Power = P(reject H0 | H1 true)
    # Using non-central t-distribution
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    return float(np.clip(power, 0, 1))


def minimum_detectable_effect(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> float:
    """
    Calculate minimum detectable effect size given sample size.

    Inverts the sample size formula to find the smallest effect
    that can be detected with the given sample and power.

    Args:
        n_per_group: Available sample size (deaths per mode)
        alpha: Significance level
        power: Target power

    Returns:
        Minimum Cohen's d detectable with given parameters

    Example:
        >>> minimum_detectable_effect(n_per_group=30, power=0.80)
        0.73  # Can detect effects of d >= 0.73
    """
    if n_per_group <= 1:
        return float('inf')

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Rearranged sample size formula
    d = (z_alpha + z_beta) * np.sqrt(2 / n_per_group)

    return float(d)


def power_analysis(
    effect_size: Optional[float] = None,
    pilot_data: Optional[dict] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    deaths_per_1k_steps: float = 3.0,
) -> PowerAnalysisResult:
    """
    Full power analysis with practical guidance.

    Can use either a specified effect size or estimate from pilot data.

    Args:
        effect_size: Expected Cohen's d. If None, estimate from pilot_data.
        pilot_data: Optional dict with 'ground_truth' and 'proxy' efficiency lists
        alpha: Significance level
        power: Target power
        deaths_per_1k_steps: Typical death rate for timestep estimation

    Returns:
        PowerAnalysisResult with sample size requirements and guidance

    Example:
        # From expected effect size
        result = power_analysis(effect_size=2.0)

        # From pilot data
        pilot = {
            'ground_truth': [0.92, 0.94, 0.91, 0.93],
            'proxy': [0.48, 0.52, 0.45, 0.51]
        }
        result = power_analysis(pilot_data=pilot)
    """
    # Determine effect size
    if effect_size is None and pilot_data is not None:
        gt = pilot_data.get('ground_truth', [])
        px = pilot_data.get('proxy', [])
        if gt and px:
            from goodharts.analysis.stats_helpers import compute_effect_sizes
            effect_size, _ = compute_effect_sizes(gt, px)
            effect_size = abs(effect_size)  # Use absolute value
        else:
            raise ValueError("pilot_data must contain 'ground_truth' and 'proxy' lists")
    elif effect_size is None:
        raise ValueError("Must provide either effect_size or pilot_data")

    # Calculate required sample size
    n_per_group = required_sample_size(effect_size, alpha, power)
    total_n = n_per_group * 2

    # Estimate timesteps needed
    estimated_timesteps = int(n_per_group * 1000 / deaths_per_1k_steps)

    # Generate interpretation
    interpretation = _interpret_sample_size(n_per_group, effect_size, power)

    return PowerAnalysisResult(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        n_per_group=n_per_group,
        total_n=total_n,
        estimated_timesteps=estimated_timesteps,
        deaths_per_1k_assumed=deaths_per_1k_steps,
        interpretation=interpretation,
    )


def _interpret_sample_size(n: int, d: float, power: float) -> str:
    """Generate human-readable interpretation of sample size requirements."""
    if n < 10:
        feasibility = "Very achievable - minimal data collection needed"
    elif n < 30:
        feasibility = "Easily achievable with a short evaluation run"
    elif n < 100:
        feasibility = "Achievable with a moderate evaluation run"
    elif n < 500:
        feasibility = "Requires extended evaluation; consider longer timesteps"
    else:
        feasibility = "Large sample needed; may want to increase effect size expectations"

    if d >= 2.0:
        effect_note = f"Effect size d={d:.1f} is large; typical of severe Goodhart failure"
    elif d >= 0.8:
        effect_note = f"Effect size d={d:.1f} is large; should be detectable"
    elif d >= 0.5:
        effect_note = f"Effect size d={d:.1f} is medium; more samples improve reliability"
    else:
        effect_note = f"Effect size d={d:.1f} is small; may need many more samples"

    return f"{feasibility}. {effect_note}."


def estimate_effect_size_from_efficiencies(
    gt_efficiency: float,
    proxy_efficiency: float,
    efficiency_std: float = 0.15,
) -> float:
    """
    Estimate Cohen's d from expected efficiency values.

    Useful for planning when you know approximate expected efficiencies
    but not the full distributions.

    Args:
        gt_efficiency: Expected ground_truth efficiency (0-1)
        proxy_efficiency: Expected proxy efficiency (0-1)
        efficiency_std: Assumed standard deviation of efficiency (default 0.15)

    Returns:
        Estimated Cohen's d

    Example:
        # Ground truth at 95%, proxy at 50%
        >>> estimate_effect_size_from_efficiencies(0.95, 0.50)
        3.0  # Very large effect
    """
    diff = abs(gt_efficiency - proxy_efficiency)
    return diff / efficiency_std


# Convenience function for CLI
def print_power_table(
    effect_sizes: list[float] = None,
    powers: list[float] = None,
    alpha: float = 0.05,
):
    """
    Print a table of sample sizes for various effect sizes and power levels.

    Useful for experiment planning.
    """
    if effect_sizes is None:
        effect_sizes = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    if powers is None:
        powers = [0.80, 0.90, 0.95]

    print(f"\nRequired Sample Sizes (n per group, alpha={alpha})")
    print("=" * 60)

    # Header
    header = "Effect Size (d) |"
    for p in powers:
        header += f" Power={p:.0%} |"
    print(header)
    print("-" * 60)

    # Rows
    for d in effect_sizes:
        row = f"      {d:.1f}        |"
        for p in powers:
            n = required_sample_size(d, alpha, p)
            row += f"    {n:>4}    |"
        print(row)

    print("=" * 60)
    print("\nNote: These are deaths per mode needed for two-sample t-test.")
    print("Estimate timesteps as: n * 1000 / (deaths per 1k steps)")
