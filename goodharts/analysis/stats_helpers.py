"""
Statistical helpers for Goodhart's Law analysis.

Provides dataclasses and functions for computing and formatting statistical
results suitable for publication-quality figures and reports.

Core abstraction: StatisticalComparison captures everything needed to report
a comparison between two groups (ground_truth vs proxy).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class StatisticalComparison:
    """
    Complete statistical comparison between two groups.

    Contains all the information needed for publication-quality reporting:
    sample statistics, confidence intervals, test statistics, and effect sizes.
    """
    metric: str
    group_a: str
    group_b: str

    # Sample sizes
    n_a: int
    n_b: int

    # Descriptive statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float

    # Confidence intervals (default 95%)
    ci_a: tuple[float, float]  # (lower, upper) for group A
    ci_b: tuple[float, float]  # (lower, upper) for group B
    ci_diff: tuple[float, float]  # CI of mean difference

    # Test statistics (Welch's t-test)
    t_statistic: float
    p_value: float
    df: float  # Welch-Satterthwaite degrees of freedom

    # Effect sizes
    cohens_d: float
    hedges_g: float  # Bias-corrected Cohen's d

    # Derived properties
    @property
    def mean_diff(self) -> float:
        """Mean difference (A - B)."""
        return self.mean_a - self.mean_b

    @property
    def significant(self) -> bool:
        """Is the difference significant at p < 0.05?"""
        return self.p_value < 0.05

    @property
    def significance_stars(self) -> str:
        """
        Return significance stars for annotation.

        *** p < 0.001
        **  p < 0.01
        *   p < 0.05
        ns  not significant
        """
        if self.p_value < 0.001:
            return '***'
        elif self.p_value < 0.01:
            return '**'
        elif self.p_value < 0.05:
            return '*'
        return 'ns'

    @property
    def effect_magnitude(self) -> str:
        """
        Cohen's d interpretation with extended scale for extreme effects.

        Following Cohen (1988) conventions, extended for simulation contexts
        where distributions may be nearly non-overlapping:
        - negligible: |d| < 0.2
        - small: 0.2 <= |d| < 0.5
        - medium: 0.5 <= |d| < 0.8
        - large: 0.8 <= |d| < 2.0
        - very large: 2.0 <= |d| < 10.0
        - near-complete separation: |d| >= 10.0

        Note: Effect sizes above ~10 indicate distributions that barely
        overlap, where Cohen's d becomes less meaningful as a metric.
        """
        d = abs(self.cohens_d)
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        elif d < 2.0:
            return 'large'
        elif d < 10.0:
            return 'very large'
        return 'near-complete separation'

    def format_compact(self) -> str:
        """Compact format for plot annotations."""
        return f"p={format_p_value(self.p_value)}{self.significance_stars}, d={self.cohens_d:.2f}"

    def format_full(self) -> str:
        """Full format for reports."""
        return (
            f"t({self.df:.1f}) = {self.t_statistic:.2f}, "
            f"p = {format_p_value(self.p_value)} {self.significance_stars}\n"
            f"Cohen's d = {self.cohens_d:.2f} ({self.effect_magnitude})\n"
            f"95% CI of difference: [{self.ci_diff[0]:.3f}, {self.ci_diff[1]:.3f}]"
        )

    def format_table_row(self) -> str:
        """Format as a table row for console output."""
        return (
            f"{self.metric:<20} "
            f"{self.mean_diff:>+8.3f} "
            f"[{self.ci_diff[0]:>+7.3f}, {self.ci_diff[1]:>+7.3f}] "
            f"{format_p_value(self.p_value):>8} {self.significance_stars:<3} "
            f"{self.cohens_d:>+6.2f} ({self.effect_magnitude})"
        )


def compute_comparison(
    values_a: list[float],
    values_b: list[float],
    metric: str = 'value',
    group_a: str = 'A',
    group_b: str = 'B',
    confidence: float = 0.95,
) -> StatisticalComparison:
    """
    Compute full statistical comparison between two groups.

    Uses Welch's t-test (unequal variances assumed) which is more robust
    than Student's t-test and appropriate when group variances differ.

    Args:
        values_a: Measurements from group A
        values_b: Measurements from group B
        metric: Name of the metric being compared
        group_a: Label for group A
        group_b: Label for group B
        confidence: Confidence level (default 0.95)

    Returns:
        StatisticalComparison with all computed statistics
    """
    a = np.array(values_a)
    b = np.array(values_b)

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    std_a = float(np.std(a, ddof=1)) if n_a > 1 else 0.0
    std_b = float(np.std(b, ddof=1)) if n_b > 1 else 0.0

    # Confidence intervals for each group
    ci_a = compute_confidence_interval(values_a, confidence)
    ci_b = compute_confidence_interval(values_b, confidence)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Welch-Satterthwaite degrees of freedom
    if n_a > 1 and n_b > 1 and (std_a > 0 or std_b > 0):
        var_a, var_b = std_a**2, std_b**2
        num = (var_a/n_a + var_b/n_b)**2
        denom = (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
        df = num / denom if denom > 0 else n_a + n_b - 2
    else:
        df = n_a + n_b - 2

    # Confidence interval for difference
    se_diff = np.sqrt(std_a**2/n_a + std_b**2/n_b) if n_a > 0 and n_b > 0 else 0
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df) if df > 0 else 1.96
    margin = t_crit * se_diff
    diff = mean_a - mean_b
    ci_diff = (diff - margin, diff + margin)

    # Effect sizes
    cohens_d, hedges_g = compute_effect_sizes(values_a, values_b)

    return StatisticalComparison(
        metric=metric,
        group_a=group_a,
        group_b=group_b,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        ci_a=ci_a,
        ci_b=ci_b,
        ci_diff=ci_diff,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        df=float(df),
        cohens_d=cohens_d,
        hedges_g=hedges_g,
    )


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute confidence interval using t-distribution.

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(values[0]), float(values[0]))

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_crit * std / np.sqrt(n)

    return (mean - margin, mean + margin)


def compute_effect_sizes(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    """
    Compute Cohen's d and Hedges' g effect sizes.

    Cohen's d uses pooled standard deviation.
    Hedges' g applies small-sample correction (less biased for n < 20).

    Args:
        values_a: Group A values
        values_b: Group B values

    Returns:
        (cohens_d, hedges_g) tuple
    """
    a = np.array(values_a)
    b = np.array(values_b)

    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0

    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a = np.var(a, ddof=1) if n_a > 1 else 0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0, 0.0

    # Cohen's d
    cohens_d = float((mean_a - mean_b) / pooled_std)

    # Hedges' g (small-sample correction factor)
    # J = 1 - 3 / (4*(n1+n2) - 9)
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    hedges_g = cohens_d * correction

    return cohens_d, float(hedges_g)


def format_p_value(p: float, threshold: float = 0.001) -> str:
    """
    Format p-value for display.

    Returns '< 0.001' for very small values, otherwise formats with precision.

    Args:
        p: The p-value
        threshold: Values below this shown as '< threshold'

    Returns:
        Formatted string like '< 0.001' or '0.023'
    """
    if p < threshold:
        return f"< {threshold}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size with context.

    Extended beyond Cohen's original conventions to handle simulation
    contexts where distributions may be nearly non-overlapping.

    Args:
        d: Cohen's d value

    Returns:
        Human-readable interpretation
    """
    magnitude = abs(d)
    direction = "higher" if d > 0 else "lower"

    if magnitude < 0.2:
        return f"Negligible difference (d={d:.2f})"
    elif magnitude < 0.5:
        return f"Small effect: Group A {direction} (d={d:.2f})"
    elif magnitude < 0.8:
        return f"Medium effect: Group A {direction} (d={d:.2f})"
    elif magnitude < 2.0:
        return f"Large effect: Group A substantially {direction} (d={d:.2f})"
    elif magnitude < 10.0:
        return f"Very large effect: Group A dramatically {direction} (d={d:.2f})"
    else:
        return f"Near-complete separation: distributions barely overlap (d={d:.1f})"


def compute_goodhart_failure_index(
    gt_efficiency: float,
    proxy_efficiency: float,
) -> float:
    """
    Compute Goodhart Failure Index (GFI).

    GFI = (ground_truth_efficiency - proxy_efficiency) / ground_truth_efficiency

    Interpretation:
    - 0.0: No Goodhart failure (proxy performs as well as ground_truth)
    - 0.5: Moderate failure (proxy at 50% of ground_truth performance)
    - 1.0: Complete failure (proxy has zero efficiency)

    Args:
        gt_efficiency: Ground truth mode's efficiency (0-1)
        proxy_efficiency: Proxy mode's efficiency (0-1)

    Returns:
        GFI value (typically 0-1, can exceed 1 if proxy worse than random)
    """
    if gt_efficiency <= 0:
        return 0.0
    return (gt_efficiency - proxy_efficiency) / gt_efficiency
