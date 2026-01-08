"""
Report generation for Goodhart's Law experiments.

Generates unified markdown reports with embedded figures and complete
statistical analysis. Designed for publication-ready output suitable
for publication-ready output.

Key features:
- Executive summary with Goodhart Failure Index prominently displayed
- Per-metric statistical comparisons with p-values, CIs, effect sizes
- Embedded figure references
- Power analysis section
- Auto-generated methodology description
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import numpy as np

from goodharts.analysis.stats_helpers import (
    StatisticalComparison,
    compute_comparison,
    format_p_value,
)
from goodharts.analysis.power import power_analysis, achieved_power


# Energy calculation constants (from config.default.toml)
_FOOD_REWARD = 1.0
_POISON_PENALTY = 2.0
_MOVE_COST_PER_1K = 10.0  # 0.01 * 1000

# Preferred mode ordering for reports (narrative progression: aligned → misaligned)
_MODE_ORDER = ['ground_truth', 'ground_truth_blinded', 'proxy_mortal', 'proxy']


def _sort_modes(modes: list[str]) -> list[str]:
    """Sort modes in narrative order, with unknown modes at the end."""
    def key(m):
        try:
            return _MODE_ORDER.index(m)
        except ValueError:
            return len(_MODE_ORDER)  # Unknown modes go last
    return sorted(modes, key=key)


def _calculate_energy_per_1k(agg: dict) -> float:
    """
    Calculate energy/1k from food and poison rates.

    Used as fallback when energy_per_1k_steps is missing from aggregates
    (for backward compatibility with results generated before the fix).

    Formula: food_rate * food_reward - poison_rate * poison_penalty - move_cost_per_1k
    """
    # First check if energy_per_1k_steps is already present
    if 'energy_per_1k_steps' in agg and agg['energy_per_1k_steps'] != 0:
        return agg['energy_per_1k_steps']

    # Calculate from food/poison rates
    food_rate = agg.get('food_per_1k_steps', 0)
    poison_rate = agg.get('poison_per_1k_steps', 0)

    if food_rate == 0 and poison_rate == 0:
        return 0.0

    return food_rate * _FOOD_REWARD - poison_rate * _POISON_PENALTY - _MOVE_COST_PER_1K


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Goodhart's Law Experiment Results"
    output_dir: Path = field(default_factory=lambda: Path('generated/reports'))
    include_figures: bool = True
    include_power_analysis: bool = True
    include_methodology: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    @property
    def report_dir(self) -> Path:
        """Directory for this report's outputs."""
        return self.output_dir / self.timestamp

    @property
    def figures_dir(self) -> Path:
        """Directory for figures."""
        return self.report_dir / 'figures'

    @property
    def report_path(self) -> Path:
        """Path to the markdown report."""
        return self.report_dir / 'report.md'


class ReportGenerator:
    """
    Generates unified markdown reports from evaluation results.

    Usage:
        generator = ReportGenerator()
        generator.add_data('generated/eval_results.json')
        report_path = generator.generate()
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.data: dict = {}
        self.comparisons: dict[str, StatisticalComparison] = {}
        self.figure_paths: list[Path] = []

    def add_data(self, path: str) -> 'ReportGenerator':
        """
        Load evaluation results from JSON.

        Args:
            path: Path to JSON file from evaluate.py

        Returns:
            self for method chaining
        """
        with open(path, 'r') as f:
            self.data = json.load(f)
        return self

    def add_figures(self, figure_dir: Path) -> 'ReportGenerator':
        """
        Register generated figures for embedding.

        Args:
            figure_dir: Directory containing PNG figures

        Returns:
            self for method chaining
        """
        self.figure_paths = list(figure_dir.glob('*.png'))
        return self

    def _extract_mode_data(self) -> dict[str, list[dict]]:
        """Extract per-death/episode AND survivor data for each mode.

        CRITICAL: Must include survivors for accurate distributions.
        Without survivors, ground_truth would only show the rare deaths,
        completely misrepresenting the typical ~100% efficiency behavior.
        """
        mode_data = {}

        results = self.data.get('results', {})
        for mode, result in results.items():
            mode_data[mode] = []

            # Extract deaths
            for d in result.get('deaths', []):
                food = d.get('food_eaten', 0)
                poison = d.get('poison_eaten', 0)
                total = food + poison
                efficiency = food / total if total > 0 else 1.0
                mode_data[mode].append({
                    'efficiency': efficiency,
                    'survival_steps': d.get('survival_time', 0),
                    'food_eaten': food,
                    'poison_eaten': poison,
                    'total_reward': d.get('total_reward', 0),
                    'censored': False,
                })

            # Extract survivors (right-censored data - they lived AT LEAST this long)
            for s in result.get('survivors', []):
                food = s.get('food_eaten', 0)
                poison = s.get('poison_eaten', 0)
                total = food + poison
                efficiency = food / total if total > 0 else 1.0
                mode_data[mode].append({
                    'efficiency': efficiency,
                    'survival_steps': s.get('survival_time', 0),
                    'food_eaten': food,
                    'poison_eaten': poison,
                    'total_reward': 0,  # Survivors don't have final reward
                    'censored': True,
                })

        return mode_data

    def _compute_pairwise_comparison(
        self,
        mode_data: dict,
        mode_a: str,
        mode_b: str
    ) -> dict[str, StatisticalComparison]:
        """Compute efficiency, survival, and poison comparisons for a mode pair."""
        if mode_a not in mode_data or mode_b not in mode_data:
            return {}

        data_a = mode_data[mode_a]
        data_b = mode_data[mode_b]

        if not data_a or not data_b:
            return {}

        comparisons = {}

        # Efficiency comparison
        eff_a = [d['efficiency'] * 100 for d in data_a]
        eff_b = [d['efficiency'] * 100 for d in data_b]
        comparisons['efficiency'] = compute_comparison(
            eff_a, eff_b,
            metric='efficiency',
            group_a=mode_a,
            group_b=mode_b
        )

        # Survival comparison
        surv_a = [d['survival_steps'] for d in data_a]
        surv_b = [d['survival_steps'] for d in data_b]
        comparisons['survival'] = compute_comparison(
            surv_a, surv_b,
            metric='survival',
            group_a=mode_a,
            group_b=mode_b
        )

        # Poison comparison
        poison_a = [d['poison_eaten'] for d in data_a]
        poison_b = [d['poison_eaten'] for d in data_b]
        comparisons['poison'] = compute_comparison(
            poison_a, poison_b,
            metric='poison_eaten',
            group_a=mode_a,
            group_b=mode_b
        )

        return comparisons

    def _compute_comparisons(self, mode_data: dict) -> dict[str, dict[str, StatisticalComparison]]:
        """
        Compute statistical comparisons for all meaningful mode pairs.

        Returns a nested dict: {pair_name: {metric: StatisticalComparison}}

        Comparison pairs and what they test:
        - gt_vs_proxy: Primary Goodhart effect (full alignment gap)
        - gt_vs_blinded: Impact of observation quality (can't see reality)
        - mortal_vs_proxy: Impact of training mortality (reality check)
        - blinded_vs_mortal: Reward quality vs mortality (which matters more)
        """
        all_comparisons = {}

        # Primary comparison: The Goodhart effect
        gt_vs_proxy = self._compute_pairwise_comparison(
            mode_data, 'ground_truth', 'proxy'
        )
        if gt_vs_proxy:
            all_comparisons['gt_vs_proxy'] = gt_vs_proxy

        # Control 1: Impact of observation quality
        gt_vs_blinded = self._compute_pairwise_comparison(
            mode_data, 'ground_truth', 'ground_truth_blinded'
        )
        if gt_vs_blinded:
            all_comparisons['gt_vs_blinded'] = gt_vs_blinded

        # Control 2: Does mortality ground proxy agents?
        mortal_vs_proxy = self._compute_pairwise_comparison(
            mode_data, 'proxy_mortal', 'proxy'
        )
        if mortal_vs_proxy:
            all_comparisons['mortal_vs_proxy'] = mortal_vs_proxy

        # Control 3: Right reward vs mortality - which matters more?
        blinded_vs_mortal = self._compute_pairwise_comparison(
            mode_data, 'ground_truth_blinded', 'proxy_mortal'
        )
        if blinded_vs_mortal:
            all_comparisons['blinded_vs_mortal'] = blinded_vs_mortal

        self.comparisons = all_comparisons
        return all_comparisons

    def _generate_executive_summary(self, mode_data: dict) -> str:
        """Generate executive summary section."""
        lines = ['## Executive Summary', '']

        # Survival Collapse Ratio - the real measure of catastrophic failure
        results = self.data.get('results', {})
        gt_agg = results.get('ground_truth', {}).get('aggregates', {})
        px_agg = results.get('proxy', {}).get('aggregates', {})

        if gt_agg and px_agg:
            gt_surv = gt_agg.get('survival_mean', 1)
            px_surv = px_agg.get('survival_mean', 1)
            scr = gt_surv / px_surv if px_surv > 0 else float('inf')

            gt_energy = _calculate_energy_per_1k(gt_agg)
            px_energy = _calculate_energy_per_1k(px_agg)

            lines.append(f'**Survival Collapse Ratio: {scr:.0f}x**')
            lines.append('')
            if gt_energy > 0 and px_energy < 0:
                lines.append(f'> Ground truth agents gain energy (+{gt_energy:.1f}/1k steps) and thrive. '
                            f'Proxy agents hemorrhage energy ({px_energy:.1f}/1k steps) and die. '
                            f'This is Goodhart\'s Law: optimizing the proxy metric leads to catastrophic failure.')
            else:
                lines.append(f'> Ground truth agents survive {scr:.0f}x longer than proxy agents. '
                            f'Ground truth energy: {gt_energy:+.1f}/1k, Proxy energy: {px_energy:+.1f}/1k.')
            lines.append('')

        # Summary table - energy per 1k steps is THE key metric
        lines.append('### Performance Summary')
        lines.append('')
        lines.append('| Mode | Energy/1k | Efficiency | Survival | Deaths | Deaths/1k |')
        lines.append('|------|-----------|------------|----------|--------|-----------|')

        results = self.data.get('results', {})
        for mode in _sort_modes(results.keys()):
            agg = results[mode].get('aggregates', {})
            if agg:
                energy = _calculate_energy_per_1k(agg)
                energy_str = f"{'+' if energy >= 0 else ''}{energy:.1f}"
                eff = agg.get('overall_efficiency', 0)
                surv = agg.get('survival_mean', 0)
                # Multi-run aggregates use 'total_deaths', single-run uses 'n_deaths'
                deaths = agg.get('total_deaths', agg.get('n_deaths', 0))
                d1k = agg.get('deaths_per_1k_steps', 0)
                lines.append(f'| {mode} | {energy_str} | {eff:.1%} | {surv:.1f} | {deaths:,} | {d1k:.2f} |')

        lines.append('')
        lines.append('> *Energy/1k*: Net energy change per 1000 steps (positive = thriving, negative = dying)')
        lines.append('> *Efficiency*: Food eaten / total consumed (above 50% = better than random)')
        lines.append('')
        return '\n'.join(lines)

    def _format_comparison_block(
        self,
        comp: StatisticalComparison,
        label_a: str,
        label_b: str
    ) -> list[str]:
        """Format a single metric comparison as markdown lines."""
        lines = []
        lines.append(f'- **{label_a}**: {comp.mean_a:.2f} (95% CI: [{comp.ci_a[0]:.2f}, {comp.ci_a[1]:.2f}])')
        lines.append(f'- **{label_b}**: {comp.mean_b:.2f} (95% CI: [{comp.ci_b[0]:.2f}, {comp.ci_b[1]:.2f}])')
        lines.append(f'- **Difference**: {comp.mean_diff:+.2f} (95% CI: [{comp.ci_diff[0]:.2f}, {comp.ci_diff[1]:.2f}])')
        lines.append(f'- t({comp.df:.1f}) = {comp.t_statistic:.2f}, '
                    f'p = {format_p_value(comp.p_value)} {comp.significance_stars}, '
                    f"Cohen's d = {comp.cohens_d:.2f} ({comp.effect_magnitude})")
        return lines

    def _generate_statistical_analysis(self) -> str:
        """Generate statistical analysis section with all comparison pairs."""
        lines = ['## Statistical Analysis', '']

        if not self.comparisons:
            lines.append('*No comparison data available (requires multiple modes)*')
            return '\n'.join(lines)

        # Comparison metadata: pair_key -> (title, description, label_a, label_b)
        comparison_info = {
            'gt_vs_proxy': (
                'Primary: Ground Truth vs Proxy (The Goodhart Effect)',
                'The core demonstration: agents optimizing for a proxy metric versus '
                'agents with access to ground truth. This measures the full alignment gap.',
                'Ground Truth', 'Proxy'
            ),
            'gt_vs_blinded': (
                'Control 1: Ground Truth vs Blinded (Observation Quality)',
                'Both modes receive energy-based rewards and can die, but blinded agents '
                'cannot distinguish food from poison visually. Tests whether observation '
                'quality alone impacts survival.',
                'Ground Truth', 'Blinded'
            ),
            'mortal_vs_proxy': (
                'Control 2: Proxy Mortal vs Proxy (Mortality as Grounding)',
                'Both modes see proxy observations and receive proxy rewards, but proxy_mortal '
                'agents can die during training. Tests whether mortality provides a "reality check" '
                'that helps ground agents despite misaligned reward signals.',
                'Proxy Mortal', 'Proxy'
            ),
            'blinded_vs_mortal': (
                'Control 3: Blinded vs Proxy Mortal (Reward vs Mortality)',
                'Both modes can die and see proxy observations. Blinded has correct reward signals; '
                'proxy_mortal has incorrect rewards but experienced mortality during training. '
                'Tests which factor matters more for survival.',
                'Blinded', 'Proxy Mortal'
            ),
        }

        for pair_key, pair_comparisons in self.comparisons.items():
            if pair_key not in comparison_info:
                continue

            title, description, label_a, label_b = comparison_info[pair_key]

            lines.append(f'### {title}')
            lines.append('')
            lines.append(f'> {description}')
            lines.append('')

            # Show efficiency as the key metric with full detail
            if 'efficiency' in pair_comparisons:
                comp = pair_comparisons['efficiency']
                lines.append('**Efficiency** (% food of total consumed):')
                lines.append('')
                lines.extend(self._format_comparison_block(comp, label_a, label_b))
                lines.append('')

            # Show survival and poison as secondary metrics in compact form
            secondary = []
            if 'survival' in pair_comparisons:
                comp = pair_comparisons['survival']
                secondary.append(f'**Survival**: {label_a} {comp.mean_a:.1f} vs {label_b} {comp.mean_b:.1f} steps '
                               f'(d={comp.cohens_d:.2f}, p={format_p_value(comp.p_value)})')
            if 'poison' in pair_comparisons:
                comp = pair_comparisons['poison']
                secondary.append(f'**Poison**: {label_a} {comp.mean_a:.1f} vs {label_b} {comp.mean_b:.1f} eaten '
                               f'(d={comp.cohens_d:.2f}, p={format_p_value(comp.p_value)})')

            if secondary:
                lines.append('Secondary metrics:')
                for s in secondary:
                    lines.append(f'- {s}')
                lines.append('')

        return '\n'.join(lines)

    def _generate_power_analysis(self, mode_data: dict) -> str:
        """Generate power analysis section for all comparison pairs."""
        lines = ['## Power Analysis', '']

        # Check if we have the primary comparison
        gt_vs_proxy = self.comparisons.get('gt_vs_proxy', {})
        if 'efficiency' not in gt_vs_proxy:
            lines.append('*Power analysis requires efficiency comparison data*')
            return '\n'.join(lines)

        # Primary comparison power analysis
        comp = gt_vs_proxy['efficiency']
        n_gt = len(mode_data.get('ground_truth', []))
        n_px = len(mode_data.get('proxy', []))
        effect_size = abs(comp.cohens_d)

        lines.append('### Achieved Power (Primary Comparison)')
        lines.append('')
        pwr = achieved_power(n_gt, n_px, effect_size)
        lines.append(f'For the ground_truth vs proxy comparison with effect size d = {effect_size:.2f} '
                    f'and sample sizes (n_gt = {n_gt}, n_proxy = {n_px}):')
        lines.append('')
        lines.append(f'- **Achieved power: {pwr:.1%}**')
        lines.append('')

        if pwr >= 0.80:
            lines.append('> The primary comparison has adequate power (>= 80%).')
        else:
            lines.append(f'> The primary comparison is underpowered. For 80% power, '
                        f'approximately {power_analysis(effect_size).n_per_group} samples per mode needed.')
        lines.append('')

        # Power summary for all comparisons
        lines.append('### Power Summary (All Comparisons)')
        lines.append('')
        lines.append('| Comparison | Effect Size (d) | Power | Adequate? |')
        lines.append('|------------|-----------------|-------|-----------|')

        comparison_labels = {
            'gt_vs_proxy': 'GT vs Proxy',
            'gt_vs_blinded': 'GT vs Blinded',
            'mortal_vs_proxy': 'Mortal vs Proxy',
            'blinded_vs_mortal': 'Blinded vs Mortal',
        }

        sample_sizes = {
            'ground_truth': len(mode_data.get('ground_truth', [])),
            'proxy': len(mode_data.get('proxy', [])),
            'ground_truth_blinded': len(mode_data.get('ground_truth_blinded', [])),
            'proxy_mortal': len(mode_data.get('proxy_mortal', [])),
        }

        pair_modes = {
            'gt_vs_proxy': ('ground_truth', 'proxy'),
            'gt_vs_blinded': ('ground_truth', 'ground_truth_blinded'),
            'mortal_vs_proxy': ('proxy_mortal', 'proxy'),
            'blinded_vs_mortal': ('ground_truth_blinded', 'proxy_mortal'),
        }

        for pair_key, label in comparison_labels.items():
            if pair_key not in self.comparisons:
                continue
            pair_comps = self.comparisons[pair_key]
            if 'efficiency' not in pair_comps:
                continue

            eff_comp = pair_comps['efficiency']
            d = abs(eff_comp.cohens_d)
            mode_a, mode_b = pair_modes[pair_key]
            n_a, n_b = sample_sizes.get(mode_a, 0), sample_sizes.get(mode_b, 0)

            if n_a > 0 and n_b > 0:
                pwr = achieved_power(n_a, n_b, d)
                adequate = 'Yes' if pwr >= 0.80 else 'No'
                lines.append(f'| {label} | {d:.2f} | {pwr:.1%} | {adequate} |')

        lines.append('')

        # Sample size guidance
        lines.append('### Sample Size Guidance for Future Experiments')
        lines.append('')
        lines.append('| Effect Size (d) | N per group (80% power) | N per group (95% power) |')
        lines.append('|-----------------|------------------------|------------------------|')

        from goodharts.analysis.power import required_sample_size
        for d in [0.5, 0.8, 1.0, 1.5, 2.0]:
            n80 = required_sample_size(d, power=0.80)
            n95 = required_sample_size(d, power=0.95)
            lines.append(f'| {d:.1f} | {n80} | {n95} |')

        lines.append('')
        return '\n'.join(lines)

    def _generate_figures_section(self) -> str:
        """Generate figures section with embedded images."""
        lines = ['## Figures', '']

        if not self.figure_paths:
            lines.append('*No figures available*')
            return '\n'.join(lines)

        # Group figures by type
        figure_groups = {
            'comparison': [],
            'distribution': [],
            'summary': [],
            'other': [],
        }

        for path in sorted(self.figure_paths):
            name = path.stem
            if 'comparison' in name:
                figure_groups['comparison'].append(path)
            elif 'distribution' in name:
                figure_groups['distribution'].append(path)
            elif 'summary' in name or 'goodhart' in name:
                figure_groups['summary'].append(path)
            else:
                figure_groups['other'].append(path)

        # Output in order
        for group_name, paths in figure_groups.items():
            if not paths:
                continue

            lines.append(f'### {group_name.replace("_", " ").title()} Plots')
            lines.append('')

            for path in paths:
                # Use relative path from report location
                rel_path = f'figures/{path.name}'
                caption = path.stem.replace('_', ' ').title()
                lines.append(f'![{caption}]({rel_path})')
                lines.append('')
                lines.append(f'*Figure: {caption}*')
                lines.append('')

        return '\n'.join(lines)

    def _generate_methodology(self) -> str:
        """Generate methodology section."""
        lines = ['## Methodology', '']

        lines.append('### Experimental Design')
        lines.append('')
        lines.append('This experiment uses a **continuous survival paradigm** where agents '
                    'navigate a 2D grid world distinguishing food from poison. The key '
                    'innovation is that agents run continuously until death (energy depletion), '
                    'then auto-respawn, allowing natural measurement of survival behavior.')
        lines.append('')

        lines.append('### Training Modes (Factorial Design)')
        lines.append('')
        lines.append('The experiment uses four training modes that vary along three dimensions: '
                    'observation quality, reward alignment, and training mortality.')
        lines.append('')
        lines.append('| Mode | Observations | Reward | Can Die (Train) |')
        lines.append('|------|--------------|--------|-----------------|')
        lines.append('| ground_truth | One-hot cell types | Energy-based | Yes |')
        lines.append('| ground_truth_blinded | Proxy (interestingness) | Energy-based | Yes |')
        lines.append('| proxy_mortal | Proxy (interestingness) | Interestingness | Yes |')
        lines.append('| proxy | Proxy (interestingness) | Interestingness | No (immortal) |')
        lines.append('')
        lines.append('**Mode descriptions:**')
        lines.append('')
        lines.append('- **Ground Truth (Baseline)**: Agents see reality (one-hot cell types) and '
                    'receive energy-based rewards. Fully aligned - can distinguish food from poison.')
        lines.append('- **Ground Truth Blinded**: Agents cannot see cell types (only interestingness values) '
                    'but receive correct energy-based rewards. Tests whether observation quality matters.')
        lines.append('- **Proxy Mortal**: Agents see proxy observations and receive interestingness-based rewards, '
                    'but can die during training. Tests whether mortality provides grounding despite wrong rewards.')
        lines.append('- **Proxy (Goodhart Case)**: Agents see proxy observations, receive interestingness rewards, '
                    'and are immortal during training. Full misalignment - optimizes proxy with no reality check.')
        lines.append('')
        lines.append('**Control comparisons:**')
        lines.append('')
        lines.append('- GT vs Proxy: Primary effect (full alignment gap)')
        lines.append('- GT vs Blinded: Impact of observation quality')
        lines.append('- Mortal vs Proxy: Impact of training mortality')
        lines.append('- Blinded vs Mortal: Reward alignment vs mortality (which helps more?)')
        lines.append('')

        lines.append('### Key Metrics')
        lines.append('')
        lines.append('- **Energy per 1k Steps**: Net energy change per 1000 steps. THE key metric. '
                    'Positive values mean agents are thriving (gaining energy faster than losing). '
                    'Negative values mean agents are dying (hemorrhaging energy). '
                    'Formula: (food_reward * food - poison_penalty * poison - move_cost * steps) / steps * 1000')
        lines.append('- **Efficiency**: Food consumed / Total consumed. Secondary metric. '
                    'Below 50% means worse than random chance.')
        lines.append('- **Survival Time**: Steps lived before each death. Higher is better.')
        lines.append('- **Deaths per 1000 Steps**: Population death rate. Lower is better.')
        lines.append('- **Survival Collapse Ratio (SCR)**: GT_survival / Proxy_survival. '
                    'Captures the magnitude of catastrophic failure.')
        lines.append('')

        lines.append('### Statistical Methods')
        lines.append('')
        lines.append("- **Welch's t-test**: Used for comparing means between groups (robust "
                    "to unequal variances and unequal sample sizes)")
        lines.append("- **Cohen's d**: Standardized effect size measure")
        lines.append("- **95% Confidence Intervals**: Computed using t-distribution")
        lines.append('')

        lines.append('### Sample Size Asymmetry')
        lines.append('')
        lines.append("Sample sizes differ between modes because each sample represents one agent's "
                    "lifetime (death or survival to evaluation end). Ground truth agents rarely die, "
                    "so most samples are survivors who lived the full evaluation period. Proxy agents "
                    "die frequently, generating many more death events. This asymmetry is expected "
                    "and reflects the core Goodhart effect. Welch's t-test handles unequal sample "
                    "sizes appropriately.")
        lines.append('')

        return '\n'.join(lines)

    def generate(self) -> Path:
        """
        Generate the complete markdown report.

        Returns:
            Path to the generated report
        """
        # Setup directories
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)

        # Extract and analyze data
        mode_data = self._extract_mode_data()
        self._compute_comparisons(mode_data)

        # Generate figures if requested
        if self.config.include_figures and mode_data:
            from goodharts.analysis.visualize import generate_all_figures
            self.figure_paths = generate_all_figures(
                mode_data,
                self.config.figures_dir,
                annotated=True,
                distributions=True,
            )

        # Build report sections
        sections = []

        # Title
        sections.append(f'# {self.config.title}')
        sections.append('')
        sections.append(f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')
        sections.append('')

        # Executive summary
        sections.append(self._generate_executive_summary(mode_data))

        # Statistical analysis
        sections.append(self._generate_statistical_analysis())

        # Power analysis
        if self.config.include_power_analysis:
            sections.append(self._generate_power_analysis(mode_data))

        # Figures
        if self.config.include_figures:
            sections.append(self._generate_figures_section())

        # Methodology
        if self.config.include_methodology:
            sections.append(self._generate_methodology())

        # Write report
        report_content = '\n'.join(sections)
        with open(self.config.report_path, 'w') as f:
            f.write(report_content)

        print(f"Report generated: {self.config.report_path}")
        return self.config.report_path

    def generate_console_summary(self) -> str:
        """
        Generate a console-printable summary of key findings.

        Returns:
            Formatted string for terminal output
        """
        lines = []
        results = self.data.get('results', {})

        # Survival Collapse Ratio header
        gt_agg = results.get('ground_truth', {}).get('aggregates', {})
        px_agg = results.get('proxy', {}).get('aggregates', {})

        if gt_agg and px_agg:
            gt_surv = gt_agg.get('survival_mean', 1)
            px_surv = px_agg.get('survival_mean', 1)
            scr = gt_surv / px_surv if px_surv > 0 else float('inf')

            lines.append('=' * 80)
            lines.append(f"{'SURVIVAL COLLAPSE RATIO: ' + f'{scr:.0f}x':^80}")
            lines.append('=' * 80)
        else:
            lines.append('=' * 80)
            lines.append(f"{'EVALUATION RESULTS':^80}")
            lines.append('=' * 80)

        # Results table - use aggregate stats (not per-death which gives different values)
        lines.append('')
        lines.append(f"{'Mode':<22} {'Energy/1k':>12} {'Efficiency':>12} {'Survival':>12} {'Deaths/1k':>12}")
        lines.append('-' * 80)

        for mode in _sort_modes(results.keys()):
            agg = results[mode].get('aggregates', {})
            if not agg:
                continue

            energy = _calculate_energy_per_1k(agg)
            energy_str = f"{'+' if energy >= 0 else ''}{energy:.1f}"
            eff = agg.get('overall_efficiency', 0)
            surv = agg.get('survival_mean', 0)
            d1k = agg.get('deaths_per_1k_steps', 0)

            lines.append(f'{mode:<22} {energy_str:>12} {eff:>11.1%} {surv:>12.1f} {d1k:>12.2f}')

        lines.append('=' * 80)

        return '\n'.join(lines)


def main():
    """CLI for report generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('--input', required=True, help='JSON file from evaluate.py')
    parser.add_argument('--output-dir', default='generated/reports',
                        help='Output directory (default: generated/reports)')
    parser.add_argument('--title', default="Goodhart's Law Experiment Results",
                        help='Report title')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--no-power', action='store_true',
                        help='Skip power analysis section')
    parser.add_argument('--console', action='store_true',
                        help='Print console summary only')

    args = parser.parse_args()

    config = ReportConfig(
        title=args.title,
        output_dir=Path(args.output_dir),
        include_figures=not args.no_figures,
        include_power_analysis=not args.no_power,
    )

    generator = ReportGenerator(config)
    generator.add_data(args.input)

    if args.console:
        print(generator.generate_console_summary())
    else:
        report_path = generator.generate()
        print(f"\nReport saved to: {report_path}")
        print(generator.generate_console_summary())


if __name__ == '__main__':
    main()
