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
    compute_goodhart_failure_index,
    format_p_value,
)
from goodharts.analysis.power import power_analysis, achieved_power


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

    def _compute_comparisons(self, mode_data: dict) -> dict[str, StatisticalComparison]:
        """Compute statistical comparisons for ground_truth vs proxy."""
        comparisons = {}

        if 'ground_truth' not in mode_data or 'proxy' not in mode_data:
            return comparisons

        gt = mode_data['ground_truth']
        px = mode_data['proxy']

        # Efficiency comparison (the key Goodhart metric)
        gt_eff = [d['efficiency'] * 100 for d in gt]
        px_eff = [d['efficiency'] * 100 for d in px]
        comparisons['efficiency'] = compute_comparison(
            gt_eff, px_eff,
            metric='efficiency',
            group_a='ground_truth',
            group_b='proxy'
        )

        # Survival comparison
        gt_surv = [d['survival_steps'] for d in gt]
        px_surv = [d['survival_steps'] for d in px]
        comparisons['survival'] = compute_comparison(
            gt_surv, px_surv,
            metric='survival',
            group_a='ground_truth',
            group_b='proxy'
        )

        # Poison consumption
        gt_poison = [d['poison_eaten'] for d in gt]
        px_poison = [d['poison_eaten'] for d in px]
        comparisons['poison'] = compute_comparison(
            gt_poison, px_poison,
            metric='poison_eaten',
            group_a='ground_truth',
            group_b='proxy'
        )

        self.comparisons = comparisons
        return comparisons

    def _generate_executive_summary(self, mode_data: dict) -> str:
        """Generate executive summary section."""
        lines = ['## Executive Summary', '']

        # Goodhart Failure Index - use aggregate efficiency from results (not per-death mean)
        results = self.data.get('results', {})
        gt_agg = results.get('ground_truth', {}).get('aggregates', {})
        px_agg = results.get('proxy', {}).get('aggregates', {})

        if gt_agg and px_agg:
            gt_eff = gt_agg.get('overall_efficiency', 0)
            px_eff = px_agg.get('overall_efficiency', 0)
            gfi = compute_goodhart_failure_index(gt_eff, px_eff)

            lines.append(f'**Goodhart Failure Index: {gfi:.1%}**')
            lines.append('')
            lines.append('> The Goodhart Failure Index measures how much performance '
                        'degrades when optimizing for a proxy metric instead of the '
                        'true objective. A GFI of 0% indicates no failure; 100% indicates '
                        'complete failure.')
            lines.append('')

        # Summary table - show both aggregate and per-agent efficiency
        lines.append('### Performance Summary')
        lines.append('')
        lines.append('| Mode | Aggregate Eff. | Per-Agent Eff. | Survival | Deaths | Deaths/1k |')
        lines.append('|------|----------------|----------------|----------|--------|-----------|')

        results = self.data.get('results', {})
        mode_data = self._extract_mode_data()
        for mode in sorted(results.keys()):
            agg = results[mode].get('aggregates', {})
            if agg:
                agg_eff = agg.get('overall_efficiency', 0)
                surv = agg.get('survival_mean', 0)
                # Multi-run aggregates use 'total_deaths', single-run uses 'n_deaths'
                deaths = agg.get('total_deaths', agg.get('n_deaths', 0))
                d1k = agg.get('deaths_per_1k_steps', 0)
                # Compute per-agent mean efficiency from extracted data
                if mode in mode_data and mode_data[mode]:
                    individual_effs = [d['efficiency'] for d in mode_data[mode]]
                    agent_eff = sum(individual_effs) / len(individual_effs)
                else:
                    agent_eff = agg_eff
                lines.append(f'| {mode} | {agg_eff:.1%} | {agent_eff:.1%} | {surv:.1f} | {deaths:,} | {d1k:.2f} |')

        lines.append('')
        lines.append('> *Aggregate Eff.*: Total food / total consumed (weights by volume)')
        lines.append('> *Per-Agent Eff.*: Mean of individual agent efficiencies (weights equally)')
        lines.append('')
        return '\n'.join(lines)

    def _generate_statistical_analysis(self) -> str:
        """Generate statistical analysis section."""
        lines = ['## Statistical Analysis', '']

        if not self.comparisons:
            lines.append('*No comparison data available (requires both ground_truth and proxy modes)*')
            return '\n'.join(lines)

        lines.append('### Ground Truth vs Proxy Comparison')
        lines.append('')

        # Note about which metric is used
        lines.append('> Statistical tests use per-agent means (see "Per-Agent Eff." in summary table).')
        lines.append('')

        for metric, comp in self.comparisons.items():
            lines.append(f'#### {metric.replace("_", " ").title()}')
            lines.append('')
            lines.append(f'- **Ground Truth**: {comp.mean_a:.2f} (95% CI: [{comp.ci_a[0]:.2f}, {comp.ci_a[1]:.2f}])')
            lines.append(f'- **Proxy**: {comp.mean_b:.2f} (95% CI: [{comp.ci_b[0]:.2f}, {comp.ci_b[1]:.2f}])')
            lines.append(f'- **Difference**: {comp.mean_diff:+.2f} (95% CI: [{comp.ci_diff[0]:.2f}, {comp.ci_diff[1]:.2f}])')
            lines.append('')
            lines.append(f'Statistical test (Welch\'s t-test):')
            lines.append(f'- t({comp.df:.1f}) = {comp.t_statistic:.2f}')
            lines.append(f'- p = {format_p_value(comp.p_value)} {comp.significance_stars}')
            lines.append(f'- Cohen\'s d = {comp.cohens_d:.2f} ({comp.effect_magnitude} effect)')
            lines.append('')

        return '\n'.join(lines)

    def _generate_power_analysis(self, mode_data: dict) -> str:
        """Generate power analysis section."""
        lines = ['## Power Analysis', '']

        if 'efficiency' not in self.comparisons:
            lines.append('*Power analysis requires efficiency comparison data*')
            return '\n'.join(lines)

        comp = self.comparisons['efficiency']
        n_gt = len(mode_data.get('ground_truth', []))
        n_px = len(mode_data.get('proxy', []))
        effect_size = abs(comp.cohens_d)

        # Achieved power
        pwr = achieved_power(n_gt, n_px, effect_size)
        lines.append(f'### Achieved Power')
        lines.append('')
        lines.append(f'With the observed effect size (d = {effect_size:.2f}) and sample sizes '
                    f'(n_gt = {n_gt}, n_proxy = {n_px}):')
        lines.append('')
        lines.append(f'- **Achieved power: {pwr:.1%}**')
        lines.append('')

        if pwr >= 0.80:
            lines.append('> The study has adequate power (>= 80%) to detect the observed effect.')
        else:
            lines.append(f'> The study is underpowered. For 80% power with this effect size, '
                        f'approximately {power_analysis(effect_size).n_per_group} deaths per mode '
                        f'would be needed.')

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

        lines.append('### Training Modes')
        lines.append('')
        lines.append('- **Ground Truth**: Agents receive one-hot encoded cell type observations '
                    '(can see if cells are food, poison, or empty) and energy-based rewards.')
        lines.append('- **Proxy**: Agents receive only "interestingness" values (a scalar proxy '
                    'for cell type) and interestingness-based rewards. This simulates optimizing '
                    'for a measurable but imperfect proxy metric.')
        lines.append('')

        lines.append('### Key Metrics')
        lines.append('')
        lines.append('- **Efficiency**: Food consumed / Total consumed. Measures how well '
                    'agents distinguish food from poison.')
        lines.append('- **Survival Time**: Steps lived before each death. Higher is better.')
        lines.append('- **Deaths per 1000 Steps**: Population death rate. Lower is better.')
        lines.append('- **Goodhart Failure Index (GFI)**: (GT_efficiency - Proxy_efficiency) / GT_efficiency')
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

        # Goodhart Failure Index header - use aggregate efficiency
        gt_agg = results.get('ground_truth', {}).get('aggregates', {})
        px_agg = results.get('proxy', {}).get('aggregates', {})

        if gt_agg and px_agg:
            gt_eff = gt_agg.get('overall_efficiency', 0)
            px_eff = px_agg.get('overall_efficiency', 0)
            gfi = compute_goodhart_failure_index(gt_eff, px_eff)

            lines.append('=' * 80)
            lines.append(f"{'GOODHART FAILURE INDEX: ' + f'{gfi:.1%}':^80}")
            lines.append('=' * 80)
        else:
            lines.append('=' * 80)
            lines.append(f"{'EVALUATION RESULTS':^80}")
            lines.append('=' * 80)

        # Results table - use aggregate stats (not per-death which gives different values)
        lines.append('')
        lines.append(f"{'Mode':<25} {'Efficiency':>12} {'Survival':>12} {'Deaths':>12} {'Deaths/1k':>12}")
        lines.append('-' * 80)

        for mode in sorted(results.keys()):
            agg = results[mode].get('aggregates', {})
            if not agg:
                continue

            eff = agg.get('overall_efficiency', 0)
            surv = agg.get('survival_mean', 0)
            deaths = agg.get('total_deaths', agg.get('n_deaths', 0))
            d1k = agg.get('deaths_per_1k_steps', 0)

            lines.append(f'{mode:<25} {eff:>11.1%} {surv:>12.1f} {deaths:>12,} {d1k:>12.2f}')

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
