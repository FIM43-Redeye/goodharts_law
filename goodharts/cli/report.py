"""
Regenerate the final report from saved evaluation results.

Use this after modifying report.py templates or prose to regenerate
the report without re-running the full evaluation.

Usage:
    python main.py report
    python main.py report --no-figures  # Text only
    python main.py report --input path/to/results.json
"""
import argparse
from pathlib import Path

# Default paths
REPORTS_DIR = Path('generated/reports')
DEFAULT_OUTPUT = Path('generated/reports/final')


def find_latest_results() -> Path | None:
    """Find the most recent results.json in generated/reports/*/."""
    if not REPORTS_DIR.exists():
        return None

    results_files = list(REPORTS_DIR.glob('*/results.json'))
    if not results_files:
        return None

    # Sort by modification time, newest first
    results_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return results_files[0]


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate final report from saved evaluation results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=None,
        help='Path to results.json (default: auto-detect latest)',
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output directory (default: {DEFAULT_OUTPUT})',
    )
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Skip figure generation (text report only)',
    )
    parser.add_argument(
        '--keep-existing-figures',
        action='store_true',
        help='Keep existing figures instead of regenerating',
    )

    args = parser.parse_args()

    # Auto-discover latest results if not specified
    if args.input is None:
        args.input = find_latest_results()
        if args.input is None:
            print("Error: No results.json found in generated/reports/")
            print("Run a full evaluation first, or specify --input path/to/results.json")
            return 1
        print(f"Auto-detected: {args.input}")

    if not args.input.exists():
        print(f"Error: Results file not found: {args.input}")
        print("Run a full evaluation first, or specify --input path/to/results.json")
        return 1

    # Import here to avoid slow startup for --help
    from goodharts.analysis.report import ReportGenerator, ReportConfig

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / 'figures'

    # Handle figures
    generate_figures = not args.no_figures and not args.keep_existing_figures

    if args.keep_existing_figures and figures_dir.exists():
        print(f"Keeping existing figures in {figures_dir}")
        generate_figures = False
    elif not args.no_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    # Configure report generation
    # Use empty timestamp to write directly to output dir
    config = ReportConfig(
        title="Goodhart's Law Experiment Results",
        output_dir=args.output.parent,
        timestamp=args.output.name,  # Use output dir name as "timestamp"
        include_figures=generate_figures,
        include_power_analysis=True,
    )

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Figures: {'regenerating' if generate_figures else 'skipping'}")
    print()

    # Generate report
    generator = ReportGenerator(config)
    generator.add_data(str(args.input))

    if args.keep_existing_figures and figures_dir.exists():
        # Register existing figures for the report
        generator.add_figures(figures_dir)

    report_path = generator.generate()

    # Print summary
    print()
    print(generator.generate_console_summary())
    print(f"\nReport written to: {report_path}")

    return 0


if __name__ == '__main__':
    exit(main())
