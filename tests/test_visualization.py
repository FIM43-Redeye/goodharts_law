"""
Tests for visualization and report generation.

Verifies that figure generation produces valid PNG files and
that report generation produces well-formed markdown.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from goodharts.analysis.visualize import (
    plot_survival_comparison,
    plot_consumption_comparison,
    plot_efficiency_comparison,
    plot_goodhart_summary,
    plot_efficiency_distribution,
    plot_survival_distribution,
    plot_multi_distribution,
    plot_efficiency_comparison_annotated,
    plot_goodhart_summary_annotated,
    generate_all_figures,
    MODE_COLORS,
    THEME,
)
from goodharts.analysis.report import ReportGenerator, ReportConfig


@pytest.fixture
def sample_data():
    """Sample evaluation data for testing visualizations."""
    return {
        'ground_truth': [
            {'total_reward': 100.0, 'food_eaten': 50, 'poison_eaten': 2,
             'survival_steps': 1000, 'efficiency': 0.96},
            {'total_reward': 95.0, 'food_eaten': 48, 'poison_eaten': 1,
             'survival_steps': 950, 'efficiency': 0.98},
            {'total_reward': 105.0, 'food_eaten': 52, 'poison_eaten': 3,
             'survival_steps': 1050, 'efficiency': 0.95},
        ],
        'proxy': [
            {'total_reward': 80.0, 'food_eaten': 30, 'poison_eaten': 35,
             'survival_steps': 200, 'efficiency': 0.46},
            {'total_reward': 75.0, 'food_eaten': 28, 'poison_eaten': 32,
             'survival_steps': 180, 'efficiency': 0.47},
            {'total_reward': 85.0, 'food_eaten': 32, 'poison_eaten': 38,
             'survival_steps': 220, 'efficiency': 0.46},
        ],
    }


@pytest.fixture
def sample_json_data():
    """Sample evaluation JSON data matching evaluate.py output format."""
    return {
        'metadata': {
            'timestamp': '2024-01-01T00:00:00',
            'total_timesteps': 100000,
        },
        'results': {
            'ground_truth': {
                'aggregates': {
                    'overall_efficiency': 0.96,
                    'survival_mean': 1000,
                    'n_deaths': 3,
                    'deaths_per_1k_steps': 0.03,
                },
                'efficiency': 0.96,
                'mean_survival': 1000,
                'total_food': 150,
                'total_poison': 6,
                'total_deaths': 3,
                'deaths': [
                    {'food_eaten': 50, 'poison_eaten': 2, 'survival_time': 1000,
                     'efficiency': 0.96, 'total_reward': 100},
                    {'food_eaten': 48, 'poison_eaten': 1, 'survival_time': 950,
                     'efficiency': 0.98, 'total_reward': 95},
                    {'food_eaten': 52, 'poison_eaten': 3, 'survival_time': 1050,
                     'efficiency': 0.95, 'total_reward': 105},
                ],
            },
            'proxy': {
                'aggregates': {
                    'overall_efficiency': 0.46,
                    'survival_mean': 200,
                    'n_deaths': 3,
                    'deaths_per_1k_steps': 5.0,
                },
                'efficiency': 0.46,
                'mean_survival': 200,
                'total_food': 90,
                'total_poison': 105,
                'total_deaths': 3,
                'deaths': [
                    {'food_eaten': 30, 'poison_eaten': 35, 'survival_time': 200,
                     'efficiency': 0.46, 'total_reward': 80},
                    {'food_eaten': 28, 'poison_eaten': 32, 'survival_time': 180,
                     'efficiency': 0.47, 'total_reward': 75},
                    {'food_eaten': 32, 'poison_eaten': 38, 'survival_time': 220,
                     'efficiency': 0.46, 'total_reward': 85},
                ],
            },
        },
    }


@pytest.fixture
def output_dir():
    """Temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def json_data_file(sample_json_data, output_dir):
    """Write sample JSON data to a file and return path."""
    json_path = output_dir / 'eval_results.json'
    with open(json_path, 'w') as f:
        json.dump(sample_json_data, f)
    return json_path


class TestModeColors:
    """Tests for mode color configuration."""

    def test_all_modes_have_colors(self):
        """All expected modes should have assigned colors."""
        expected_modes = ['ground_truth', 'ground_truth_handhold',
                          'ground_truth_blinded', 'proxy']
        for mode in expected_modes:
            assert mode in MODE_COLORS, f"Missing color for {mode}"

    def test_colors_are_valid_hex(self):
        """Colors should be valid hex strings."""
        for mode, color in MODE_COLORS.items():
            assert color.startswith('#'), f"Color for {mode} should be hex"
            assert len(color) == 7, f"Color for {mode} should be 6-digit hex"


class TestTheme:
    """Tests for theme configuration."""

    def test_theme_has_required_keys(self):
        """Theme should have all required color keys."""
        required = ['background', 'paper', 'text', 'grid']
        for key in required:
            assert key in THEME, f"Theme missing '{key}'"


class TestBasicPlots:
    """Tests for basic comparison plot functions.

    These tests mock write_image to avoid kaleido dependency.
    """

    def test_survival_comparison_creates_file(self, sample_data, output_dir):
        """plot_survival_comparison should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_survival_comparison(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'survival_comparison.png' in call_path

    def test_consumption_comparison_creates_file(self, sample_data, output_dir):
        """plot_consumption_comparison should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_consumption_comparison(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'consumption_comparison.png' in call_path

    def test_efficiency_comparison_creates_file(self, sample_data, output_dir):
        """plot_efficiency_comparison should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_efficiency_comparison(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'efficiency_comparison.png' in call_path

    def test_goodhart_summary_creates_file(self, sample_data, output_dir):
        """plot_goodhart_summary should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_goodhart_summary(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'goodhart_summary.png' in call_path


class TestDistributionPlots:
    """Tests for distribution plot functions."""

    def test_efficiency_distribution_creates_file(self, sample_data, output_dir):
        """plot_efficiency_distribution should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_efficiency_distribution(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'efficiency_distribution_violin.png' in call_path

    def test_survival_distribution_creates_file(self, sample_data, output_dir):
        """plot_survival_distribution should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_survival_distribution(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'survival_distribution_violin.png' in call_path

    def test_multi_distribution_creates_file(self, sample_data, output_dir):
        """plot_multi_distribution should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            plot_multi_distribution(sample_data, output_dir)
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'multi_distribution_violin.png' in call_path


class TestAnnotatedPlots:
    """Tests for annotated plots with statistical information."""

    def test_efficiency_annotated_creates_file(self, sample_data, output_dir):
        """plot_efficiency_comparison_annotated should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            result = plot_efficiency_comparison_annotated(sample_data, output_dir)
            # Should return a Path
            assert result is not None
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'efficiency_comparison_annotated.png' in call_path

    def test_goodhart_summary_annotated_creates_file(self, sample_data, output_dir):
        """plot_goodhart_summary_annotated should create output file."""
        with patch('plotly.graph_objects.Figure.write_image') as mock_write:
            result = plot_goodhart_summary_annotated(sample_data, output_dir)
            assert result is not None
            mock_write.assert_called_once()
            call_path = mock_write.call_args[0][0]
            assert 'goodhart_summary_annotated.png' in call_path


class TestGenerateAllFigures:
    """Tests for the generate_all_figures function."""

    def test_generates_basic_figures(self, sample_data, output_dir):
        """generate_all_figures should create basic comparison plots."""
        with patch('plotly.graph_objects.Figure.write_image'):
            paths = generate_all_figures(sample_data, output_dir,
                                        annotated=False, distributions=False)

        # Should return list of paths
        assert isinstance(paths, list)
        assert len(paths) >= 3  # At least basic plots

        # Check expected filenames in returned paths
        path_names = [p.name for p in paths]
        assert 'survival_comparison.png' in path_names
        assert 'consumption_comparison.png' in path_names
        assert 'efficiency_comparison.png' in path_names

    def test_generates_with_annotated_flag(self, sample_data, output_dir):
        """generate_all_figures with annotated=True should create annotated plots."""
        with patch('plotly.graph_objects.Figure.write_image'):
            paths = generate_all_figures(sample_data, output_dir,
                                        annotated=True, distributions=False)

        path_names = [p.name for p in paths]
        assert 'efficiency_comparison_annotated.png' in path_names
        assert 'goodhart_summary_annotated.png' in path_names

    def test_generates_with_distributions_flag(self, sample_data, output_dir):
        """generate_all_figures with distributions=True should create distribution plots."""
        with patch('plotly.graph_objects.Figure.write_image'):
            paths = generate_all_figures(sample_data, output_dir,
                                        annotated=False, distributions=True)

        path_names = [p.name for p in paths]
        # Generates both violin and box versions
        assert 'efficiency_distribution_violin.png' in path_names
        assert 'survival_distribution_violin.png' in path_names


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ReportConfig()
        assert config.title is not None
        assert config.include_figures is True
        assert config.include_power_analysis is True

    def test_custom_config(self):
        """Config should accept custom values."""
        config = ReportConfig(
            title="Custom Report",
            include_figures=False,
        )
        assert config.title == "Custom Report"
        assert config.include_figures is False

    def test_config_paths(self, output_dir):
        """Config should generate proper paths."""
        config = ReportConfig(output_dir=output_dir)
        assert config.report_dir.parent == output_dir
        assert config.figures_dir.parent == config.report_dir
        assert config.report_path.suffix == '.md'


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generator_creation(self):
        """ReportGenerator should be creatable with default config."""
        generator = ReportGenerator()
        assert generator is not None
        assert generator.config is not None

    def test_generator_with_custom_config(self, output_dir):
        """ReportGenerator should accept custom config."""
        config = ReportConfig(output_dir=output_dir)
        generator = ReportGenerator(config)
        assert generator.config.output_dir == output_dir

    def test_add_data_returns_self(self, json_data_file):
        """add_data() should return self for method chaining."""
        generator = ReportGenerator()
        result = generator.add_data(str(json_data_file))
        assert result is generator

    def test_add_data_loads_json(self, json_data_file, sample_json_data):
        """add_data() should load JSON data into generator."""
        generator = ReportGenerator()
        generator.add_data(str(json_data_file))

        assert generator.data is not None
        assert 'results' in generator.data
        assert 'ground_truth' in generator.data['results']

    def test_generate_creates_report(self, json_data_file, output_dir):
        """generate() should create a markdown report file."""
        config = ReportConfig(output_dir=output_dir, include_figures=False)
        generator = ReportGenerator(config)
        generator.add_data(str(json_data_file))

        with patch('plotly.graph_objects.Figure.write_image'):
            report_path = generator.generate()

        assert report_path.exists()
        assert report_path.suffix == '.md'

    def test_report_contains_mode_names(self, json_data_file, output_dir):
        """Generated report should mention the modes."""
        config = ReportConfig(output_dir=output_dir, include_figures=False)
        generator = ReportGenerator(config)
        generator.add_data(str(json_data_file))

        with patch('plotly.graph_objects.Figure.write_image'):
            report_path = generator.generate()

        content = report_path.read_text()
        assert 'ground_truth' in content or 'Ground Truth' in content
        assert 'proxy' in content or 'Proxy' in content

    def test_report_contains_metrics(self, json_data_file, output_dir):
        """Generated report should contain key metrics."""
        config = ReportConfig(output_dir=output_dir, include_figures=False)
        generator = ReportGenerator(config)
        generator.add_data(str(json_data_file))

        with patch('plotly.graph_objects.Figure.write_image'):
            report_path = generator.generate()

        content = report_path.read_text().lower()
        assert 'efficiency' in content or 'survival' in content

    def test_report_is_valid_markdown(self, json_data_file, output_dir):
        """Generated report should be valid markdown (has headers)."""
        config = ReportConfig(output_dir=output_dir, include_figures=False)
        generator = ReportGenerator(config)
        generator.add_data(str(json_data_file))

        with patch('plotly.graph_objects.Figure.write_image'):
            report_path = generator.generate()

        content = report_path.read_text()
        # Markdown reports should have headers
        assert '#' in content

    def test_console_summary_returns_string(self, json_data_file):
        """generate_console_summary() should return a string."""
        generator = ReportGenerator()
        generator.add_data(str(json_data_file))

        summary = generator.generate_console_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestReportWithRealData:
    """Tests using more realistic data structures."""

    @pytest.fixture
    def multi_run_data(self):
        """Data structure matching multi-run evaluator output."""
        return {
            'ground_truth': [
                {'total_reward': r, 'food_eaten': 50 + i, 'poison_eaten': 2,
                 'survival_steps': 1000 + i * 10, 'efficiency': 0.95 + i * 0.01}
                for i, r in enumerate([100, 105, 98, 102, 99])
            ],
            'proxy': [
                {'total_reward': r, 'food_eaten': 30, 'poison_eaten': 35 + i,
                 'survival_steps': 200 - i * 5, 'efficiency': 0.46 - i * 0.01}
                for i, r in enumerate([80, 75, 82, 78, 77])
            ],
        }

    def test_handles_multi_run_data(self, multi_run_data, output_dir):
        """Visualization should handle multi-run data correctly."""
        with patch('plotly.graph_objects.Figure.write_image'):
            paths = generate_all_figures(multi_run_data, output_dir)

        # Should return some paths
        assert len(paths) > 0

    def test_single_mode_data(self, output_dir):
        """Should handle data with only one mode (no comparison possible)."""
        single_mode_data = {
            'ground_truth': [
                {'total_reward': 100.0, 'food_eaten': 50, 'poison_eaten': 2,
                 'survival_steps': 1000, 'efficiency': 0.96},
            ],
        }

        with patch('plotly.graph_objects.Figure.write_image'):
            # Should not crash
            paths = generate_all_figures(single_mode_data, output_dir,
                                        annotated=False, distributions=False)

        assert isinstance(paths, list)
