import pytest
from goodharts.behaviors import create_learned_behavior, get_behavior, LEARNED_PRESETS
from goodharts.simulation import Simulation
from goodharts.configs.default_config import get_config

def test_create_learned_behavior_presets():
    """Verify all defined presets can be created."""
    for preset_name in LEARNED_PRESETS:
        behavior = create_learned_behavior(preset_name)
        assert behavior is not None
        assert behavior.requirements is not None

def test_create_learned_behavior_invalid():
    """Verify invalid preset validation."""
    with pytest.raises(ValueError, match="Unknown preset"):
        create_learned_behavior("non_existent_preset")

def test_simulation_loads_learned_preset():
    """Verify Simulation can load a learned behavior by preset name."""
    config = get_config()
    # Replace default agents with a learned preset
    config['AGENTS_SETUP'] = [
        {'behavior_class': 'ground_truth', 'count': 1}
    ]
    # Reduce size
    config['GRID_WIDTH'] = 10
    config['GRID_HEIGHT'] = 10
    
    sim = Simulation(config)
    assert len(sim.agents) == 1
    # Check that it loaded a LearnedBehavior
    from goodharts.behaviors.learned import LearnedBehavior
    assert isinstance(sim.agents[0].behavior, LearnedBehavior)

def test_simulation_loads_hardcoded():
    """Verify Simulation can still load hardcoded behaviors."""
    config = get_config()
    config['AGENTS_SETUP'] = [
        {'behavior_class': 'OmniscientSeeker', 'count': 1}
    ]
    config['GRID_WIDTH'] = 10
    config['GRID_HEIGHT'] = 10
    
    sim = Simulation(config)
    assert len(sim.agents) == 1
    # Check type
    from goodharts.behaviors import OmniscientSeeker
    assert isinstance(sim.agents[0].behavior, OmniscientSeeker)

def test_legacy_names_removed():
    """Verify legacy class names are no longer importable or loadable."""
    # Attempt to use legacy name in simulation
    config = get_config()
    config['AGENTS_SETUP'] = [
        {'behavior_class': 'LearnedGroundTruth', 'count': 1}
    ]
    config['GRID_WIDTH'] = 10
    config['GRID_HEIGHT'] = 10
    
    # Needs to fail because we removed the class
    with pytest.raises((ValueError, ImportError, AttributeError)):
        # Depending on how it fails:
        # If get_behavior fails: ValueError
        # If we try to import it: ImportError/AttributeError
        Simulation(config)
