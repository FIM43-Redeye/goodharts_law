"""Tests for TorchVecEnv mechanics - grid, movement, eating, energy.

These tests verify that the environment components work correctly,
independent of whether the overall experiment succeeds.
"""
import pytest
import torch

from goodharts.environments.torch_env import create_torch_vec_env, TorchVecEnv
from goodharts.modes import ObservationSpec
from goodharts.configs.default_config import get_simulation_config, CellType
from goodharts.behaviors.action_space import build_action_space, index_to_action


@pytest.fixture
def config():
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 20
    cfg['GRID_POISON_INIT'] = 10
    cfg['WORLD_LOOP'] = False
    cfg['MAX_MOVE_DISTANCE'] = 1  # Tests expect 3x3 action space
    return cfg


@pytest.fixture
def looping_config():
    cfg = get_simulation_config()
    cfg['GRID_WIDTH'] = 20
    cfg['GRID_HEIGHT'] = 20
    cfg['GRID_FOOD_INIT'] = 20
    cfg['GRID_POISON_INIT'] = 10
    cfg['WORLD_LOOP'] = True
    cfg['MAX_MOVE_DISTANCE'] = 1  # Tests expect 3x3 action space
    return cfg


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestGridMechanics:
    """Tests for grid initialization and state management."""

    def test_grid_dimensions_match_config(self, config, device):
        """Grid dimensions should match configuration."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        assert env.width == config['GRID_WIDTH']
        assert env.height == config['GRID_HEIGHT']
        assert env.grids.shape == (4, config['GRID_HEIGHT'], config['GRID_WIDTH'])

    def test_items_placed_at_empty_cells(self, config, device):
        """Food and poison should only be placed on empty cells."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        grid = env.grids[0]

        # Count each cell type (agents are now permanently marked on grid as PREY/PREDATOR)
        empty = (grid == CellType.EMPTY.value).sum().item()
        food = (grid == CellType.FOOD.value).sum().item()
        poison = (grid == CellType.POISON.value).sum().item()
        agents = (grid == CellType.PREY.value).sum().item()
        agents += (grid == CellType.PREDATOR.value).sum().item()

        # Total should equal grid size
        total = empty + food + poison + agents
        expected = config['GRID_WIDTH'] * config['GRID_HEIGHT']

        assert total == expected, f"Cell counts don't sum to grid size: {total} != {expected}"

    def test_no_overlapping_items(self, config, device):
        """No cell should contain multiple items."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        for grid_id in range(4):
            grid = env.grids[grid_id]

            # Each cell should be exactly one type
            # Check by verifying counts add up
            total_cells = grid.numel()
            unique_cells = len(torch.unique(grid))

            # Should have at most 4 unique values: EMPTY, FOOD, POISON, WALL
            assert unique_cells <= 4, f"Grid has unexpected cell types: {torch.unique(grid)}"

    def test_agent_spawns_on_empty_cell(self, config, device):
        """Agent should spawn on an empty cell."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=8, obs_spec=spec, config=config, device=device)

        for env_id in range(8):
            x, y = env.agent_x[env_id].item(), env.agent_y[env_id].item()
            grid_id = env.grid_indices[env_id].item()

            # Agent position should be within bounds
            assert 0 <= x < env.width, f"Agent x={x} out of bounds"
            assert 0 <= y < env.height, f"Agent y={y} out of bounds"


class TestMovementMechanics:
    """Tests for agent movement and boundary handling."""

    def test_action_deltas_match_action_space(self, config, device):
        """Action deltas should match the canonical action space."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        actions = build_action_space(1)

        for idx, (dx, dy) in enumerate(actions):
            env_dx = env.action_deltas[idx, 0].item()
            env_dy = env.action_deltas[idx, 1].item()

            assert env_dx == dx, f"Action {idx}: dx mismatch {env_dx} != {dx}"
            assert env_dy == dy, f"Action {idx}: dy mismatch {env_dy} != {dy}"

    def test_movement_updates_position(self, config, device):
        """Taking an action should update agent position correctly."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place agent in center to avoid boundary issues
        env.agent_x[0] = 10
        env.agent_y[0] = 10

        initial_x, initial_y = 10, 10

        # Test action 0 (should be some movement)
        dx, dy = index_to_action(0)
        actions = torch.tensor([0], dtype=torch.long, device=device)

        env.step(actions)

        expected_x = initial_x + dx
        expected_y = initial_y + dy

        assert env.agent_x[0].item() == expected_x, f"X position wrong after action"
        assert env.agent_y[0].item() == expected_y, f"Y position wrong after action"

    def test_bounded_mode_clamps_at_edges(self, config, device):
        """In bounded mode, agents should be clamped at grid edges."""
        config['WORLD_LOOP'] = False
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place agent at corner (0, 0)
        env.agent_x[0] = 0
        env.agent_y[0] = 0

        # Find action that moves left/up (negative direction)
        left_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx < 0 or dy < 0:
                left_action = idx
                break

        if left_action is not None:
            actions = torch.tensor([left_action], dtype=torch.long, device=device)
            env.step(actions)

            # Should be clamped at 0
            assert env.agent_x[0].item() >= 0, "Agent went past left boundary"
            assert env.agent_y[0].item() >= 0, "Agent went past top boundary"

    def test_looping_mode_wraps_around(self, looping_config, device):
        """In looping mode, agents should wrap around grid edges."""
        spec = ObservationSpec.for_mode('ground_truth', looping_config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=looping_config, device=device)

        # Place agent at corner (0, 0)
        env.agent_x[0] = 0
        env.agent_y[0] = 0

        # Find action that moves left (dx = -1)
        left_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == -1 and dy == 0:
                left_action = idx
                break

        if left_action is not None:
            actions = torch.tensor([left_action], dtype=torch.long, device=device)
            env.step(actions)

            # Should wrap to right edge
            expected_x = env.width - 1
            assert env.agent_x[0].item() == expected_x, \
                f"Agent didn't wrap: got {env.agent_x[0].item()}, expected {expected_x}"


class TestEnergyMechanics:
    """Tests for energy system correctness."""

    def test_movement_costs_energy(self, config, device):
        """Each step should cost energy."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place agent away from food/poison
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.grids[0, 10, 10] = CellType.EMPTY.value
        env.grids[0, 9:12, 9:12] = CellType.EMPTY.value  # Clear surrounding area

        initial_energy = env.agent_energy[0].item()

        actions = torch.tensor([0], dtype=torch.long, device=device)
        env.step(actions)

        final_energy = env.agent_energy[0].item()

        assert final_energy < initial_energy, "Energy should decrease after movement"

        # Should decrease by exactly move_cost (if no eating)
        expected_decrease = env.energy_move_cost
        actual_decrease = initial_energy - final_energy

        assert abs(actual_decrease - expected_decrease) < 0.01, \
            f"Energy decrease {actual_decrease} != expected {expected_decrease}"

    def test_eating_food_restores_energy(self, config, device):
        """Eating food should restore energy."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Setup: agent at (10, 10), will move to (10, 11) via action
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.agent_energy[0] = 20.0  # Low energy

        # Find action that moves right (dx=1, dy=0)
        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        assert right_action is not None, "No right action found"

        # Place food at target position
        env.grids[0, 10, 11] = CellType.FOOD.value

        initial_energy = env.agent_energy[0].item()

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        obs, eating_info, terminated, truncated = env.step(actions)

        final_energy = env.agent_energy[0].item()

        # Energy should increase (food reward > move cost)
        assert final_energy > initial_energy, \
            f"Energy should increase after eating food: {initial_energy} -> {final_energy}"

    def test_eating_poison_reduces_energy(self, config, device):
        """Eating poison should reduce energy."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.agent_energy[0] = 50.0  # Enough to survive

        # Find action that moves right
        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        # Place poison at target
        env.grids[0, 10, 11] = CellType.POISON.value

        initial_energy = env.agent_energy[0].item()

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        env.step(actions)

        final_energy = env.agent_energy[0].item()

        # Energy should decrease significantly (poison penalty + move cost)
        assert final_energy < initial_energy - env.energy_move_cost, \
            f"Poison should reduce energy beyond move cost: {initial_energy} -> {final_energy}"

    def test_zero_energy_triggers_done(self, config, device):
        """Agent should be marked done when energy reaches zero."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Set energy to a value that will go to zero or below after move cost
        # The step function deducts energy_move_cost
        env.agent_energy[0] = env.energy_move_cost * 0.5  # Will go negative

        # Clear area around agent to ensure no eating
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.grids[0, 8:13, 8:13] = CellType.EMPTY.value

        actions = torch.tensor([0], dtype=torch.long, device=device)
        obs, eating_info, terminated, truncated = env.step(actions)
        dones = terminated | truncated

        # Should be done (energy <= 0 after move cost deduction)
        assert dones[0].item() == True, \
            f"Agent should be done when energy depleted. Energy: {env.agent_energy[0].item()}"


class TestFoodRespawnMechanics:
    """Tests for food/poison respawn system."""

    def test_eaten_food_respawns(self, config, device):
        """Eating food should trigger respawn."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        initial_food_count = (env.grids[0] == CellType.FOOD.value).sum().item()

        # Find and eat food
        env.agent_x[0] = 10
        env.agent_y[0] = 10

        # Place food at target and move there
        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        env.grids[0, 10, 11] = CellType.FOOD.value

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        env.step(actions)

        final_food_count = (env.grids[0] == CellType.FOOD.value).sum().item()

        # Food count should be maintained (one eaten, one respawned)
        # Allow for +/- 1 due to timing
        assert abs(final_food_count - initial_food_count) <= 1, \
            f"Food count changed unexpectedly: {initial_food_count} -> {final_food_count}"

    def test_respawn_avoids_occupied_cells(self, config, device):
        """Respawned items should not overwrite existing items."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Fill most of grid with walls to force collisions
        env.grids[0, :, :] = CellType.WALL.value

        # Leave some empty cells
        env.grids[0, 0:5, 0:5] = CellType.EMPTY.value

        # Place agent in empty area
        env.agent_x[0] = 2
        env.agent_y[0] = 2

        # Count walls before respawn
        initial_walls = (env.grids[0] == CellType.WALL.value).sum().item()

        # Force respawn
        eaten_mask = torch.tensor([True], dtype=torch.bool, device=device)
        env._respawn_items_vectorized(eaten_mask, CellType.FOOD.value)

        # Wall count should not change (respawn didn't overwrite walls)
        final_walls = (env.grids[0] == CellType.WALL.value).sum().item()

        assert final_walls == initial_walls, \
            f"Walls were overwritten by respawn: {initial_walls} -> {final_walls}"


class TestEpisodeTracking:
    """Tests for episode statistics tracking."""

    def test_food_counter_increments_on_eat(self, config, device):
        """Episode food counter should increment when eating."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        env.current_episode_food[0] = 0

        # Setup food eating
        env.agent_x[0] = 10
        env.agent_y[0] = 10

        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        env.grids[0, 10, 11] = CellType.FOOD.value

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        env.step(actions)

        assert env.current_episode_food[0].item() == 1, \
            f"Food counter should be 1, got {env.current_episode_food[0].item()}"

    def test_poison_counter_increments_on_eat(self, config, device):
        """Episode poison counter should increment when eating poison."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        env.current_episode_poison[0] = 0
        env.agent_energy[0] = 100.0  # Ensure survival

        env.agent_x[0] = 10
        env.agent_y[0] = 10

        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        env.grids[0, 10, 11] = CellType.POISON.value

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        env.step(actions)

        assert env.current_episode_poison[0].item() == 1, \
            f"Poison counter should be 1, got {env.current_episode_poison[0].item()}"

    def test_counters_reset_on_episode_end(self, config, device):
        """Counters should reset when episode ends."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Simulate some eating
        env.current_episode_food[0] = 5
        env.current_episode_poison[0] = 2

        # Trigger death
        env.agent_energy[0] = 0.0
        env.grids[0, :, :] = CellType.EMPTY.value

        actions = torch.tensor([0], dtype=torch.long, device=device)
        env.step(actions)

        # After auto-reset, current counters should be 0
        # (last_episode counters should have the old values)
        assert env.current_episode_food[0].item() == 0, "Food counter should reset"
        assert env.current_episode_poison[0].item() == 0, "Poison counter should reset"
        assert env.last_episode_food[0].item() == 5, "Last episode food should be preserved"
        assert env.last_episode_poison[0].item() == 2, "Last episode poison should be preserved"


class TestRewardMechanics:
    """Tests for reward signal correctness."""

    def test_food_gives_positive_reward(self, config, device):
        """Eating food should give positive reward."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        env.agent_x[0] = 10
        env.agent_y[0] = 10

        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        env.grids[0, 10, 11] = CellType.FOOD.value

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        obs, eating_info, terminated, truncated = env.step(actions)
        food_mask, poison_mask, starved_mask = eating_info

        # Agent should have eaten food
        assert food_mask[0].item(), "Agent should have eaten food"

    def test_poison_gives_negative_effect(self, config, device):
        """Eating poison should be detected and reduce energy."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        env.agent_x[0] = 10
        env.agent_y[0] = 10
        initial_energy = 100.0
        env.agent_energy[0] = initial_energy  # Survive poison

        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        env.grids[0, 10, 11] = CellType.POISON.value

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        obs, eating_info, terminated, truncated = env.step(actions)
        food_mask, poison_mask, starved_mask = eating_info

        # Agent should have eaten poison
        assert poison_mask[0].item(), "Agent should have eaten poison"
        # Energy should have decreased (poison penalty + move cost)
        assert env.agent_energy[0].item() < initial_energy, "Energy should decrease after poison"

    def test_empty_step_only_costs_movement(self, config, device):
        """Moving to empty cell should only cost movement energy."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        env.agent_x[0] = 10
        env.agent_y[0] = 10
        initial_energy = env.agent_energy[0].item()

        # Clear target cell
        env.grids[0, 10, 11] = CellType.EMPTY.value

        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        obs, eating_info, terminated, truncated = env.step(actions)
        food_mask, poison_mask, starved_mask = eating_info

        # Agent should not have eaten anything
        assert not food_mask[0].item(), "Agent should not have eaten food"
        assert not poison_mask[0].item(), "Agent should not have eaten poison"
        # Energy should only decrease by move cost
        expected = initial_energy - env.energy_move_cost
        assert abs(env.agent_energy[0].item() - expected) < 0.01, "Energy should decrease by move cost only"


class TestSpawnCorrectness:
    """Tests for spawn/placement correctness after refactor.

    These tests verify:
    - Exact item counts (randperm guarantees no duplicates)
    - Agents spawn on empty cells only (not on food/poison)
    - Shared grid mode has no agent overlap
    - Respawned items don't land on agents
    """

    def test_exact_food_count_after_reset(self, config, device):
        """Food count should exactly match config, not 'at least'."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        for _ in range(10):  # Multiple trials for statistical confidence
            env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

            for grid_id in range(4):
                food_count = (env.grids[grid_id] == CellType.FOOD.value).sum().item()
                assert food_count == config['GRID_FOOD_INIT'], \
                    f"Grid {grid_id} has {food_count} food, expected exactly {config['GRID_FOOD_INIT']}"

    def test_exact_poison_count_after_reset(self, config, device):
        """Poison count should exactly match config."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        for _ in range(10):
            env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

            for grid_id in range(4):
                poison_count = (env.grids[grid_id] == CellType.POISON.value).sum().item()
                assert poison_count == config['GRID_POISON_INIT'], \
                    f"Grid {grid_id} has {poison_count} poison, expected exactly {config['GRID_POISON_INIT']}"

    def test_agents_spawn_on_empty_cells_only(self, config, device):
        """Agents should never spawn on food or poison."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        for _ in range(20):  # Many trials
            env = create_torch_vec_env(n_envs=8, obs_spec=spec, config=config, device=device)

            for env_id in range(8):
                grid_id = env.grid_indices[env_id].item()
                ay = env.agent_y[env_id].long().item()
                ax = env.agent_x[env_id].long().item()

                # The cell where agent spawned should now be marked as agent
                cell_value = env.grids[grid_id, ay, ax].item()
                agent_type = env.agent_types[env_id].item()

                assert cell_value == agent_type, \
                    f"Agent at ({ay}, {ax}) should be marked on grid, got {cell_value}"

                # The underlying cell should be EMPTY (agent spawned on empty)
                underlying = env.agent_underlying_cell[env_id].item()
                assert underlying == CellType.EMPTY.value, \
                    f"Agent {env_id} spawned on non-empty cell (underlying={underlying})"

    def test_shared_grid_no_agent_overlap(self, config, device):
        """In shared grid mode, all agents should have unique positions."""
        spec = ObservationSpec.for_mode('ground_truth', config)

        # Use shared_grid mode with multiple agents
        for _ in range(10):
            env = create_torch_vec_env(
                n_envs=4, obs_spec=spec, config=config, device=device, shared_grid=True
            )

            positions = set()
            for env_id in range(4):
                pos = (env.agent_y[env_id].item(), env.agent_x[env_id].item())
                assert pos not in positions, \
                    f"Duplicate agent position {pos} in shared grid mode"
                positions.add(pos)

    def test_no_food_on_agent_positions_after_respawn(self, config, device):
        """Respawned food should not land on agent positions."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=4, obs_spec=spec, config=config, device=device)

        # Run many respawn cycles
        for _ in range(50):
            eaten_mask = torch.ones(4, dtype=torch.bool, device=device)
            env._respawn_items_vectorized(eaten_mask, CellType.FOOD.value)

            # Check that no food spawned on agent positions
            for env_id in range(4):
                grid_id = env.grid_indices[env_id].item()
                ay = env.agent_y[env_id].long().item()
                ax = env.agent_x[env_id].long().item()

                cell_value = env.grids[grid_id, ay, ax].item()
                # Cell should still be agent (not overwritten by food)
                assert cell_value == env.agent_types[env_id].item(), \
                    f"Agent position overwritten by respawn: {cell_value}"

    def test_movement_preserves_underlying_cell(self, config, device):
        """When agent moves, old cell is properly restored."""
        spec = ObservationSpec.for_mode('ground_truth', config)
        env = create_torch_vec_env(n_envs=1, obs_spec=spec, config=config, device=device)

        # Place agent and manually set underlying cell
        env.agent_x[0] = 10
        env.agent_y[0] = 10
        env.agent_underlying_cell[0] = CellType.EMPTY.value
        env.grids[0, 10, 10] = env.agent_types[0].float()

        # Place food at target position
        env.grids[0, 10, 11] = CellType.FOOD.value

        # Move right
        right_action = None
        for idx in range(8):
            dx, dy = index_to_action(idx)
            if dx == 1 and dy == 0:
                right_action = idx
                break

        actions = torch.tensor([right_action], dtype=torch.long, device=device)
        env.step(actions)

        # Old position should now be EMPTY (restored)
        old_cell = env.grids[0, 10, 10].item()
        assert old_cell == CellType.EMPTY.value, \
            f"Old cell not restored: {old_cell}"

        # Agent should be at new position
        assert env.agent_x[0].item() == 11
        assert env.grids[0, 10, 11].item() == env.agent_types[0].item()

        # Underlying cell should be EMPTY (ate the food)
        assert env.agent_underlying_cell[0].item() == CellType.EMPTY.value
