#!/usr/bin/env python3
"""
Test that respawn is truly random across the grid.

Directly tests the respawn algorithm by forcing many respawns
and tracking where items land. No waiting for agents to eat!
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    from goodharts.environments.torch_env import create_torch_vec_env
    from goodharts.modes import ObservationSpec
    from goodharts.configs.default_config import get_config

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name()}")

    config = get_config()
    obs_spec = ObservationSpec.for_mode('ground_truth', config)

    # Use many envs for parallel sampling
    n_envs = 256
    vec_env = create_torch_vec_env(
        n_envs=n_envs,
        device=device,
        obs_spec=obs_spec,
        shared_grid=False,
    )

    height, width = vec_env.height, vec_env.width
    print(f"Grid size: {height}x{width}")
    print(f"Envs: {n_envs}")

    # Track where food respawns (aggregate across all envs)
    respawn_heatmap = torch.zeros(height, width, device=device)

    # Number of forced respawn rounds
    n_rounds = 1000
    items_per_round = 50  # Force this many respawns per env per round

    vec_env.reset()

    print(f"Running {n_rounds} rounds x {n_envs} envs x {items_per_round} respawns...")
    print(f"Total respawns: {n_rounds * n_envs * items_per_round:,}")

    for round_idx in range(n_rounds):
        # For each round, force respawns by directly calling the respawn function
        # We'll simulate "eaten" items by creating a mask

        for _ in range(items_per_round):
            # Track food positions BEFORE respawn
            food_before = (vec_env.grids == vec_env.CellType.FOOD.value)  # (n_envs, H, W)

            # Create a fake "eaten" mask - pretend one random food was eaten per env
            # This triggers the respawn logic
            eaten_mask = torch.ones(n_envs, dtype=torch.bool, device=device)

            # Call respawn directly
            vec_env._respawn_items_vectorized(eaten_mask, vec_env.CellType.FOOD.value)

            # Track food positions AFTER respawn
            food_after = (vec_env.grids == vec_env.CellType.FOOD.value)  # (n_envs, H, W)

            # New food locations (respawned)
            new_food = food_after & ~food_before  # (n_envs, H, W)

            # Aggregate across all envs
            respawn_heatmap += new_food.sum(dim=0).float()

        if (round_idx + 1) % 100 == 0:
            print(f"  Round {round_idx + 1}/{n_rounds}")

    # Move to CPU for analysis
    heatmap = respawn_heatmap.cpu().numpy()

    # Statistics
    total_respawns = heatmap.sum()
    mean_per_cell = heatmap.mean()
    std_per_cell = heatmap.std()
    min_per_cell = heatmap.min()
    max_per_cell = heatmap.max()

    print(f"\nResults:")
    print(f"  Total respawns tracked: {total_respawns:,.0f}")
    print(f"  Mean per cell: {mean_per_cell:.1f}")
    print(f"  Std per cell: {std_per_cell:.1f}")
    print(f"  Min: {min_per_cell:.0f}, Max: {max_per_cell:.0f}")

    if mean_per_cell > 0:
        cv = std_per_cell / mean_per_cell
        expected_cv = 1 / np.sqrt(mean_per_cell)
        print(f"  Coefficient of variation: {cv:.2%}")
        print(f"  Expected CV for uniform: ~{expected_cv:.2%}")

        # Check for suspicious patterns
        if cv > expected_cv * 2:
            print("  WARNING: CV much higher than expected - possible bias!")
        elif cv < expected_cv * 0.5:
            print("  WARNING: CV much lower than expected - suspicious uniformity!")
        else:
            print("  CV is within expected range for uniform distribution")

    # Visual check
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Respawn count')
    plt.title(f'Respawn Heatmap\n({total_respawns:,.0f} total respawns)')

    plt.subplot(2, 2, 2)
    plt.hist(heatmap.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Respawns per cell')
    plt.ylabel('Number of cells')
    plt.title('Distribution of respawn counts')
    plt.axvline(mean_per_cell, color='red', linestyle='--', label=f'Mean: {mean_per_cell:.1f}')
    plt.legend()

    plt.subplot(2, 2, 3)
    # Show cells with zero respawns (potential dead zones)
    zero_mask = (heatmap == 0).astype(float)
    plt.imshow(zero_mask, cmap='Reds', interpolation='nearest')
    plt.colorbar(label='Zero respawns')
    plt.title(f'Dead zones (zero respawns)\n{zero_mask.sum():.0f} cells ({zero_mask.mean()*100:.1f}%)')

    plt.subplot(2, 2, 4)
    # Row and column sums to check for patterns
    row_sums = heatmap.sum(axis=1)
    col_sums = heatmap.sum(axis=0)
    plt.plot(row_sums, label='Row sums', alpha=0.7)
    plt.plot(col_sums, label='Col sums', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Total respawns')
    plt.title('Row/Column totals (should be flat)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('respawn_heatmap.png', dpi=150)
    print(f"\nHeatmap saved to respawn_heatmap.png")

    # Chi-squared test for uniformity (on cells that could have respawns)
    from scipy import stats
    observed = heatmap.flatten()
    expected_val = mean_per_cell
    expected_arr = np.full_like(observed, expected_val)

    # Only test cells with expected > 5 (chi-squared assumption)
    if expected_val > 5:
        chi2, p_value = stats.chisquare(observed, expected_arr)
        print(f"\nChi-squared test for uniformity:")
        print(f"  Chi2 = {chi2:.1f}, p-value = {p_value:.6f}")
        if p_value > 0.05:
            print("  PASS: Distribution appears uniform (p > 0.05)")
        else:
            print("  FAIL: Distribution is NOT uniform (p < 0.05)")
    else:
        print(f"\nNot enough data for chi-squared test (need mean > 5, got {mean_per_cell:.1f})")


if __name__ == "__main__":
    main()
