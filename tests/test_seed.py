"""Tests for seed management utilities."""
import random

import numpy as np
import pytest
import torch

from goodharts.utils.seed import set_seed, get_random_seed


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_returns_seed(self):
        """set_seed should return the seed used."""
        seed = set_seed(42)
        assert seed == 42

    def test_set_seed_with_none_returns_random_seed(self):
        """set_seed(None) should generate and return a random seed."""
        seed = set_seed(None)
        assert isinstance(seed, int)
        assert seed >= 0

    def test_set_seed_makes_random_deterministic(self):
        """After set_seed, random.random() should be reproducible."""
        set_seed(12345)
        values1 = [random.random() for _ in range(10)]

        set_seed(12345)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    def test_set_seed_makes_numpy_deterministic(self):
        """After set_seed, numpy random should be reproducible."""
        set_seed(12345)
        arr1 = np.random.rand(10)

        set_seed(12345)
        arr2 = np.random.rand(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_set_seed_makes_torch_deterministic(self):
        """After set_seed, torch random should be reproducible."""
        set_seed(12345)
        t1 = torch.rand(10)

        set_seed(12345)
        t2 = torch.rand(10)

        assert torch.equal(t1, t2)

    def test_different_seeds_give_different_results(self):
        """Different seeds should produce different random sequences."""
        set_seed(111)
        val1 = random.random()

        set_seed(222)
        val2 = random.random()

        assert val1 != val2


class TestGetRandomSeed:
    """Tests for get_random_seed function."""

    def test_returns_positive_integer(self):
        """get_random_seed should return a positive integer."""
        seed = get_random_seed()
        assert isinstance(seed, int)
        assert seed >= 0

    def test_returns_different_values(self):
        """get_random_seed should return different values on each call."""
        seeds = [get_random_seed() for _ in range(5)]
        # At least some should be different (extremely unlikely all same)
        assert len(set(seeds)) > 1
