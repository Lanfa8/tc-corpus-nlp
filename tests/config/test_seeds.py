import random

import numpy as np

from triage.config.seeds import SEED, set_seed


def test_seed_is_42():
    assert SEED == 42


def test_set_seed_makes_random_reproducible():
    set_seed()
    first = [random.random() for _ in range(5)]
    set_seed()
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seed_makes_numpy_reproducible():
    set_seed()
    first = np.random.rand(5)
    set_seed()
    second = np.random.rand(5)
    assert np.array_equal(first, second)
