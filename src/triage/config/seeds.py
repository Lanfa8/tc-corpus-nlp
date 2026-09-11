"""Semente global de aleatoriedade do projeto."""

import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fixa as sementes de `random` e `numpy` para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
