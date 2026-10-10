"""Reference trajectories generated with Meta's mochi physics engine (the `superdex` package), used to check the
Genesis port against the original implementation.

Mochi scenes are authored Y-up; `mochi_to_genesis` maps them onto the Genesis Z-up frame by a +90 degree rotation about
x: (x, y, z) -> (x, -z, y).
"""

import importlib.util
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def mochi_to_genesis(vec):
    vec = np.asarray(vec)
    return np.stack([vec[..., 0], -vec[..., 2], vec[..., 1]], axis=-1)


def load_reference(name):
    """Load the named reference case, regenerating it with mochi when the package is importable and the file is
    missing, else skipping the test."""
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        if importlib.util.find_spec("superdex") is None:
            import pytest

            pytest.skip(f"Reference '{name}' is missing and mochi (superdex) is not importable to regenerate it.")
        from .gen_reference import generate

        generate(name, DATA_DIR)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
