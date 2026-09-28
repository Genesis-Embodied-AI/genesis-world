"""
USD (Universal Scene Description) utilities for parsing and processing USD files.

This module provides functionality for parsing USD stages and converting them into Genesis
entities. It supports parsing rigid bodies, articulations, materials, and geometry from
USD files.

The module requires the `usd-core` package to be installed. If USD support is not available,
importing this module will raise an exception with installation instructions.

Key Components
--------------
- `UsdContext`: Context manager for USD stage operations, providing utilities for
  transform computation, material binding, and entity discovery.
- `parse_usd_stage`: Parse a USD stage and extract all rigid entities as separate USD morphs.
- `parse_usd_rigid_entity`: Parse a single rigid entity (articulation or rigid body) from USD.

Dependencies
------------
- `pxr` (USD Python bindings): Required for USD support. Install with `pip install genesis-world[usd]`.
- `omni.kit_app` (Omniverse Kit): Optional, required for USD material baking. Only available
  when CUDA is available and Omniverse Kit is installed.

Examples
--------
>>> import genesis as gs
>>> from genesis.utils.usd import parse_usd_stage
>>>
>>> # Create a USD morph and parse the stage, adding its entities while the stage stays open
>>> usd_morph = gs.morphs.USD(file="scene.usd")
>>> scene = gs.Scene()
>>> with parse_usd_stage(usd_morph) as entity_morphs:
...     for entity_morph in entity_morphs:
...         scene.add_entity(entity_morph)
>>> scene.build()
"""

import genesis as gs

# Check if USD support is available before importing modules that depend on it
try:
    from pxr import Usd

except ImportError as e:
    gs.raise_exception_from("usd-core module not found. Please install it with `pip install genesis-world[usd]`.", e)

# USD support is available - import the parser modules
from .usd_stage import parse_usd_stage
from .usd_rigid_entity import parse_usd_rigid_entity
from .usd_context import UsdContext, HAS_OMNIVERSE_KIT_SUPPORT
