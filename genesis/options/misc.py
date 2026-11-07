from typing import Literal

from pydantic import Field, StrictBool, StrictInt

from genesis.typing import NonNegativeInt, PositiveInt, Vec4FType

from .options import Options


class FoamOptions(Options):
    """
    Options for foam generation.
    """

    radius_scale: float = 0.2  # foam particle radius w.r.t fluid particle radius
    color: Vec4FType = (0.7, 0.7, 0.7, 0.7)
    spray_decay: float = 2.0  # The dissipation rate of spray foam
    foam_decay: float = 1.0  # The dissipation rate of foam foam
    bubble_decay: float = 5.0  # The dissipation rate of bubble foam
    k_foam: float = 1000.0  # amount of foam generated per frame


class CoacdOptions(Options):
    """
    Options for configuring coacd convex decomposition.
    Reference: https://github.com/SarahWeiii/CoACD
    """

    # Main parameter to tune to improve the accuracy.
    # As a rule of thumbs, dividing the threshold by two would double the number of convex hulls.
    threshold: float = 0.1

    max_convex_hull: StrictInt = Field(default=-1, ge=-1)
    preprocess_mode: Literal["on", "off", "auto"] = "auto"
    preprocess_resolution: PositiveInt = 30
    resolution: PositiveInt = 1000
    mcts_nodes: PositiveInt = 20
    mcts_iterations: PositiveInt = 100
    mcts_max_depth: PositiveInt = 3
    pca: StrictBool = False
    merge: StrictBool = True
    decimate: StrictBool = False
    max_ch_vertex: PositiveInt = 256
    extrude: StrictBool = False
    extrude_margin: float = 0.1
    apx_mode: Literal["ch", "box"] = "ch"
    seed: NonNegativeInt = 0
