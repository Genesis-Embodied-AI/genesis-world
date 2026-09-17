from pydantic import StrictBool

from genesis.typing import PositiveInt

from .options import Options


class ProfilingOptions(Options):
    """
    Profiling options

    Parameters
    ----------
    show_FPS : bool
        Whether to show the frame rate each step. Default true
    FPS_tracker_alpha: float
        Exponential decay momentum for FPS moving average
    timings_window: int
        Number of latest steps the step timings (see Scene.timings) are averaged over. A longer window smooths a rate
        read from them at the cost of as many steps of delay. Default 1, the last step alone.
    """

    show_FPS: StrictBool = True
    FPS_tracker_alpha: float = 0.95
    timings_window: PositiveInt = 1
