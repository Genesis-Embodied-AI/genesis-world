from pydantic import StrictBool

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
    """

    show_FPS: StrictBool = True
    FPS_tracker_alpha: float = 0.95
