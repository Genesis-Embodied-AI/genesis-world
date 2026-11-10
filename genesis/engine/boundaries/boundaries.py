import quadrants as qd

from genesis.utils.data import Data, DataKind, field


class CubeBoundary(Data, kind=DataKind.CONSTANT, shape=()):
    """An axis-aligned box confining the particles, which bounce off its walls with the given restitution."""

    lower: qd.Tensor = field((float, 3))
    upper: qd.Tensor = field((float, 3))
    restitution: qd.Tensor = field(float)


@qd.func
def func_impose_cube_boundary(pos: qd.types.vector(3), vel: qd.types.vector(3), boundary: CubeBoundary):
    """Clamp a position inside a cube boundary, the velocity bouncing off any wall it crosses or presses against."""
    lower = boundary.lower[None]
    upper = boundary.upper[None]
    is_bouncing = ((pos >= upper) & (vel >= 0)) | ((pos <= lower) & (vel <= 0))
    vel = qd.select(is_bouncing, -boundary.restitution[None] * vel, vel)
    pos = qd.max(qd.min(pos, upper), lower)

    return pos, vel
