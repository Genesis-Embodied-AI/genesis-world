from typing import TYPE_CHECKING

from .base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Kinematic[EntityT: "Entity"](Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection, or constraint
    solving.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        Expose this entity's visual mesh to every consumer that casts against visual geometry: raycaster sensors, and
        viewer plugins built on RaycasterViewerPlugin with use_visual_geom set. Each such consumer then re-scans the
        entity's visual triangles as it moves, so reserve it for the entities meant to be sensed or picked.
        Default is False.
    """
