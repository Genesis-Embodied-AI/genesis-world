from typing import TYPE_CHECKING

import genesis as gs
from genesis.repr_base import RBC
from genesis.utils.misc import BuildableMixin

from .description import RigidSiteDescription

if TYPE_CHECKING:
    from genesis.engine.solvers.kinematic_solver import KinematicSolver

    from .rigid_entity import KinematicEntity
    from .rigid_link import KinematicLink


class RigidSite(BuildableMixin, RBC):
    """Attachment or landmark frame rigidly mounted on a `KinematicLink` / `RigidLink`."""

    def __init__(self, link: "KinematicLink", idx: int, desc: RigidSiteDescription):
        self.desc: RigidSiteDescription = desc
        self._link: "KinematicLink" = link
        self._entity: "KinematicEntity" = link.entity
        self._solver: "KinematicSolver" = link.entity.solver

        self._uid = gs.UID()
        self._idx: int = idx

    @gs.assert_unbuilt
    def _build(self) -> None:
        self._is_built = True

    def _destroy(self) -> None:
        self._is_built = False

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """Get the world position of the site."""
        return self._solver.get_sites_pos(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """Get the world orientation quaternion (w, x, y, z) of the site."""
        return self._solver.get_sites_quat(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_vel(self, envs_idx=None, *, relative=True):
        """Get the world linear velocity of the site."""
        return self._solver.get_sites_vel(self._idx, envs_idx, relative=relative)[..., 0, :]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """The unique ID of the site."""
        return self._uid

    @property
    def name(self) -> str:
        """The name of the site."""
        return self.desc.name

    @property
    def idx(self) -> int:
        """The global index of the site in the solver."""
        return self._idx

    @property
    def idx_local(self) -> int:
        """The local index of the site within its entity."""
        return self._idx - self._entity._site_start

    @property
    def link(self) -> "KinematicLink":
        """The link that owns the site."""
        return self._link

    @property
    def entity(self) -> "KinematicEntity":
        """The entity that owns the site."""
        return self._entity

    @property
    def solver(self) -> "KinematicSolver":
        """The solver that simulates the site's entity."""
        return self._solver

    @property
    def type(self) -> gs.GEOM_TYPE:
        """Primitive shape type authored on the site."""
        return self.desc.type

    @property
    def init_pos(self):
        """Position of the site in its link frame."""
        return self.desc.pos

    @property
    def init_quat(self):
        """Quaternion (w, x, y, z) of the site in its link frame."""
        return self.desc.quat

    @property
    def size(self):
        """Size parameters authored on the site."""
        return self.desc.size

    @property
    def rgba(self):
        """RGBA color authored on the site."""
        return self.desc.rgba

    @property
    def group(self) -> int:
        """Visualization group authored on the site."""
        return self.desc.group

    @property
    def is_fixed(self) -> bool:
        """Whether the site's link is fixed in the world."""
        return self._link.is_fixed

    def _repr_brief(self):
        return f"{self.__repr_name__()}: {self._uid}, name: '{self.name}', idx: {self._idx}"
