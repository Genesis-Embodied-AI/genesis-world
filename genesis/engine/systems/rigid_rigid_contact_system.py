"""System of the contact between rigid geoms, which opens the second graph of every rigid substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Require, System, register_system
from genesis.engine.solvers.rigid.collider.collider import (
    collider_clear,
    collider_detection,
    collider_reset,
    func_detection,
)

from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class RigidRigidContactSystem(System):
    """System detecting the contacts between the rigid geoms of the scene, which it adds when it enables collisions.

    It covers every pair of rigid geoms, terrains and planes included, and hands the contacts to the constraint solve.
    The contacts of other solvers with rigid geoms belong to their couplings. The collision data belongs to the rigid
    data for now (see ColliderData), which the substep graphs read through RigidSolverData. Its Actions bind this
    System, which the Initializer has record the static configuration of the detection once the rigid data exists, and
    the graphs read it when they compile.
    """

    rigid_solver_system = Require(RigidSolverSystem)

    @System.action(kind="host")
    def initialize(self):
        return self.record_configuration

    @System.action(kind="host")
    def prepare(self):
        return self.clear_contact_data_cache

    @System.action(kind="stage")
    def detect(self):
        return func_substep_detection, self

    @System.action(kind="host")
    def detect_kernels(self):
        return self.detect_launching_kernels

    @System.action(kind="host")
    def reset(self):
        return self.reset_contacts

    @System.action(kind="host")
    def clear(self):
        return self.clear_contacts

    def build(self):
        self.rigid_solver_system.on_detect(prepare=self.prepare, detect=self.detect, detect_kernels=self.detect_kernels)
        self.rigid_solver_system.on_reset(reset=self.reset, clear=self.clear)

    def record_configuration(self) -> None:
        """Record the static configuration of the collision detection, which the detection graph compiles against."""
        collider_data = self.rigid_solver_system.collider_data
        self.collider_data = collider_data
        self.collider_config = collider_data.collider_config
        self.gjk_config = collider_data.gjk.gjk_config
        self.has_possible_pairs = collider_data._n_possible_pairs > 0
        self.split_narrowphase = collider_data._use_split_narrowphase
        self.coop_dedup = collider_data._use_coop_dedup

    def clear_contact_data_cache(self) -> None:
        """Drop the contacts read back during the previous substep, which the detection is about to replace."""
        self.collider_data._contact_data_cache.clear()

    def detect_launching_kernels(self) -> None:
        """Detect the contacts between rigid geoms of a substep through the detection kernel alone."""
        collider_detection(self.collider_data, self.rigid_solver_system)

    def reset_contacts(self, envs_idx) -> None:
        """Restart the warm start of the detection in the given environments."""
        collider_reset(self.collider_data, self.rigid_solver_system, envs_idx)

    def clear_contacts(self, envs_idx) -> None:
        """Clear the contacts and the warm start of the detection in the given environments."""
        collider_clear(self.collider_data, self.rigid_solver_system, envs_idx)


@register_system(SimulatorEngine)
def create_rigid_rigid_contact_system(scene: "Scene") -> RigidRigidContactSystem | None:
    """Create the contact between rigid geoms, unless the scene disables collisions."""
    rigid_solver = scene.sim.rigid_solver
    return RigidRigidContactSystem(scene) if rigid_solver.is_active and rigid_solver._enable_collision else None


@qd.func(requires_top_level=True)
def func_substep_detection(contact_system: qd.template()):
    """Detect the contacts between rigid geoms of a substep, configured by the static configuration the System
    records (see func_detection)."""
    rigid_data = contact_system.rigid_solver_system.data
    func_detection(
        geoms_init_AABB=rigid_data.geoms_init_AABB,
        dyn_state=rigid_data.dyn_state,
        collider_state=rigid_data.collider_state,
        mpr_state=rigid_data.mpr_state,
        gjk_state=rigid_data.gjk_state,
        diff_contact_input=rigid_data.gjk_state.diff_contact_input,
        contact0_mpr_state=rigid_data.contact0_mpr_state,
        contact0_gjk_state=rigid_data.contact0_gjk_state,
        multicontact_mpr_state=rigid_data.multicontact_mpr_state,
        multicontact_gjk_state=rigid_data.multicontact_gjk_state,
        constraint_state=rigid_data.constraint_state,
        dyn_info=rigid_data.dyn_info,
        rigid_info=rigid_data.rigid_info,
        collider_info=rigid_data.collider_info,
        rigid_config=contact_system.rigid_solver_system.rigid_config,
        collider_static_config=contact_system.collider_config,
        gjk_static_config=contact_system.gjk_config,
        has_possible_pairs=contact_system.has_possible_pairs,
        split_narrowphase=contact_system.split_narrowphase,
        coop_dedup=contact_system.coop_dedup,
        errno=rigid_data.errno,
    )
