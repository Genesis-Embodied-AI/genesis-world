"""System of the constraint solve of the rigid solver, which spans all three graphs of every rigid substep."""

from typing import TYPE_CHECKING

import quadrants as qd

from genesis.engine.core import Require, System, register_system
from genesis.engine.solvers.rigid.constraint.solver import (
    func_add_equality_constraints,
    func_add_inequality_constraints,
    func_resolve_post,
    func_solve_body,
)

from .rigid_solver_system import RigidSolverSystem
from .simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


class ConstraintSolverSystem(System):
    """System running the constraint solve of the rigid solver, which the scene adds unless it disables constraints.

    The constraint solve belongs to the rigid data (see RigidSolverSystem), which the substep graphs read through
    RigidSolverData. It assembles the equality constraints after the forward dynamics and the inequality constraints
    after the collision detection, solves them on the host, then turns the solved forces into accelerations and contact
    forces. Its Actions bind this System, which the Initializer has record the static configuration of the solve once
    the rigid data exists, and the graphs read it when they compile.
    """

    rigid_solver_system = Require(RigidSolverSystem)

    @System.action(kind="host")
    def prepare(self):
        return self.clear_equality_info_cache

    @System.action(kind="stage")
    def assemble_equalities(self):
        return func_substep_add_equality_constraints, self

    @System.action(kind="stage")
    def assemble_inequalities(self):
        return func_substep_add_inequality_constraints, self

    @System.action(kind="host")
    def solve(self):
        return self.solve_constraints

    @System.action(kind="stage")
    def update_forces(self):
        return func_substep_resolve_post, self

    def build(self):
        self.rigid_solver_system.on_constrain(
            prepare=self.prepare,
            assemble_equalities=self.assemble_equalities,
            assemble_inequalities=self.assemble_inequalities,
            solve=self.solve,
            update_forces=self.update_forces,
        )

    @System.action(kind="host")
    def initialize(self):
        return self.record_configuration

    def record_configuration(self) -> None:
        """Record the static configuration of the constraint solve, which the substep graphs compile against."""
        rigid_solver_system = self.rigid_solver_system
        self.constraint_solver = rigid_solver_system.constraint_solver
        self.collider_config = rigid_solver_system.collider_data.collider_config
        self.n_iterations = self.constraint_solver._n_iterations
        self.noslip = rigid_solver_system._options.noslip_iterations > 0

    def clear_equality_info_cache(self) -> None:
        """Drop the equality constraints read back during the previous substep, which the assembly is about to
        replace."""
        self.constraint_solver._eq_const_info_cache.clear()

    def solve_constraints(self) -> None:
        """Solve the constraints of a substep, through the solver arm func_solve_body selects at run time."""
        rigid_data = self.rigid_solver_system.data
        func_solve_body(
            rigid_data.dyn_state,
            rigid_data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            self.rigid_solver_system.rigid_config,
            self.n_iterations,
        )


@register_system(SimulatorEngine)
def create_constraint_solver_system(scene: "Scene") -> ConstraintSolverSystem | None:
    """Create the constraint solve of the rigid substep, unless the scene disables constraints."""
    rigid_solver = scene.sim.rigid_solver
    return ConstraintSolverSystem(scene) if rigid_solver.is_active and not rigid_solver._disable_constraint else None


@qd.func(requires_top_level=True)
def func_substep_add_equality_constraints(constraint_solver_system: qd.template()):
    """Assemble the equality constraints of a substep."""
    rigid_data = constraint_solver_system.rigid_solver_system.data
    func_add_equality_constraints(
        rigid_data.dyn_state,
        rigid_data.collider_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        constraint_solver_system.rigid_solver_system.rigid_config,
    )


@qd.func(requires_top_level=True)
def func_substep_add_inequality_constraints(constraint_solver_system: qd.template()):
    """Assemble the inequality constraints of a substep, from its detected contacts among others."""
    rigid_data = constraint_solver_system.rigid_solver_system.data
    func_add_inequality_constraints(
        rigid_data.dyn_state,
        rigid_data.collider_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        constraint_solver_system.rigid_solver_system.rigid_config,
        constraint_solver_system.collider_config,
    )


@qd.func(requires_top_level=True)
def func_substep_resolve_post(constraint_solver_system: qd.template()):
    """Update the accelerations and the contact forces of a substep from its solved constraint forces (see
    func_resolve_post for noslip)."""
    rigid_data = constraint_solver_system.rigid_solver_system.data
    func_resolve_post(
        rigid_data.dyn_state,
        rigid_data.collider_state,
        rigid_data.constraint_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        constraint_solver_system.rigid_solver_system.rigid_config,
        constraint_solver_system.noslip,
        rigid_data.errno,
    )
