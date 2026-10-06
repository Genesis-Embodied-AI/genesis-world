from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.options.solvers import NewtonCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator
    from genesis.engine.systems import SimEngine


class NewtonCoupler(RBC):
    """Drive the graph-native QCloth, Rigid, and IPC runtime from ``Scene.step``."""

    defer_build_warmup = True

    def __init__(self, simulator: "Simulator", options: NewtonCouplerOptions) -> None:
        self.sim = simulator
        self.options = options
        self.rigid_solver = simulator.rigid_solver
        self.fem_solver = simulator.fem_solver
        self._engine: SimEngine | None = None

    def build(self) -> None:
        if gs.backend == gs.cpu:
            gs.raise_exception("NewtonCoupler requires a GPU backend.")
        if gs.qd_float != qd.f64:
            gs.raise_exception("NewtonCoupler requires 'precision=\"64\"'.")
        if not gs.use_ndarray:
            gs.raise_exception("NewtonCoupler requires the Quadrants ndarray backend.")
        if self.sim.requires_grad:
            gs.raise_exception("NewtonCoupler does not support differentiable simulation.")
        if self.sim.substeps != 1:
            gs.raise_exception("NewtonCoupler first version requires 'SimOptions.substeps=1'.")
        if self.sim.n_envs != 0:
            gs.raise_exception("NewtonCoupler first version supports only a single unbatched environment.")
        if not self.fem_solver.is_active:
            gs.raise_exception("NewtonCoupler requires at least one FEM.QCloth entity.")

        unsupported_solvers = [
            type(solver).__name__
            for solver in self.sim.active_solvers
            if solver not in (self.rigid_solver, self.fem_solver)
        ]
        if unsupported_solvers:
            gs.raise_exception(
                "NewtonCoupler first version supports only RigidSolver and FEMSolver, got active "
                f"{', '.join(unsupported_solvers)}."
            )

        unsupported_materials = [
            type(entity.material).__name__
            for entity in self.fem_solver.entities
            if not isinstance(entity.material, gs.materials.FEM.QCloth)
        ]
        if unsupported_materials:
            gs.raise_exception(
                f"NewtonCoupler accepts only FEM.QCloth FEM entities, got {', '.join(unsupported_materials)}."
            )

    def _ensure_engine(self) -> "SimEngine":
        if self._engine is None:
            # Import lazily so users can set qpos, controller gains, and QCloth
            # vertex constraints after Scene.build and before the first step.
            from genesis.engine.systems import ContactTabular, build_scene_engine

            contact_tabular = ContactTabular()
            contact_tabular.default_model(
                friction_rate=self.options.contact_friction_mu,
                resistance=self.options.contact_resistance,
            )
            halfplanes = None
            if not self.rigid_solver.is_active:
                halfplanes = (
                    np.array([[0.0, 0.0, self.fem_solver.floor_height]], dtype=np.float64),
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
                )
            self._engine = build_scene_engine(
                self.sim.scene,
                contact_config={
                    "contact/d_hat": self.options.contact_d_hat,
                    "contact/init_collision_pair_capacity": 20_000,
                    "friction/eps_v": self.options.contact_eps_velocity,
                },
                contact_tabular=contact_tabular,
                halfplanes=halfplanes,
            )
        return self._engine

    def step(self) -> None:
        self._ensure_engine().step()

    def reset(self, envs_idx=None) -> None:
        if envs_idx is not None and self._engine is not None:
            gs.raise_exception("NewtonCoupler first version does not support partial environment reset.")
        # Solver state is authoritative across Scene state/reset APIs. Rebuild
        # the adapter lazily so FEM staging and rigid proxies read that state.
        self._engine = None

    @property
    def engine(self) -> "SimEngine":
        if self._engine is None:
            raise RuntimeError("NewtonCoupler engine is created lazily by the first Scene.step()")
        return self._engine

    @property
    def is_active(self) -> bool:
        return True
