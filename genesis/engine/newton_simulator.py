from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.options.engines import NewtonEngineOptions

from .simulator import BaseSimulator

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.systems import SimEngine
    from genesis.options.scene import SceneOptions


class NewtonSimulator(BaseSimulator):
    """Run the graph-native Rigid, QCloth, and Consistent IPC pipeline."""

    def __init__(self, scene: "Scene", options: "SceneOptions"):
        super().__init__(scene, options)
        if not isinstance(options.engine, NewtonEngineOptions):
            raise TypeError("NewtonSimulator requires NewtonEngineOptions")
        self.engine_options = options.engine
        self._engine: SimEngine | None = None

    def _build_runtime(self) -> None:
        if gs.backend == gs.cpu:
            gs.raise_exception("NewtonSimulator requires a GPU backend.")
        if gs.qd_float != qd.f64:
            gs.raise_exception("NewtonSimulator requires 'precision=\"64\"'.")
        if not gs.use_ndarray:
            gs.raise_exception("NewtonSimulator requires the Quadrants ndarray backend.")
        if self.requires_grad:
            gs.raise_exception("NewtonSimulator does not support differentiable simulation.")
        if self.substeps != 1:
            gs.raise_exception("NewtonSimulator first version requires 'SimOptions.substeps=1'.")
        if self.n_envs != 0:
            gs.raise_exception("NewtonSimulator first version supports only a single unbatched environment.")
        if not self.fem_solver.is_active:
            gs.raise_exception("NewtonSimulator requires at least one FEM.QCloth entity.")

        unsupported_solvers = [
            type(solver).__name__
            for solver in self.active_solvers
            if solver not in (self.rigid_solver, self.fem_solver)
        ]
        if unsupported_solvers:
            gs.raise_exception(
                "NewtonSimulator first version supports only RigidSolver and FEMSolver, got active "
                f"{', '.join(unsupported_solvers)}."
            )

        unsupported_materials = [
            type(entity.material).__name__
            for entity in self.fem_solver.entities
            if not isinstance(entity.material, gs.materials.FEM.QCloth)
        ]
        if unsupported_materials:
            gs.raise_exception(
                f"NewtonSimulator accepts only FEM.QCloth FEM entities, got {', '.join(unsupported_materials)}."
            )

        from genesis.engine.systems import ContactTabular, build_scene_engine

        if not self.fem_solver._constraints_initialized:
            self.fem_solver.init_constraints()

        contact_tabular = ContactTabular()
        contact_tabular.default_model(
            friction_rate=self.engine_options.contact_friction_mu,
            resistance=self.engine_options.contact_resistance,
        )
        halfplanes = None
        if not self.rigid_solver.is_active:
            halfplanes = (
                np.array([[0.0, 0.0, self.fem_solver.floor_height]], dtype=np.float64),
                np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            )
        self._engine = build_scene_engine(
            self.scene,
            contact_config={
                "contact/d_hat": self.engine_options.contact_d_hat,
                "contact/init_collision_pair_capacity": 20_000,
                "friction/eps_v": self.engine_options.contact_eps_velocity,
            },
            contact_tabular=contact_tabular,
            halfplanes=halfplanes,
        )

    def _reset_runtime(self, envs_idx=None) -> None:
        if envs_idx is not None:
            gs.raise_exception("NewtonSimulator first version does not support partial environment reset.")
        self.engine.sync_from_solvers()
        self._state_dirty.clear()

    def _step_physics(self, in_backward: bool) -> None:
        if in_backward:
            gs.raise_exception("NewtonSimulator does not support backward simulation.")
        self.process_input(in_backward=False)
        if self._state_dirty:
            self.engine.sync_from_solvers()
            self._state_dirty.clear()
        self.engine.step()
        self._cur_substep_global += 1

    def _step_grad(self):
        gs.raise_exception("NewtonSimulator does not support backward simulation.")

    @property
    def engine(self) -> "SimEngine":
        engine = self._engine
        if engine is None:
            raise RuntimeError("NewtonSimulator has not been built")
        return engine
