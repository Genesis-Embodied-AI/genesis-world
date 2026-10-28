"""Particle state and reorder maps for smoothed particle hydrodynamics (SPH)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs

from . import SystemData, SystemDataDescription

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class SPHData(SystemData):
    """The reordered fields hold the particles sorted by spatial hash cell, which the substeps and the coupling read."""

    particles: qd.Field
    particles_ng: qd.Field
    particles_info: qd.Field
    particles_reordered: qd.Field
    particles_ng_reordered: qd.Field
    particles_info_reordered: qd.Field


@dataclass(frozen=True, kw_only=True)
class SPHDescription(SystemDataDescription[SPHData]):
    _B: int
    n_particles: int

    def allocate(self, owner: "Solver") -> SPHData:
        # dynamic particle state
        particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            acc=gs.qd_vec3,  # acceleration
            rho=gs.qd_float,  # density
            p=gs.qd_float,  # pressure
            dfsph_factor=gs.qd_float,  # DFSPH use: Factor for Divergence and density solver
            drho=gs.qd_float,  # density deritivate
        )
        # dynamic particle state without gradient
        particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
        )
        # static particle info
        particle_info = qd.types.struct(
            rho=gs.qd_float,  # rest density
            mass=gs.qd_float,  # mass
            stiffness=gs.qd_float,
            exponent=gs.qd_float,
            mu=gs.qd_float,  # viscosity
            gamma=gs.qd_float,  # surface tension
        )
        particles = particle_state.field(shape=(self.n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA)
        particles_ng = particle_state_ng.field(
            shape=(self.n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_info = particle_info.field(shape=(self.n_particles,), needs_grad=False, layout=qd.Layout.SOA)
        particles_reordered = particle_state.field(
            shape=(self.n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_ng_reordered = particle_state_ng.field(
            shape=(self.n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_info_reordered = particle_info.field(
            shape=(self.n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        return SPHData(
            owner=owner,
            particles=particles,
            particles_ng=particles_ng,
            particles_info=particles_info,
            particles_reordered=particles_reordered,
            particles_ng_reordered=particles_ng_reordered,
            particles_info_reordered=particles_info_reordered,
        )
