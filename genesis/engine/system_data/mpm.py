"""Particle history and coupling grid for the material point method (MPM)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs

from . import SystemData, SystemDataDescription

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class MPMData(SystemData):
    """Particle mass and grid mass are multiplied by particle_volume_scale."""

    particle_volume_scale: float
    particles: qd.Field
    particles_ng: qd.Field
    particles_info: qd.Field
    grid: qd.Field


@dataclass(frozen=True, kw_only=True)
class MPMDescription(SystemDataDescription[MPMData]):
    _B: int
    particle_volume_scale: float
    substeps_local: int
    n_particles: int
    grid_res: np.ndarray

    def allocate(self, owner: "Solver") -> MPMData:
        # dynamic particle state
        particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            C=gs.qd_mat3,  # affine velocity field
            F=gs.qd_mat3,  # deformation gradient
            F_tmp=gs.qd_mat3,  # temp deformation gradient
            U=gs.qd_mat3,  # SVD
            V=gs.qd_mat3,  # SVD
            S=gs.qd_mat3,  # SVD
            actu=gs.qd_float,  # actuation
            Jp=gs.qd_float,  # volume ratio
        )
        # dynamic particle state without gradient
        particle_state_ng = qd.types.struct(active=gs.qd_bool)
        # static particle info
        particle_info = qd.types.struct(
            material_idx=gs.qd_int,
            mass=gs.qd_float,
            default_Jp=gs.qd_float,
            free=gs.qd_bool,
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )
        grid_cell_state = qd.types.struct(
            mass=gs.qd_float,  # mass
            vel_in=gs.qd_vec3,  # input momentum/velocity
            vel_out=gs.qd_vec3,  # output momentum/velocity
        )
        particles = particle_state.field(
            shape=(self.substeps_local + 1, self.n_particles, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        particles_ng = particle_state_ng.field(
            shape=(self.substeps_local + 1, self.n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        particles_info = particle_info.field(shape=self.n_particles, needs_grad=False, layout=qd.Layout.SOA)
        # Grid is only ever indexed at [f] (never [f+1]) in p2g/g2p/reset/coupler, so substeps_local frames are enough.
        # Particles still need substeps_local + 1 because g2p writes the next-frame state at [f+1].
        grid = grid_cell_state.field(
            shape=(self.substeps_local, *self.grid_res, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        return MPMData(
            owner=owner,
            particle_volume_scale=self.particle_volume_scale,
            particles=particles,
            particles_ng=particles_ng,
            particles_info=particles_info,
            grid=grid,
        )
