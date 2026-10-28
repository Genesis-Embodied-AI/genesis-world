"""Particle state and rest topology for position-based dynamics (PBD)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import quadrants as qd

import genesis as gs

from . import SystemData, SystemDataDescription
from .particles import ParticleEntityData

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class PBDData(SystemData):
    """The reordered fields hold the particles sorted by spatial hash cell, which the substeps and the coupling read."""

    particles_info: qd.Field
    particles_info_reordered: qd.Field
    particles: qd.Field
    particles_reordered: qd.Field
    particles_ng: qd.Field
    particles_ng_reordered: qd.Field
    edges_info: qd.Field
    inner_edges_info: qd.Field
    elems_info: qd.Field


@dataclass(frozen=True, kw_only=True, eq=False)
class PBDEntityData(ParticleEntityData):
    """The particle range of one entity and the ranges of its rest topology, each empty where it holds none."""

    edge_start: int
    edge_end: int
    inner_edge_start: int
    inner_edge_end: int
    elem_start: int
    elem_end: int


@dataclass(frozen=True, kw_only=True)
class PBDDescription(SystemDataDescription[PBDData]):
    _B: int
    n_particles: int
    n_edges: int
    n_inner_edges: int
    n_elems: int

    def allocate(self, owner: "Solver") -> PBDData:
        # particles information (static)
        particle_info = qd.types.struct(
            mass=gs.qd_float,
            pos_rest=gs.qd_vec3,
            rho_rest=gs.qd_float,
            material_type=gs.qd_int,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            air_resistance=gs.qd_float,
            density_relaxation=gs.qd_float,
            viscosity_relaxation=gs.qd_float,
        )
        # particles state (dynamic). A particle with 'free' cleared ignores the internal forces, and the user alone
        # moves it until its release.
        particle_state = qd.types.struct(
            free=gs.qd_bool,
            pos=gs.qd_vec3,  # position
            ipos=gs.qd_vec3,  # initial position
            dpos=gs.qd_vec3,  # delta position
            vel=gs.qd_vec3,  # velocity
            lam=gs.qd_float,
            rho=gs.qd_float,
        )
        # dynamic particle state without gradient
        particle_state_ng = qd.types.struct(reordered_idx=gs.qd_int, active=gs.qd_bool)
        # Stretch constraint between the vertices (v1, v2) of an edge
        edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            stretch_compliance=gs.qd_float,
            stretch_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
        )
        # Bending constraint across the edge (v1, v2) shared by the faces (v1, v2, v3) and (v1, v2, v4)
        inner_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            bending_compliance=gs.qd_float,
            bending_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        elem_info = qd.types.struct(
            vol_rest=gs.qd_float,
            volume_compliance=gs.qd_float,
            volume_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        particles_info = particle_info.field(shape=(self.n_particles,), layout=qd.Layout.SOA)
        particles_info_reordered = particle_info.field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        particles = particle_state.field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        particles_reordered = particle_state.field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        particles_ng = particle_state_ng.field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        particles_ng_reordered = particle_state_ng.field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        edges_info = edge_info.field(shape=(max(self.n_edges, 1),), layout=qd.Layout.SOA)
        inner_edges_info = inner_edge_info.field(shape=(max(self.n_inner_edges, 1),), layout=qd.Layout.SOA)
        elems_info = elem_info.field(shape=(max(self.n_elems, 1),), layout=qd.Layout.SOA)
        return PBDData(
            owner=owner,
            particles_info=particles_info,
            particles_info_reordered=particles_info_reordered,
            particles=particles,
            particles_reordered=particles_reordered,
            particles_ng=particles_ng,
            particles_ng_reordered=particles_ng_reordered,
            edges_info=edges_info,
            inner_edges_info=inner_edges_info,
            elems_info=elems_info,
        )
