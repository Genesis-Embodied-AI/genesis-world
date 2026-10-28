"""Shared finite element geometry, material data and time-indexed state."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs

from . import SystemData, SystemDataDescription

if TYPE_CHECKING:
    from genesis.engine.solvers.base_solver import Solver


@dataclass(frozen=True, kw_only=True, eq=False)
class FEMData(SystemData):
    """Element mass_scaled and V_scaled are multiplied by vol_scale."""

    vol_scale: float
    elements_v: qd.Field
    elements_el: qd.Field
    elements_el_ng: qd.Field
    elements_i: qd.Field
    elements_v_info: qd.Field
    surface: qd.Field
    surface_vertices: qd.Field
    surface_elements: qd.Field
    surface_vert_mass: qd.Field


@dataclass(frozen=True, kw_only=True, eq=False)
class FEMEntityData(SystemData):
    data: FEMData
    vert_start: int
    vert_end: int
    elem_start: int
    elem_end: int
    surface_start: int
    surface_end: int


@dataclass(frozen=True, kw_only=True)
class FEMDescription(SystemDataDescription[FEMData]):
    _B: int
    vol_scale: float
    substeps_local: int
    n_vertices: int
    n_elements: int
    n_surfaces: int
    surface_vertices: np.ndarray
    surface_elements: np.ndarray
    surface_vert_mass: np.ndarray

    def allocate(self, owner: "Solver") -> FEMData:
        # element state in vertices
        element_state_v = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
        )
        # element state in elements
        element_state_el = qd.types.struct(
            actu=gs.qd_float,  # actuation
        )
        # element state without gradient
        element_state_el_ng = qd.types.struct(
            active=gs.qd_bool,
        )
        # element info (properties that remain static through time)
        element_info = qd.types.struct(
            el2v=gs.qd_ivec4,  # vertex index of an element
            mu=gs.qd_float,  # lame parameters (1)
            lam=gs.qd_float,  # lame parameters (2)
            mass_scaled=gs.qd_float,  # scaled element mass. The real mass is mass_scaled / vol_scale
            mat_idx=gs.qd_int,  # material model index
            B=gs.qd_mat3,  # inverse of the deformation gradient at rest state
            V=gs.qd_float,  # rest volume of the element
            V_scaled=gs.qd_float,  # scaled rest volume of the element
            friction_mu=gs.qd_float,  # friction coefficient for contact
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )
        element_v_info = qd.types.struct(
            mass=gs.qd_float,  # mass of the vertex
            mass_inv=gs.qd_float,  # inverse mass of the vertex
            mass_over_dt2=gs.qd_float,  # scaled mass of the vertex over dt^2
            friction_mu=gs.qd_float,  # friction coefficient for contact
        )
        elements_v = element_state_v.field(
            shape=(self.substeps_local + 1, self.n_vertices, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        elements_el = element_state_el.field(
            shape=(self.substeps_local + 1, self.n_elements, self._B), needs_grad=True, layout=qd.Layout.SOA
        )
        elements_el_ng = element_state_el_ng.field(
            shape=(self.substeps_local + 1, self.n_elements, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        elements_i = element_info.field(shape=self.n_elements, needs_grad=False, layout=qd.Layout.SOA)
        elements_v_info = element_v_info.field(shape=self.n_vertices, needs_grad=False, layout=qd.Layout.SOA)
        # surface info (for coupling)
        surface_state = qd.types.struct(
            tri2v=gs.qd_ivec3,  # vertex index of a triangle
            tri2el=gs.qd_int,  # element index of a triangle
            active=gs.qd_bool,
        )
        surface = surface_state.field(shape=self.n_surfaces, layout=qd.Layout.SOA)
        surface_vertices = qd.field(dtype=gs.qd_int, shape=len(self.surface_vertices))
        surface_elements = qd.field(dtype=gs.qd_int, shape=len(self.surface_elements))
        surface_vert_mass = qd.field(dtype=gs.qd_float, shape=len(self.surface_vertices))
        surface_vertices.from_numpy(self.surface_vertices)
        surface_elements.from_numpy(self.surface_elements)
        surface_vert_mass.from_numpy(self.surface_vert_mass)
        return FEMData(
            owner=owner,
            vol_scale=self.vol_scale,
            elements_v=elements_v,
            elements_el=elements_el,
            elements_el_ng=elements_el_ng,
            elements_i=elements_i,
            elements_v_info=elements_v_info,
            surface=surface,
            surface_vertices=surface_vertices,
            surface_elements=surface_elements,
            surface_vert_mass=surface_vert_mass,
        )
