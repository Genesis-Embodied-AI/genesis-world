"""Inertial properties of rigid links: the estimate from geometry, and how authored values complete it."""

from itertools import starmap
from typing import NamedTuple, Sequence

import numpy as np

import genesis as gs
from genesis.engine.mesh import InertialProperties
from genesis.typing import Matrix3x3Type, UnitVec4FType, Vec3FType
from genesis.utils import geom as gu

RHO_OBJECT = 600.0
RHO_ROBOT = 1500.0
RHO_MUJOCO = 1000.0

# If mass is too small, we do not care much about spatial inertia discrepancy
MASS_EPS = 0.005
AABB_EPS = 0.002
INERTIA_RATIO_MAX = 100.0
# Rounding of an authored inertia tensor, relative to its largest eigenvalue and absolute, that its checks forgive
INERTIA_EIGVAL_RTOL = 1e-6
INERTIA_EIGVAL_ATOL = 1e-9


def get_local_inertial_from_geom(
    geom_type: gs.GEOM_TYPE, data: np.ndarray | None, mesh: "gs.Mesh | None", rho: float = 1.0
) -> InertialProperties:
    """Local inertial properties (mass, center of mass, inertia tensor) of one described geometry.

    Primitive types use the analytic formula on `data`; MESH defers to the mesh's cached unit-density mass properties
    (`Mesh.inertial`) scaled by `rho`. This is the load-time computation available to kinematic and rigid
    entities alike (it operates on the parsed info, not on a geom object), so the anchor it feeds matches the finalized
    geom-derived inertia exactly.
    """
    geom_com_local = np.zeros(3)
    if geom_type == gs.GEOM_TYPE.PLANE:
        geom_mass = 0.0
        geom_inertia_local = np.zeros(3, dtype=gs.np_float)
    elif geom_type == gs.GEOM_TYPE.SPHERE:
        radius = data[0]
        geom_mass = (4.0 / 3.0) * np.pi * radius**3 * rho
        I = (2.0 / 5.0) * geom_mass * radius**2
        geom_inertia_local = np.diag([I, I, I])
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        hx, hy, hz = data[:3]
        geom_mass = (4.0 / 3.0) * np.pi * hx * hy * hz * rho
        geom_inertia_local = (geom_mass / 5.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        radius, height = data[:2]
        geom_mass = np.pi * radius**2 * height * rho
        I_r = (geom_mass / 12.0) * (3.0 * radius**2 + height**2)
        I_z = 0.5 * geom_mass * radius**2
        geom_inertia_local = np.diag([I_r, I_r, I_z])
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        radius, height = data[:2]
        m_cyl = np.pi * radius**2 * height * rho
        m_sph = (4.0 / 3.0) * np.pi * radius**3 * rho
        geom_mass = m_cyl + m_sph
        I_r = (m_cyl * radius**2 / 12.0 * (3.0 + height**2 / radius**2)) + (
            m_sph * radius**2 / 4.0 * (83.0 / 80.0 + (height / radius + 3.0 / 4.0) ** 2)
        )
        I_h = 0.5 * m_cyl * radius**2 + (2.0 / 5.0) * m_sph * radius**2
        geom_inertia_local = np.diag([I_r, I_r, I_h])
    elif geom_type == gs.GEOM_TYPE.BOX:
        hx, hy, hz = data[:3]
        geom_mass = (hx * hy * hz) * rho
        geom_inertia_local = (geom_mass / 12.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    else:
        # MESH type: reuse the mesh's cached unit-density mass properties; mass and inertia scale linearly with density.
        geom_mass = mesh.inertial.mass * rho
        geom_com_local = mesh.inertial.com
        geom_inertia_local = mesh.inertial.i * rho

    return InertialProperties(geom_mass, geom_com_local, geom_inertia_local)


class GeomInertialInfo(NamedTuple):
    """A geom's intrinsic 'inertial' (in the geom's own frame) placed at pose 'pos'/'quat' in the parent link, as
    consumed by 'compose_inertial_properties'."""

    inertial: InertialProperties
    pos: Vec3FType
    quat: UnitVec4FType


def compose_inertial_properties(geoms_inertial_info: Sequence[GeomInertialInfo]) -> InertialProperties:
    """
    Compose mass, center of mass, and inertia tensor from multiple geometries.
    """
    global_mass = 0.0
    if geoms_inertial_info:
        geoms_inertial, geoms_pos, geoms_quat = zip(*geoms_inertial_info)
        geoms_mass = np.asarray([inertial.mass for inertial in geoms_inertial])
        global_mass = geoms_mass.sum()

    if global_mass == 0.0:
        return InertialProperties(0.0, np.zeros(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64))

    geoms_com_local = [inertial.com for inertial in geoms_inertial]
    geoms_I_local = [inertial.i for inertial in geoms_inertial]

    # Compute world COMs of each geom
    geoms_com_world = np.stack(
        tuple(starmap(gu.transform_by_trans_quat, zip(geoms_com_local, geoms_pos, geoms_quat))), axis=0
    )

    # Compute total COM
    global_com = (geoms_mass[:, None] * geoms_com_world).sum(axis=0) / global_mass

    # Accumulate inertia about global COM
    global_inertia = np.zeros((3, 3), dtype=np.float64)

    # Transform local inertias directly to global COM frame using parallel axis theorem
    for geom_mass, geom_I_local, geom_quat, geom_com_world in zip(
        geoms_mass, geoms_I_local, geoms_quat, geoms_com_world
    ):
        T_offset = gu.trans_quat_to_T(global_com - geom_com_world, geom_quat)
        geom_I_world = gu.transform_inertia_by_T(geom_I_local, T_offset, geom_mass)
        global_inertia += geom_I_world

    return InertialProperties(global_mass, global_com, global_inertia)


def compose_inertial_from_g_infos(g_infos: Sequence[dict], rho: float) -> InertialProperties:
    """Compose the inertial of the geoms one link holds, at 'rho' where a geom states no density of its own.

    Every primitive collision type is handled analytically, and a mesh defers to its cached unit-density mass
    properties. A geom the link is only drawn with is taken as its visual mesh, so one call covers either kind.

    Parameters
    ----------
    g_infos : Sequence[dict]
        Parsed geom infos to compose the inertial from.
    rho : float
        Material density (kg/m^3), used for every geom info without its own authored density.

    Returns
    -------
    inertial : InertialProperties
        Mass, center of mass and inertia tensor of the geoms together, in the frame they are stated in.
    """
    return compose_inertial_properties(
        tuple(
            GeomInertialInfo(
                get_local_inertial_from_geom(
                    gs.GEOM_TYPE.MESH if "vmesh" in g_info else g_info["type"],
                    None if "vmesh" in g_info else g_info.get("data"),
                    g_info["vmesh"] if "vmesh" in g_info else g_info["mesh"],
                    rho if g_info.get("density") is None else g_info["density"],
                ),
                np.asarray(g_info.get("pos", gu.zero_pos()), dtype=gs.np_float),
                np.asarray(g_info.get("quat", gu.identity_quat()), dtype=gs.np_float),
            )
            for g_info in g_infos
        )
    )


class LinkInertial(NamedTuple):
    """A link's (or variant's) finalized inertial in the solver's representation. Distinct from 'InertialProperties',
    whose 'i' is a single tensor, as the solver stores the principal tensor and its orientation separately."""

    mass: float
    """Mass of the link or variant."""
    com: Vec3FType
    """Center of mass in the link frame, the origin of the public 'link_COM' reference frame."""
    quat: UnitVec4FType
    """Orientation of the frame 'inertia' is expressed in, identity when derived from geometry. Its axes are those of
    the public 'link_COM' reference frame."""
    inertia: Matrix3x3Type
    """Inertia tensor about 'com', expressed in the 'quat' frame."""


class LinkInertialInfo(NamedTuple):
    """A link's (or variant's) load-time inertial data.

    Computed while the parsed geom infos (and their authored per-geom densities) are still available, and consumed
    by the post-load passes. 'props' feeds the align anchor. 'is_mass_explicit' feeds the all-or-none source check
    in '_align_free_roots': True when the mass is explicit in the asset (an explicit mass, or an authored density on
    every geom), False for a pure geometry estimate (the true mass is a uniform material-density rescale of it), and
    None when the link mixes geoms with and without an authored density (neither explicit nor uniformly rescalable).
    'hint' is the material-density-resolved geometry estimate that resolving a link description consumes (None for
    kinematic entities, which have no dynamics)."""

    props: LinkInertial
    is_mass_explicit: bool | None
    hint: InertialProperties | None


def finalize_inertial(
    explicit_mass, explicit_com, explicit_quat, explicit_inertia, hint_mass, hint_com, hint_inertia, clamp_min_mass=True
) -> LinkInertial:
    """Resolve a link's local inertial from its parsed explicit values and a geometry-derived estimate (hint).

    Each explicit value is used when given, and the geometry estimate fills in the rest. An explicit mass rescales a
    geometry-derived inertia to it. An omitted center of mass falls back to the geometry estimate, except when an
    explicit inertia matrix is given, in which case it is the link-frame origin.

    The geometry estimate is computed once from the geometry the link holds. Both the link description and the align
    anchor resolve their inertial here, so the rigid dynamics inertia and the align anchor agree.

    With ``clamp_min_mass`` the resolved mass is floored at ``gs.EPS`` so a geometry-less moving link stays
    non-singular in the dynamics. The align stash and the variants of a link fixed to the world pass ``False``, so a
    genuinely massless link keeps its ``0.0`` mass there and leaves the fixed-subtree composite alone.
    """
    mass, com, quat, inertia = explicit_mass, explicit_com, explicit_quat, explicit_inertia
    if mass is not None and hint_mass > 0.0:
        hint_inertia = hint_inertia * (mass / hint_mass)
        hint_mass = mass
    if mass is None:
        mass = hint_mass
    if inertia is None:
        if com is None:
            com = hint_com
        inertia, quat = hint_inertia, gu.identity_quat()
    elif com is None:
        # Falling back to the geometry estimate here would discard an otherwise complete authored inertia.
        com = gu.zero_pos()
    if quat is None:
        quat = gu.identity_quat()
    return LinkInertial(
        # For non-fixed links, the mass must be non-zero in case for coupling with deformable body solvers.
        max(mass, gs.EPS) if clamp_min_mass else mass,
        np.asarray(com, dtype=gs.np_float),
        np.asarray(quat, dtype=gs.np_float),
        np.asarray(inertia, dtype=gs.np_float),
    )
