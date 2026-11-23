"""The data and kernels of the visual meshes of a kinematic or rigid solver and of their poses, which the scoped render
contexts of these solvers hold (see 'KinematicScopedRenderContext').
"""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.data import GEOMS, LinksState
from genesis.utils.data import ENVS, Data, DataKind, Size, eoa, field

VGEOMS = Size("vgeoms")
VVERTS = Size("vverts")
VFACES = Size("vfaces")
# The propellers of the drones of a solver whose propellers turn as drawn
PROPELLERS = Size("propellers")
# The visual geoms showing DynamicVertices, their vertices, those of the visual geoms holding a batched variant, the
# variants of each, and the vertices of the variants a mesh states.
DYNAMIC_VGEOMS = Size("dynamic_vgeoms")
DYNAMIC_VVERTS = Size("dynamic_vverts")
BATCHED_VVERTS = Size("batched_vverts")
VGEOMS_VARIANTS = Size("vgeoms_variants")
VARIANTS_VVERTS = Size("variants_vverts")

# Whether each environment holds each geom or visual geom, which the heterogeneous variants alone require
VARIANTS = "variants"


class RigidRenderVGeomsInfo(Data, kind=DataKind.CONSTANT, shape=(VGEOMS,)):
    """The pose of each visual geom in its link, the range of its vertices, and the propeller it draws if any."""

    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    vvert_start: qd.Tensor = field(int)
    vvert_end: qd.Tensor = field(int)
    link_idx: qd.Tensor = field(int)
    # The propeller among those of the drones of the solver, -1 for a visual geom drawing none
    propeller_idx: qd.Tensor = field(int)
    # The angle the propeller turns by per revolution its drone accumulated (see 'DroneEntity.set_propellers_rpm')
    propeller_rad_per_rev: qd.Tensor = field(float)


class RigidRenderVVertsInfo(Data, kind=DataKind.CONSTANT, shape=(VVERTS,)):
    """The vertices of the meshes of the visual geoms, in the frame of their visual geom."""

    init_pos: qd.Tensor = field((float, 3))
    init_normal: qd.Tensor = field((float, 3))


class RigidRenderVFacesInfo(Data, kind=DataKind.CONSTANT, shape=(VFACES,)):
    """The triangles of the meshes of the visual geoms, as vertices among those of the solver."""

    vverts_idx: qd.Tensor = field((int, 3))


class RigidRenderVGeomsState(Data, kind=DataKind.DERIVED, shape=(VGEOMS, ENVS)):
    """The world pose of each visual geom in each environment, and whether the environment holds it."""

    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    is_active: qd.Tensor = field(bool, labels=VARIANTS)


class RigidRenderPropellers(Data, kind=DataKind.DERIVED, shape=(PROPELLERS, ENVS)):
    """The revolutions each propeller accumulated in each environment, which turn it as drawn."""

    revs: qd.Tensor = field(float)


class RigidDynamicVVerts(Data):
    """The variants of the vertices of the visual geoms showing DynamicVertices, the variant shown, and the vertices as
    placed in the world."""

    # The slot of each visual geom, the first of its vertices among the dynamic vertices and among the vertices of the
    # batched variants (-1 for a visual geom without), and the first of its variants.
    vgeoms_slot: qd.Tensor = field(int, shape=(DYNAMIC_VGEOMS,), kind=DataKind.CONSTANT)
    vgeoms_vvert_start: qd.Tensor = field(int, shape=(DYNAMIC_VGEOMS,), kind=DataKind.CONSTANT)
    vgeoms_batched_vvert_start: qd.Tensor = field(int, shape=(DYNAMIC_VGEOMS,), kind=DataKind.CONSTANT)
    vgeoms_variant_start: qd.Tensor = field(int, shape=(DYNAMIC_VGEOMS,), kind=DataKind.CONSTANT)
    # The variant each environment shows, -1 for the batched variant and ASSET_VARIANT_IDX for the mesh of the asset
    # (see render_context.py)
    vgeoms_variant_idx: qd.Tensor = field(int, shape=(DYNAMIC_VGEOMS, ENVS), kind=DataKind.DERIVED)
    # The first vertex of each variant among the vertices of the variants, -1 for the mesh of the asset
    variants_vvert_start: qd.Tensor = field(int, shape=(VGEOMS_VARIANTS,), kind=DataKind.CONSTANT)
    variants_pos: qd.Tensor = field((float, 3), shape=(VARIANTS_VVERTS,), kind=DataKind.CONSTANT)
    # The batched variant, in the frame of the visual geoms, and the vertices shown, in the world
    batched_pos: qd.Tensor = field((float, 3), shape=(BATCHED_VVERTS, ENVS), kind=DataKind.DERIVED)
    pos: qd.Tensor = field((float, 3), shape=(DYNAMIC_VVERTS, ENVS), kind=DataKind.DERIVED)


class RigidRenderData(Data):
    """The visual geoms of a kinematic or rigid solver as drawn, in the order of their slots (see
    'KinematicScopedRenderContext')."""

    vgeoms_info: RigidRenderVGeomsInfo
    vverts_info: RigidRenderVVertsInfo
    vfaces_info: RigidRenderVFacesInfo
    vgeoms: RigidRenderVGeomsState
    propellers: RigidRenderPropellers
    dynamic_vverts: RigidDynamicVVerts


class RigidRenderGeoms(Data, kind=DataKind.CONSTANT, shape=(GEOMS, ENVS)):
    """Whether each environment holds each collision geom of a rigid solver, which its debug representation reads."""

    is_active: qd.Tensor = field(bool, labels=VARIANTS)


@qd.kernel(fastcache=True)
def kernel_update_render_vgeoms(links_state: LinksState, render: RigidRenderData):
    """Place the visual geoms of a solver on their link in every environment, a propeller turned by the revolutions its
    drone accumulated."""
    _B = render.vgeoms.pos.shape[1]
    for i_vg, i_b in qd.ndrange(render.vgeoms_info.link_idx.shape[0], _B, axes=qd.static(eoa(render.vgeoms.pos))):
        i_l = render.vgeoms_info.link_idx[i_vg]
        vgeom_pos, vgeom_quat = gu.qd_transform_pos_quat_by_trans_quat(
            render.vgeoms_info.pos[i_vg],
            render.vgeoms_info.quat[i_vg],
            links_state.pos[i_l, i_b],
            links_state.quat[i_l, i_b],
        )
        i_pp = render.vgeoms_info.propeller_idx[i_vg]
        if i_pp >= 0:
            rad = render.propellers.revs[i_pp, i_b] * render.vgeoms_info.propeller_rad_per_rev[i_vg]
            vgeom_quat = gu.qd_transform_quat_by_quat(
                gu.qd_rotvec_to_quat(qd.Vector([0.0, 0.0, rad], dt=gs.qd_float)), vgeom_quat
            )
        render.vgeoms.pos[i_vg, i_b] = vgeom_pos
        render.vgeoms.quat[i_vg, i_b] = vgeom_quat


@qd.kernel(fastcache=True)
def kernel_update_dynamic_vverts(render: RigidRenderData):
    """Place the vertices of the variant each environment shows of the visual geoms showing DynamicVertices, at the pose
    of their visual geom, in every environment."""
    _B = render.dynamic_vverts.pos.shape[1]
    for i_dg, i_b in qd.ndrange(render.dynamic_vverts.vgeoms_slot.shape[0], _B):
        i_vg = render.dynamic_vverts.vgeoms_slot[i_dg]
        i_variant = render.dynamic_vverts.vgeoms_variant_idx[i_dg, i_b]
        # The first vertex of the variant among the vertices of the variants, -1 for the mesh of the asset
        i_pv_start = -1
        if i_variant >= 0:
            i_gv = render.dynamic_vverts.vgeoms_variant_start[i_dg] + i_variant
            i_pv_start = render.dynamic_vverts.variants_vvert_start[i_gv]
        i_vv_start = render.vgeoms_info.vvert_start[i_vg]
        i_dv_start = render.dynamic_vverts.vgeoms_vvert_start[i_dg]
        for i_vv in range(i_vv_start, render.vgeoms_info.vvert_end[i_vg]):
            i_dv = i_dv_start + i_vv - i_vv_start
            pos = render.vverts_info.init_pos[i_vv]
            if i_variant == -1:
                i_bv = render.dynamic_vverts.vgeoms_batched_vvert_start[i_dg] + i_vv - i_vv_start
                pos = render.dynamic_vverts.batched_pos[i_bv, i_b]
            elif i_pv_start >= 0:
                pos = render.dynamic_vverts.variants_pos[i_pv_start + i_vv - i_vv_start]
            render.dynamic_vverts.pos[i_dv, i_b] = gu.qd_transform_by_trans_quat(
                pos, render.vgeoms.pos[i_vg, i_b], render.vgeoms.quat[i_vg, i_b]
            )


@qd.kernel(fastcache=True)
def kernel_add_propellers_revs(
    propeller_start: int, propellers_rpm: qd.types.ndarray(), propellers: RigidRenderPropellers, revs_period: float
):
    """Add the speed in revolutions per minute of the propellers of a drone, those from `propeller_start` on, to the
    revolutions they accumulated, kept modulo `revs_period`."""
    for i_b, i_pp_ in qd.ndrange(propellers_rpm.shape[0], propellers_rpm.shape[1]):
        i_pp = propeller_start + i_pp_
        propellers.revs[i_pp, i_b] = qd.math.mod(propellers.revs[i_pp, i_b] + propellers_rpm[i_b, i_pp_], revs_period)


@qd.kernel(fastcache=True)
def kernel_reset_propellers_revs(envs_idx: qd.types.ndarray(), propellers: RigidRenderPropellers):
    for i_pp, i_b_ in qd.ndrange(propellers.revs.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        propellers.revs[i_pp, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_set_vverts_variant(
    envs_idx: qd.types.ndarray(), i_dg: int, variants_idx: qd.types.ndarray(), dynamic_vverts: RigidDynamicVVerts
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        dynamic_vverts.vgeoms_variant_idx[i_dg, i_b] = variants_idx[i_b_]


@qd.kernel(fastcache=True)
def kernel_set_batched_vverts(
    envs_idx: qd.types.ndarray(),
    i_dg: int,
    vvert_start: int,
    vverts: qd.types.ndarray(),
    dynamic_vverts: RigidDynamicVVerts,
):
    """Write the batched variant of the DynamicVertices of a visual geom, whose vertices start at `vvert_start` among
    those of the batched variant, in some environments, and show it there."""
    for i_b_, i_vv_ in qd.ndrange(envs_idx.shape[0], vverts.shape[1]):
        i_b = envs_idx[i_b_]
        for j in qd.static(range(3)):
            dynamic_vverts.batched_pos[vvert_start + i_vv_, i_b][j] = vverts[i_b_, i_vv_, j]
        if i_vv_ == 0:
            dynamic_vverts.vgeoms_variant_idx[i_dg, i_b] = -1


@qd.kernel(fastcache=True)
def kernel_reset_vverts_variant(envs_idx: qd.types.ndarray(), dynamic_vverts: RigidDynamicVVerts):
    for i_dg, i_b_ in qd.ndrange(dynamic_vverts.vgeoms_slot.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        dynamic_vverts.vgeoms_variant_idx[i_dg, i_b] = 0
