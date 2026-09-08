import os
from dataclasses import dataclass

import mujoco
import numpy as np

import genesis as gs
from genesis.utils import mjcf as mju
from genesis.utils.mesh import get_assets_dir

from .assets import get_hf_dataset


@dataclass
class MjSim:
    model: mujoco.MjModel
    data: mujoco.MjData


def build_mujoco_sim(
    xml_path,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    adjacent_collision,
    native_ccd,
    *,
    friction_cone,
):
    if gs_solver == gs.constraint_solver.CG:
        mj_solver = mujoco.mjtSolver.mjSOL_CG
    elif gs_solver == gs.constraint_solver.Newton:
        mj_solver = mujoco.mjtSolver.mjSOL_NEWTON
    else:
        raise ValueError(f"Solver '{gs_solver}' not supported")
    if gs_integrator == gs.integrator.Euler:
        mj_integrator = mujoco.mjtIntegrator.mjINT_EULER
    elif gs_integrator == gs.integrator.implicitfast:
        mj_integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:
        raise ValueError(f"Integrator '{gs_integrator}' not supported")

    file = os.path.join(get_assets_dir(), xml_path)
    if not os.path.exists(file):
        asset_path = get_hf_dataset(pattern=xml_path)
        file = os.path.join(asset_path, xml_path)

    model = mju.build_model(file, discard_visual=True, merge_fixed_links=merge_fixed_links, links_to_keep=())

    model.opt.solver = mj_solver
    model.opt.integrator = mj_integrator
    if friction_cone == gs.friction_cone.elliptic:
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    else:
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    # FIXME: Genesis gives every contact at least the sliding-friction basis, so a geom asking for a frictionless
    # contact through 'condim' is not honoured. Raising those to 3 keeps the constraint sets comparable, since MuJoCo
    # would otherwise emit a single normal row where Genesis emits the whole basis. Geoms asking for torsional or
    # rolling friction keep their own value, which Genesis follows through its friction options.
    model.geom_condim[model.geom_condim < 3] = 3
    # FIXME: Genesis has no tendons, so a tendon's range contributes no constraint row on its side. Releasing the range
    # keeps the constraint sets comparable; a model whose behaviour depends on that range cannot be compared until
    # tendons are supported.
    model.tendon_limited[:] = 0
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    # Keep midpoint integration of standalone free bodies enabled on the MuJoCo side: Genesis implements it under
    # the implicitfast integrator, and the consistency tests are its test coverage.
    model.opt.enableflags &= ~np.uint32(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    # MuJoCo's mesh processing leaves sub-epsilon center-of-mass residuals in body_ipos while Genesis re-centers
    # meshes exactly. Midpoint integration branches on an exact ipos == 0 test, so the residuals would silently
    # route the two engines through different update rules; canonicalize the dust to zero.
    model.body_ipos[np.abs(model.body_ipos) < 1e-12] = 0.0
    if native_ccd:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    else:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    if multi_contact:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    else:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    if adjacent_collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    else:
        model.opt.disableflags &= ~np.uint32(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    data = mujoco.MjData(model)

    return MjSim(model, data)


def build_genesis_sim(
    xml_paths,
    gs_solver,
    gs_integrator,
    merge_fixed_links,
    multi_contact,
    mujoco_compatibility,
    adjacent_collision,
    gjk_collision,
    show_viewer,
    mj_sim,
    *,
    friction_cone,
    friction_torsional,
    friction_rolling,
):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=mj_sim.model.opt.timestep,
            substeps=1,
            gravity=mj_sim.model.opt.gravity,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_adjacent_collision=adjacent_collision,
            integrator=gs_integrator,
            constraint_solver=gs_solver,
            iterations=mj_sim.model.opt.iterations,
            tolerance=mj_sim.model.opt.tolerance,
            ls_iterations=mj_sim.model.opt.ls_iterations,
            ls_tolerance=mj_sim.model.opt.ls_tolerance,
            friction_cone=friction_cone,
            enable_torsional_friction=friction_torsional,
            enable_rolling_friction=friction_rolling,
            box_box_detection=True,
            enable_multi_contact=multi_contact,
            enable_mujoco_compatibility=mujoco_compatibility,
            use_gjk_collision=gjk_collision,
            # None gives a geom carrying no time constant of its own the floor, twice the timestep, as Mujoco does.
            constraint_timeconst=None,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    for path in xml_paths:
        file = os.path.join(get_assets_dir(), path)
        if not os.path.exists(file):
            asset_path = get_hf_dataset(pattern=path)
            file = os.path.join(asset_path, path)

        morph_kwargs = dict(
            file=file,
            convexify=True,
            decompose_robot_error_threshold=float("inf"),
            align=False,
        )
        if path.endswith(".xml"):
            morph = gs.morphs.MJCF(**morph_kwargs)
        else:
            morph = gs.morphs.URDF(
                fixed=True,
                merge_fixed_links=merge_fixed_links,
                links_to_keep=(),
                **morph_kwargs,
            )
        scene.add_entity(
            morph,
            visualize_contact=True,
        )

    # Force recomputation of invweights to make sure it works fine
    for link in scene.rigid_solver.links:
        link.desc.invweight[:] = -1
    for joint in scene.rigid_solver.joints:
        joint.desc.dofs_invweight[:] = -1

    # Canonicalize mesh center-of-mass dust to zero: see the matching body_ipos normalization in build_mujoco_sim.
    for link in scene.rigid_solver.links:
        link.desc.inertial_pos[np.abs(link.desc.inertial_pos) < 1e-12] = 0.0

    scene.build()

    return scene.sim
