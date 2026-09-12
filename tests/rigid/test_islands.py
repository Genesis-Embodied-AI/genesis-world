import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.fixture
def multi_free_body_path(tmp_path):
    # A single MJCF entity holding several free bodies (b0, b1, b2) is common. b1 carries a hinge child b1c, so a
    # kinematic edge must keep the child in its parent's island. The ground plane lives in the entity's worldbody: the
    # entity owns a 0-dof static link that every free body contacts, yet each free body must remain its own island -
    # the shared static link must not couple them.
    mjcf = ET.Element("mujoco", model="multi_free_body")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", size="5 5 0.01")
    b0 = ET.SubElement(worldbody, "body", name="b0", pos="0.0 0.0 0.3")
    ET.SubElement(b0, "freejoint")
    ET.SubElement(b0, "geom", type="box", size="0.05 0.05 0.05")
    b1 = ET.SubElement(worldbody, "body", name="b1", pos="0.5 0.0 0.3")
    ET.SubElement(b1, "freejoint")
    ET.SubElement(b1, "geom", type="box", size="0.05 0.05 0.05")
    b1c = ET.SubElement(b1, "body", name="b1c", pos="0.0 0.0 0.12")
    ET.SubElement(b1c, "joint", type="hinge", axis="0 0 1")
    ET.SubElement(b1c, "geom", type="box", size="0.05 0.05 0.05")
    b2 = ET.SubElement(worldbody, "body", name="b2", pos="1.0 0.0 0.3")
    ET.SubElement(b2, "freejoint")
    ET.SubElement(b2, "geom", type="box", size="0.05 0.05 0.05")
    file_path = str(tmp_path / "multi_free_body.xml")
    ET.ElementTree(mjcf).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.mark.required
def test_constraint_capacity_covers_friction_rows(show_viewer):
    # The island grouping writes one entry per assembled constraint, so its buffers must cover the same bound as
    # the constraint solver, including the extra torsional and rolling rows per contact.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    for i_box in range(2):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=(0.0, 0.0, 0.2 + 0.4 * i_box),
            ),
        )
    scene.build()
    constraint_solver = scene.sim.rigid_solver.constraint_solver
    assert constraint_solver.constraint_state.island.constraint_id.shape[0] >= constraint_solver.len_constraints


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_partition_logics(show_viewer, n_envs, multi_free_body_path):
    # The welded pair never touches, so only the equality edge couples them: without it the partition would split them
    # and the weld would be solved across two islands. A fixed body carries no dofs and joins no island. The
    # multi-free-body MJCF entity (offset clear of the boxes) is a single Genesis entity that must split into one island
    # per free-body subtree, never one dense block - its hinge child stays in its parent's island via a kinematic edge.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_bottom = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_top = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.16),
        )
    )
    box_weld_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.05),
        )
    )
    box_weld_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.3, 0.0, 0.05),
        )
    )
    box_alone = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.05),
        )
    )
    multibody = scene.add_entity(
        gs.morphs.MJCF(
            file=multi_free_body_path,
            pos=(0.0, 5.0, 0.0),
        )
    )
    scene.build(n_envs=n_envs)

    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)

    for _ in range(45):
        scene.step()

    # The partition is rebuilt inside every step; inspect the one the solver actually used this step.
    solver = scene.rigid_solver
    island_state = solver.constraint_solver.constraint_state.island

    island_idx = qd_to_numpy(island_state.links_island_idx)
    island_of = {
        name: island_idx[entity.base_link_idx]
        for name, entity in (
            ("bottom", box_bottom),
            ("top", box_top),
            ("weld_a", box_weld_a),
            ("weld_b", box_weld_b),
            ("alone", box_alone),
        )
    }
    assert all((v >= 0).all() for v in island_of.values())
    assert_equal(island_of["top"], island_of["bottom"])
    assert_equal(island_of["weld_a"], island_of["weld_b"])
    # The stack, the welded pair and the lone box land in three distinct islands in every env.
    assert (island_of["bottom"] != island_of["weld_a"]).all()
    assert (island_of["bottom"] != island_of["alone"]).all()
    assert (island_of["weld_a"] != island_of["alone"]).all()

    # The multi-free-body entity splits into one island per free body (b0, b1, b2), each distinct from the others and
    # from the box islands; the entity's static ground link joins no island. Its hinge child b1c lands in its parent
    # b1's island via the kinematic edge.
    multibody_bases = [link for link in multibody.links if link.parent_idx == -1 and link.n_dofs > 0]
    assert len(multibody_bases) == 3
    base_islands = [island_idx[link.idx] for link in multibody_bases]
    assert all((isl >= 0).all() for isl in base_islands)
    for i, isl_a in enumerate(base_islands):
        for isl_b in base_islands[i + 1 :]:
            assert (isl_a != isl_b).all()
        for box_island in island_of.values():
            assert (isl_a != box_island).all()
    b1c = next(link for link in multibody.links if link.parent_idx != -1)
    assert_equal(island_idx[b1c.idx], island_idx[b1c.parent_idx])

    # Three box islands plus three free-body islands.
    assert_equal(qd_to_numpy(island_state.n_islands), 6)

    # Per env: each free box has 6 dofs (stack and welded pair hold 12 each, lone box 6; the free bodies hold 6, 7 with
    # the hinge child, and 6); per-island constraint counts sum back to the env total; and the lone island holds exactly
    # the lone box's dofs.
    n_islands = qd_to_numpy(island_state.n_islands)
    island_dof_n = qd_to_numpy(island_state.dof_slices.n)
    island_dof_start = qd_to_numpy(island_state.dof_slices.start)
    dof_id = qd_to_numpy(island_state.dof_id)
    island_constraint_n = qd_to_numpy(island_state.constraint_slices.n)
    n_constraints = qd_to_numpy(solver.constraint_solver.constraint_state.n_constraints)
    alone_dofs = list(range(box_alone.dof_start, box_alone.dof_start + box_alone.n_dofs))
    for i_env in range(island_idx.shape[1]):
        n = n_islands[i_env]
        assert sorted(island_dof_n[:n, i_env].tolist()) == [6, 6, 6, 7, 12, 12]
        assert island_constraint_n[:n, i_env].sum() == n_constraints[i_env]
        assert island_constraint_n[island_of["weld_a"][i_env], i_env] >= 1
        k = island_of["alone"][i_env]
        seg = dof_id[island_dof_start[k, i_env] : island_dof_start[k, i_env] + island_dof_n[k, i_env], i_env]
        assert sorted(seg.tolist()) == alone_dofs

    # The per-component solve keeps the free bodies stable: they settle on the plane (half-extent 0.05) rather than
    # exploding or sinking through it.
    free_body_z = np.stack(
        [np.atleast_1d(tensor_to_array(link.get_pos())[..., 2]) for link in multibody.links if link.n_dofs > 0]
    )
    assert ((free_body_z > 0.0) & (free_body_z < 0.5)).all()


@pytest.mark.required
def test_partition_maximal_and_invariance(show_viewer, fixed_base_dual_arm):
    # The dual arm hanging from a fixed torso against its twin whose free torso is welded to the world at runtime:
    # the twin is one island throughout, the fixed one splits per arm until the arms touch, and both fall alike. The
    # arms of the first env start lower, so its islands merge first.
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -4.0, 1.5),
            camera_lookat=(1.5, 0.0, 0.8),
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    dual_arm = scene.add_entity(
        gs.morphs.URDF(
            file=fixed_base_dual_arm,
            pos=(0.0, 0.0, 1.0),
            fixed=True,
        )
    )
    dual_arm_welded = scene.add_entity(
        gs.morphs.URDF(
            file=fixed_base_dual_arm,
            pos=(3.0, 0.0, 1.0),
        )
    )
    scene.build(n_envs=2)
    scene.rigid_solver.add_weld_constraint(dual_arm_welded.base_link_idx, plane.base_link_idx)
    dual_arm.set_dofs_position([[0.9, -0.9], [0.0, 0.0]])
    dual_arm_welded.set_dofs_position([[0.9, -0.9], [0.0, 0.0]], dofs_idx_local=[6, 7])
    n_islands = scene.rigid_solver.constraint_solver.constraint_state.island.n_islands

    # The welded dual arm is one island, the two arms of the fixed dual arm are two islands until they touch, one from
    # then on. Until the arms touch the twins fall alike, at rest they settle alike up to the compliance of the weld.
    has_envs_differed = False
    for i_step in range(80):
        scene.step()
        is_arms_touching = tensor_to_array(dual_arm.get_contacts(with_entity=dual_arm)["valid_mask"].any(dim=-1))
        assert_equal(qd_to_numpy(n_islands), 3 - is_arms_touching)
        has_envs_differed |= is_arms_touching[0] != is_arms_touching[1]
        if i_step == 0:
            assert not is_arms_touching.any()
        if i_step == 39:
            arms_qpos_diff = dual_arm_welded.get_dofs_position()[..., 6:] - dual_arm.get_dofs_position()
            assert_allclose(arms_qpos_diff[~is_arms_touching], 0.0, tol=1e-3)
            assert_allclose(arms_qpos_diff[is_arms_touching], 0.0, tol=5e-3)
    assert has_envs_differed
    assert_allclose(dual_arm_welded.get_dofs_position()[..., 6:], dual_arm.get_dofs_position(), tol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_partition_track_changes(show_viewer, n_envs):
    # The partition is rebuilt every step, so it must track contacts forming (merge) and breaking (split).
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_lower = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_upper = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.40),
        )
    )
    scene.build(n_envs=n_envs)

    # The step rebuilds the partition; read the island count the solver actually used this step.
    island_state = scene.rigid_solver.constraint_solver.constraint_state.island

    scene.step()
    assert_equal(qd_to_numpy(island_state.n_islands), 2)
    for _ in range(45):
        scene.step()
    assert_equal(qd_to_numpy(island_state.n_islands), 1)
    box_upper.set_pos([0.0, 0.0, 0.40])
    scene.step()
    assert_equal(qd_to_numpy(island_state.n_islands), 2)


@pytest.mark.required
@pytest.mark.parametrize("noslip_iterations", [0, 5])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_solve_correctness(show_viewer, noslip_iterations, n_envs):
    # Islands of different make-up (a two-box stack, a welded pair, a lone box) are solved side by side, each on its own
    # block of the Hessian, and the noslip pass refines the island-solved accelerations. The ground truth is the rest
    # state statics dictate: the heights of the boxes, the weld offset and a vanishing velocity.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=noslip_iterations,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_bottom = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_top = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.16),
        )
    )
    box_weld_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.05),
        )
    )
    box_weld_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.3, 0.0, 0.05),
        )
    )
    box_alone = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.05),
        )
    )
    scene.build(n_envs=n_envs)

    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)
    for _ in range(45):
        scene.step()
    boxes = (box_bottom, box_top, box_weld_a, box_weld_b, box_alone)
    heights = np.stack([np.atleast_1d(tensor_to_array(box.get_pos())[..., 2]) for box in boxes])
    assert_allclose(heights, np.array([0.05, 0.15, 0.05, 0.05, 0.05])[:, None], tol=5e-3)
    weld_offset = tensor_to_array(box_weld_b.get_pos()) - tensor_to_array(box_weld_a.get_pos())
    assert_allclose(weld_offset, [0.3, 0.0, 0.0], tol=5e-3)
    assert (scene.rigid_solver.get_dofs_velocity().abs() < 1e-4).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_islands_converge_independently(show_viewer, n_envs):
    # Each island converges on its own criterion: a 1 g die resting beside a Franka settles to the residual its own
    # inertia allows, whatever the arm beside it tolerates. CG converges gradually, so its exit is where this shows.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.CG,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -1.5, 1.0),
            camera_lookat=(1.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    die = scene.add_entity(
        gs.morphs.Box(
            size=(0.01, 0.01, 0.01),
            pos=(1.5, 0.0, 0.005),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )
    scene.build(n_envs=n_envs)

    franka.control_dofs_position(franka.get_dofs_position())
    for _ in range(40):
        scene.step()
    for _ in range(20):
        scene.step()
        assert (die.get_dofs_velocity()[..., 3:].abs() < 1e-4).all()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_monolith_seed_oversaturated(show_viewer, monkeypatch):
    # enable_cooperative_constraint_kernels is bounded by get_gpu_core_count(), so faking extreme GPU saturation
    # (get_gpu_core_count -> 1) disables the cooperative kernels at 2 envs - a small-scale stand-in for the
    # >get_gpu_core_count() env regime. With the monolith arm pinned, the seed factor + gradient come from the
    # monolith body itself (see is_self_seeding in func_solve_init); a stale Mgrad drops the boxes through the floor.
    import genesis.engine.solvers.rigid.rigid_solver as rigid_solver
    from genesis.utils.array_class import RigidSimStaticConfig

    monkeypatch.setattr(rigid_solver, "get_gpu_core_count", lambda: 1)
    # The gap is in the monolith arm's init path, so pin the arm (otherwise the autotuner might pick the decomposed
    # arm, which always seeds and would hide the regression).
    _orig_static_config_init = RigidSimStaticConfig.__init__

    def _force_monolith(self, *args, **kwargs):
        kwargs["prefer_decomposed_solver"] = 0
        _orig_static_config_init(self, *args, **kwargs)

    monkeypatch.setattr(RigidSimStaticConfig, "__init__", _force_monolith)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    # Four spaced free boxes: 24 dofs (cooperative-eligible at n_dofs >= 16, but disabled here by the faked saturation)
    # that fit the shared tile, four independent islands - the shared-fitting case where the in-kernel seed gates off.
    boxes = [
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.4 * i, 0.0, 0.2),
            )
        )
        for i in range(4)
    ]
    scene.build(n_envs=2)

    cfg = scene.rigid_solver.rigid_config
    # Guard against the test silently ceasing to exercise the gap (e.g. if the saturation heuristic changes).
    assert not cfg.enable_cooperative_constraint_kernels

    for _ in range(150):
        scene.step()

    z = np.stack([tensor_to_array(box.get_pos())[..., 2] for box in boxes])
    vel = tensor_to_array(scene.rigid_solver.get_dofs_velocity())
    assert not np.isnan(z).any()
    # The boxes rest on the floor at half their size; a stale seed would never apply the contact response.
    assert_allclose(z, 0.05, tol=5e-3)
    assert np.abs(vel).max() < 0.1


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_pruning(show_viewer, n_envs):
    # A convex-decomposed box is a compound body (27 sub-box geoms on one link), so its ground contacts pile up per
    # link-pair and pruning collapses them. The island construction reads contacts through contact_sort_idx, so pruning
    # and islands run together; each box then settles with its bottom face on the plane, center at its half-height.
    half = 0.1
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -4.0, 2.5),
            camera_lookat=(0.5, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    sub_meshes = []
    for sx, sy, sz in product((-1, 0, 1), repeat=3):
        mesh = trimesh.creation.box(extents=(2 / 3 * half,) * 3)
        mesh.apply_translation((2 / 3 * sx * half, 2 / 3 * sy * half, 2 / 3 * sz * half))
        sub_meshes.append(mesh)
    boxes = [
        scene.add_entity(
            gs.morphs.MeshSet(
                files=sub_meshes,
                pos=(i * 0.5, 0.0, 0.3),
            )
        )
        for i in range(3)
    ]
    scene.build(n_envs=n_envs)

    for _ in range(60):
        scene.step()
    for box in boxes:
        assert_allclose(box.get_pos()[..., 2], half, atol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_with_pruning(show_viewer, n_envs):
    # A convexified duck is a compound body whose many ground contacts pile into one (duck, plane) link-pair bucket,
    # so link-pair pruning collapses them into a logical permutation in contact_sort_idx. Islands read contacts through
    # that permutation and hibernation advects the resting contacts while the body sleeps, so pruning, islands and
    # hibernation all run together. contact_pruning_tolerance is set explicitly to keep pruning on alongside islands.
    # Two separated ducks give two islands (hibernation does not keep a single-island scene partitioned). Each duck
    # must reach the plane without tunnelling, hibernate, and then stay frozen in place.
    GRAVITY = -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 100.0,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            contact_pruning_tolerance=0.02,
            use_hibernation=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    ducks = [
        scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/duck.obj",
                scale=0.02,
                pos=(0.4 * i, 0.0, 0.05),
                euler=(90.0, 90.0, 0.0),
            ),
        )
        for i in range(2)
    ]
    scene.build(n_envs=n_envs)
    solver = scene.rigid_solver

    def asleep():
        return all(qd_to_numpy(solver.dyn_state.entities.is_hibernated, duck.idx).all() for duck in ducks)

    for _ in range(120):
        scene.step()
        assert all((tensor_to_array(duck.get_pos())[..., 2] > -0.05).all() for duck in ducks)
        if asleep():
            break
    assert asleep()
    z_rest = [tensor_to_array(duck.get_pos())[..., 2] for duck in ducks]

    for _ in range(100):
        scene.step()
    assert asleep()
    for duck, z in zip(ducks, z_rest):
        assert_allclose(duck.get_pos()[..., 2], z, atol=1e-5)
        # A sleeper resting on the ground keeps reporting the support force balancing its weight.
        assert_allclose(duck.get_links_net_contact_force()[..., 0, 2], -GRAVITY * duck.get_mass(), tol=2e-3)

    # Resetting wakes every body: the restored state is a discontinuity, so a body left hibernated would stay frozen
    # and never be resimulated. After reset the ducks are awake again, with their flags cleared and awake counter zeroed.
    scene.reset()
    assert not any(qd_to_numpy(solver.dyn_state.entities.is_hibernated, duck.idx).any() for duck in ducks)
    assert (qd_to_numpy(solver.dyn_state.links.awake_steps) == 0).all()


@pytest.mark.required
@pytest.mark.parametrize("mujoco_compatibility", [False, True])
def test_dof_length_scales_with_body_size(mujoco_compatibility):
    # dof_length puts each rotational dof velocity on a linear (m/s) scale by the body radius (1 for translation), so
    # the same angular velocity reads as a larger surface speed on a larger body. A free sphere gets a rotational
    # dof_length equal to its radius - both with our per-axis swept radius and with MuJoCo's COM bounding sphere
    # (gated behind mujoco_compatibility), since the two coincide for a sphere. dof_length is stored per environment,
    # so a heterogeneous entity gets a different radius per variant (each variant's geoms are active only in its own
    # envs). The two sphere variants map to envs 0-1 and 2-3; the homogeneous spheres make the scene multi-island so
    # hibernation (and thus dof_length) is active.
    radii = (0.1, 0.3)
    variant_radii = (0.02, 0.06)
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
            enable_mujoco_compatibility=mujoco_compatibility,
        ),
        show_viewer=False,
    )
    spheres = [scene.add_entity(gs.morphs.Sphere(radius=r, pos=(2.0 * i, 0.0, 1.0))) for i, r in enumerate(radii)]
    het = scene.add_entity(morph=tuple(gs.morphs.Sphere(radius=r, pos=(0.0, 2.0, 1.0)) for r in variant_radii))
    scene.build(n_envs=4)

    dof_length = qd_to_numpy(scene.rigid_solver.dyn_info.dofs.dof_length)
    for sphere, radius in zip(spheres, radii):
        dof_length_sphere = dof_length[sphere.dof_start : sphere.dof_start + sphere.n_dofs]
        assert_allclose(dof_length_sphere[:3], 1.0, tol=gs.EPS)
        assert_allclose(dof_length_sphere[3:], radius, tol=gs.EPS)
    rotational = dof_length[het.dof_start + 3 : het.dof_start + 6]  # (3, n_envs)
    assert_allclose(rotational[:, [0, 1]], variant_radii[0], tol=gs.EPS)
    assert_allclose(rotational[:, [2, 3]], variant_radii[1], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_weld_coupling(show_viewer, n_envs):
    # box2 hangs from a weld onto the anchored box1 at a horizontal offset, never touching it. Without the equality
    # edge in the partition the two land in different islands and the weld is dropped, letting box2 free-fall.
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -4.0, 2.5),
            camera_lookat=(0.15, 0.0, 0.9),
        ),
        show_viewer=show_viewer,
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 1.0),
            fixed=True,
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3, 0.0, 1.0),
        )
    )
    scene.build(n_envs=n_envs)

    scene.rigid_solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)

    z_start = box2.get_pos()[..., 2]
    for _ in range(50):
        scene.step()
    # A dropped weld would free-fall ~1 m in 1 s; the weld holds box2 near its start height.
    assert_allclose(box2.get_pos()[..., 2], z_start, tol=0.15)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_sparsity(show_viewer, n_envs):
    # On CPU the sparse Jacobian and the per-island solve exploit the same block-diagonal structure and must compose
    # (islands own the per-block factorization, the sparse jac makes products and the constraint-to-island lookup
    # O(nonzeros)). On GPU the dense tiled path wins, so sparse is dropped and islands stand alone.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            sparse_solve=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -4.0, 2.5),
            camera_lookat=(0.5, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.3),
        )
    )
    box_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.3),
        )
    )
    scene.build(n_envs=n_envs)

    for _ in range(50):
        scene.step()

    assert_allclose(box_a.get_pos()[..., 2], 0.05, atol=2e-3)
    assert_allclose(box_b.get_pos()[..., 2], 0.05, atol=2e-3)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_wakes_on_user_input(show_viewer, n_envs, tol):
    # Every user input that drives a sleeping body must wake it (and only its island) AND take effect: a hibernated
    # body's dofs are skipped by forward dynamics and integration, so the motion checks catch a body that wakes but
    # stays frozen (e.g. gravity cancelled by a neighbour's stale constraint force). Each box is separated from the
    # others, hence an island of its own, so each input wakes exactly one.
    G = 9.8
    DT = 1.0 / 60.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -G),
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -4.0, 2.5),
            camera_lookat=(3.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_force = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.1),
        )
    )
    box_pos = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.1),
        )
    )
    box_vel = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.1),
        )
    )
    box_qpos = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(3.0, 0.0, 0.1),
        )
    )
    box_cforce = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(4.0, 0.0, 0.1),
        )
    )
    box_cvel = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(5.0, 0.0, 0.1),
        )
    )
    box_cpos = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(6.0, 0.0, 0.1),
        )
    )
    box_mass = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(7.0, 0.0, 0.1),
        )
    )
    box_armature = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(8.0, 0.0, 0.1),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    def asleep(entity):
        return qd_to_numpy(solver.dyn_state.entities.is_hibernated, entity.idx).all()

    def z_of(entity):
        return entity.get_pos()[..., 2]

    # Velocity/position control need PD gains to produce a force; index 2 of each free joint is the world-z dof.
    for box in (box_cvel, box_cpos):
        box.set_dofs_kp([0.0, 0.0, 400.0, 0.0, 0.0, 0.0])
        box.set_dofs_kv([40.0, 40.0, 40.0, 4.0, 4.0, 4.0])

    n_fall = 8
    free_fall_drop = 0.5 * G * (n_fall * DT) ** 2

    for _ in range(90):
        scene.step()
    assert all(
        map(asleep, (box_force, box_pos, box_vel, box_qpos, box_cforce, box_cvel, box_cpos, box_mass, box_armature))
    )

    z0 = z_of(box_force)
    for _ in range(6):
        solver.apply_links_external_wrench([0.0, 0.0, 40.0], links_idx=box_force.base_link_idx)
        scene.step()
    assert not asleep(box_force) and (z_of(box_force) > z0 + 0.02).all()
    assert all(map(asleep, (box_pos, box_vel, box_qpos, box_cforce, box_cvel, box_cpos)))

    box_pos.set_dofs_position([1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    assert not asleep(box_pos)
    z0 = z_of(box_pos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z0 - z_of(box_pos), free_fall_drop, rtol=0.2)
    assert all(map(asleep, (box_vel, box_qpos, box_cforce, box_cvel, box_cpos)))

    box_vel.set_dofs_velocity([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert not asleep(box_vel)
    z0 = z_of(box_vel)
    for _ in range(5):
        scene.step()
    assert (z_of(box_vel) > z0 + 0.05).all()
    assert all(map(asleep, (box_qpos, box_cforce, box_cvel, box_cpos)))

    box_qpos.set_qpos([3.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0])
    assert not asleep(box_qpos)
    z0 = z_of(box_qpos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z0 - z_of(box_qpos), free_fall_drop, rtol=0.2)
    assert all(map(asleep, (box_cforce, box_cvel, box_cpos)))

    z0 = z_of(box_cforce)
    for _ in range(8):
        box_cforce.control_dofs_force([0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cforce) and (z_of(box_cforce) > z0 + 0.02).all()
    assert all(map(asleep, (box_cvel, box_cpos)))

    z0 = z_of(box_cvel)
    for _ in range(8):
        box_cvel.control_dofs_velocity([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cvel) and (z_of(box_cvel) > z0 + 0.02).all()
    assert asleep(box_cpos)

    z0 = z_of(box_cpos)
    for _ in range(12):
        box_cpos.control_dofs_position([6.0, 0.0, 0.6, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cpos) and (z_of(box_cpos) > z0 + 0.05).all()

    # A mass or an armature written moves the equilibrium a resting body found, so a body it is written on must wake
    # to settle into the new one. Both writes are made while the body sleeps, and the weight they land on is the
    # analytic one of a free body.
    MASS = 4.0
    ARMATURE = 0.5
    invweight_before = solver.get_links_invweight(box_mass.base_link_idx)
    box_mass.set_mass(MASS)
    assert not asleep(box_mass)
    assert_allclose(solver.get_links_invweight(box_mass.base_link_idx)[..., 0], 1.0 / MASS, tol=gs.EPS)
    assert ((invweight_before - solver.get_links_invweight(box_mass.base_link_idx)).abs() > 1e-3).all()

    # The mean inertia the constraint solve is quoted on is the mean of the mass-matrix diagonal, so a mass written on
    # a woken tree is in it; a tree left asleep would hold its contribution at whatever it last ran with.
    mass_mat = solver.get_mass_mat()
    diagonal = mass_mat.diagonal(dim1=-2, dim2=-1) if mass_mat.ndim > 2 else mass_mat.diagonal()
    assert_allclose(qd_to_numpy(solver.rigid_info.meaninertia), tensor_to_array(diagonal.mean(dim=-1)), tol=tol)

    invweight_before = box_armature.get_dofs_invweight()
    box_armature.set_dofs_armature([ARMATURE] * 6)
    assert not asleep(box_armature)
    # An armature adds to the mass matrix without belonging to any link, so a translational degree of freedom of a
    # free body weighs the mass it carries plus the armature on it.
    mass_armature = float(box_armature.get_mass()) + ARMATURE
    assert_allclose(box_armature.get_dofs_invweight()[..., :3], 1.0 / mass_armature, tol=tol)
    assert ((invweight_before - box_armature.get_dofs_invweight()).abs() > 1e-3).all()


@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("broadphase_traversal", [None, gs.broadphase_traversal.ALL_VS_ALL], ids=["sap", "allvsall"])
def test_hibernation_wakes_on_collision(show_viewer, n_envs, broadphase_traversal, multi_free_body_path):
    # An awake body striking a sleeping one must wake it so it responds instead of acting as an immovable obstacle.
    # This needs the broad-phase sort-buffer refresh of awake geoms (so the contact is detected) and the wake-on-contact
    # pass. The multi-free-body entity (offset clear of the boxes) checks per-component hibernation: its free bodies
    # sleep independently and disturbing one wakes only its island, which needs the wake/daisy chain to act per link
    # rather than per entity. ALL_VS_ALL exercises hibernation under the non-default traversal: it advects and skips
    # hibernated-fixed pairs exactly like SAP, and the hibernated-vs-hibernated pairs it traverses instead of skipping
    # are inert (both bodies frozen).
    GRAVITY = -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
            broadphase_traversal=broadphase_traversal,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.1, -4.0, 2.5),
            camera_lookat=(0.1, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_rest = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_hit = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.22, 0.0, 0.05),
        )
    )
    multibody = scene.add_entity(
        gs.morphs.MJCF(
            file=multi_free_body_path,
            pos=(0.0, 5.0, 0.0),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    def asleep(entity):
        return qd_to_numpy(solver.dyn_state.entities.is_hibernated, entity.idx).all()

    def link_asleep(link):
        return qd_to_numpy(solver.dyn_state.links.is_hibernated, link.idx).all()

    # One free body of the entity is dropped from higher up, so the trees of one entity land, settle and fall asleep
    # at different moments: what sleeps is the tree, not the entity that holds it.
    multibody_bases = [link for link in multibody.links if link.parent_idx == -1 and link.n_dofs > 0]
    late = multibody_bases[-1]
    lifted = solver.get_links_pos(late.idx)
    lifted[..., 2] += 0.6
    solver.set_base_links_pos(lifted, links_idx=late.idx)

    for _ in range(60):
        scene.step()
    assert asleep(box_rest) and asleep(box_hit)
    # The bodies that landed first sleep while the one still falling does not.
    assert all(link_asleep(link) for link in multibody_bases[:-1])
    assert not link_asleep(late)
    # The sleepers stay frozen while the env keeps solving the body still falling: zero velocity, and forces held at
    # their values from the moment they fell asleep, since their island is left out of the solve. A sleeper resting on
    # the ground keeps reporting the support force balancing its weight.
    sleepers_force = [box.get_dofs_force() for box in (box_rest, box_hit)]
    sleepers_contact_force = [box.get_links_net_contact_force() for box in (box_rest, box_hit)]
    for _ in range(40):
        scene.step()
    for box, force, contact_force in zip((box_rest, box_hit), sleepers_force, sleepers_contact_force):
        assert_equal(box.get_dofs_velocity(), 0.0)
        assert_equal(box.get_dofs_force(), force)
        assert_equal(box.get_links_net_contact_force(), contact_force)
        assert_allclose(contact_force[..., 0, 2], -GRAVITY * box.get_mass(), tol=2e-3)
    rest_x0 = box_rest.get_pos()[..., 0]

    box_hit.set_dofs_velocity([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        scene.step()

    # The struck sleeper woke and was knocked; the striker was stopped by it (did not tunnel through).
    assert not asleep(box_rest)
    rest_x1 = box_rest.get_pos()[..., 0]
    hit_x1 = box_hit.get_pos()[..., 0]
    assert (rest_x1 < rest_x0 - 1e-3).all()
    assert (hit_x1 > rest_x1).all()

    # The undisturbed entity's free bodies all settled and slept independently; disturbing one wakes only its island.
    assert all(link_asleep(link) for link in multibody_bases)
    disturbed = multibody_bases[0]
    solver.set_dofs_velocity(
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        list(range(disturbed.dof_start, disturbed.dof_start + disturbed.n_dofs)),
    )
    assert not link_asleep(disturbed)
    assert all(link_asleep(link) for link in multibody_bases[1:])

    # A sleeping tree is left out of the mass solve rather than solved and thrown away, which is what makes the cost
    # follow what is awake. The solve is the only pass writing the smooth accelerations, so a sentinel written into
    # them comes back untouched for a sleeping tree of the entity and rewritten for its woken sibling.
    SENTINEL = -13.0
    sentinels = qd_to_numpy(solver.dyn_state.dofs.acc_smooth)
    sentinels[disturbed.dof_start : disturbed.dof_end] = SENTINEL
    for link in multibody_bases[1:]:
        sentinels[link.dof_start : link.dof_end] = SENTINEL
    solver.dyn_state.dofs.acc_smooth.from_numpy(sentinels)
    scene.step()
    solved = qd_to_numpy(solver.dyn_state.dofs.acc_smooth)
    assert (solved[disturbed.dof_start : disturbed.dof_end] != SENTINEL).any()
    for link in multibody_bases[1:]:
        assert_allclose(solved[link.dof_start : link.dof_end], SENTINEL, tol=gs.EPS)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_wakes_on_daisy_chain(show_viewer, n_envs):
    # Two welded bodies sleep as ONE island. Disturbing only box_a must wake the WHOLE island via the daisy chain, else
    # its coupled partner stays frozen and the weld is solved against a sleeping body. A weld is used (not a contact
    # stack, whose micro-settling keeps it awake); a separated third box is its own island and stays asleep.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3, 0.0, 0.05),
        )
    )
    box_far = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.05),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    solver.add_weld_constraint(box_a.base_link_idx, box_b.base_link_idx)

    def asleep(entity):
        return qd_to_numpy(solver.dyn_state.entities.is_hibernated, entity.idx).all()

    for _ in range(50):
        scene.step()
    assert asleep(box_a) and asleep(box_b) and asleep(box_far)

    solver.apply_links_external_wrench([20.0, 0.0, 0.0], links_idx=box_a.base_link_idx)
    scene.step()
    assert not asleep(box_a)
    assert not asleep(box_b)
    assert asleep(box_far)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_repartitioning(show_viewer, n_envs):
    # Full lifecycle of hibernation and the partition together: two boxes sleep apart (2 islands); moving one onto the
    # other wakes it, it collides, and the stack sleeps as one merged island; moving a box off the hibernated stack
    # must wake the WHOLE merged island (else the stale daisy chain keeps re-connecting both); they then split back.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.3, 0.0, 0.15),
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3, 0.0, 0.15),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.sim.rigid_solver
    box1_idx = box1._idx_in_solver
    box2_idx = box2._idx_in_solver
    island_state = solver.constraint_solver.constraint_state.island

    def asleep(idx):
        return qd_to_numpy(solver.dyn_state.entities.is_hibernated, idx).all()

    def awake(idx):
        return not qd_to_numpy(solver.dyn_state.entities.is_hibernated, idx).any()

    for _ in range(60):
        scene.step()
        if asleep(box1_idx) and asleep(box2_idx):
            break
    assert asleep(box1_idx)
    assert asleep(box2_idx)
    assert_equal(qd_to_numpy(island_state.n_islands), 2)

    box2_pos = tensor_to_array(box2.get_pos())
    box1_target = box2_pos.copy()
    box1_target[..., 0] += 0.01
    box1_target[..., 1] += 0.01
    box1_target[..., 2] = 0.3
    box1.set_pos(box1_target)
    assert awake(box1_idx)
    assert (box1.get_pos()[..., 2] > 0.2).all()

    for _ in range(30):
        scene.step()
    assert awake(box1_idx)
    assert awake(box2_idx)

    for _ in range(60):
        scene.step()
        if asleep(box1_idx) and asleep(box2_idx):
            break
    assert asleep(box1_idx)
    assert asleep(box2_idx)
    assert_equal(qd_to_numpy(island_state.n_islands), 1)

    box1.set_pos([1.0, 0.0, 0.15])
    assert awake(box1_idx)
    assert awake(box2_idx)

    for _ in range(120):
        scene.step()
        if asleep(box1_idx) and asleep(box2_idx):
            break
    assert asleep(box1_idx)
    assert asleep(box2_idx)
    assert_equal(qd_to_numpy(island_state.n_islands), 2)
