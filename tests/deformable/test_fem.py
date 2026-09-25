import math

import igl
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal
from ..utils.assets import get_hf_dataset


@pytest.mark.required
def test_interior_tetrahedralized_vertex(cube_verts_and_faces, box_obj_path, show_viewer):
    # A small maxvolume introduces internal vertices during tetrahedralization: all surface vertices must
    # still lie exactly on the original quad faces, and the visualizer's mesh vertices must track the
    # simulated vertices through the visual geom's sim_verts_idx.
    verts, faces = cube_verts_and_faces

    scene = gs.Scene(show_viewer=show_viewer)
    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            minratio=1.5,
            verbose=1,
            maxvolume=0.01,
        ),
        material=gs.materials.FEM.Muscle(),
    )
    scene.build()

    state = fem.get_state()
    vertices = tensor_to_array(state.pos[0])
    surface_indices = np.unique(fem.surface_triangles)

    # Ensure there are interior vertices; this is a prerequisite for this test
    assert surface_indices.size < vertices.shape[0]

    # Verify each surface vertex lies on the original surface mesh
    def _point_on_surface(p, verts, faces, tol=1e-6):
        """Check if point p lies on any of the quad faces (as two triangles)."""
        for face in faces:
            # Convert 1-based face indices to 0-based
            idx = [i - 1 for i in face]
            # Extract vertices
            v0, v1, v2, v3 = [np.array(verts[i]) for i in idx]
            # Decompose quad into two triangles: (v0,v1,v2) and (v0,v2,v3)
            for tri in ((v0, v1, v2), (v0, v2, v3)):
                a, b, c = tri
                # Compute normal for plane
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < tol:
                    continue
                # Check distance to plane
                distance = abs(np.dot(n / norm_n, p - a))
                if distance > tol:
                    continue
                # Barycentric coordinates
                v0v1 = b - a
                v0v2 = c - a
                v0p = p - a
                dot00 = np.dot(v0v2, v0v2)
                dot01 = np.dot(v0v2, v0v1)
                dot02 = np.dot(v0v2, v0p)
                dot11 = np.dot(v0v1, v0v1)
                dot12 = np.dot(v0v1, v0p)
                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < tol:
                    continue
                u = (dot11 * dot02 - dot01 * dot12) / denom
                v = (dot00 * dot12 - dot01 * dot02) / denom
                if u >= -tol and v >= -tol and (u + v) <= 1 + tol:
                    return True
        return False

    for idx in surface_indices:
        p = vertices[idx]
        assert _point_on_surface(p, verts, faces), (
            f"Surface vertex index {idx} with coordinate {p} does not lie on any original face"
        )

    # The visualizer draws the input surface, whose vertices track the simulated vertices through the visual
    # geom's sim_verts_idx, while surface_triangles reflects the refined tetrahedral boundary: the two meshes
    # differ in granularity, so the invariant to check is per-vertex tracking.
    (vgeom,) = fem.vgeoms
    (fem_node_primitive,) = scene.visualizer.context.static_nodes[(0, vgeom.uid)].mesh.primitives
    viz_verts = fem_node_primitive.positions
    assert_allclose(viz_verts, vertices[vgeom.sim_verts_idx], tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_multi_submesh_render_decoupling(n_envs, show_viewer):
    # Cloth material keeps the entity out of the tetrahedralization path, exercising the welded surface as
    # simulation elements. The GLB asset holds 2 sub-meshes (bag and channel) with distinct materials.
    scene = gs.Scene(show_viewer=show_viewer)
    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/trashbag_rope.glb",
            group_by_material=True,
        ),
        material=gs.materials.FEM.Cloth(),
    )
    scene.build(n_envs=n_envs)

    assert len(fem.vgeoms) == 2
    # The two sub-meshes share their boundary vertices, so welding strictly reduces the vertex count.
    n_render_verts = sum(len(vgeom.vmesh.verts) for vgeom in fem.vgeoms)
    assert fem.n_vertices < n_render_verts
    vertices = tensor_to_array(fem.get_state().pos)
    for vgeom in fem.vgeoms:
        # Negative indices would wrap around silently.
        assert vgeom.sim_verts_idx.min() >= 0
        # Mapped simulated vertices must land on the render vertices they stand for, in every environment, up to
        # the quantization of the float64 authored vertices to the simulation dtype.
        assert_allclose(vertices[..., vgeom.sim_verts_idx, :], vgeom.vmesh.verts, tol=gs.EPS)

    # The rasterizer must register one node per visual geom and environment.
    static_nodes = scene.visualizer.context.static_nodes
    assert all((i_b, vgeom.uid) in static_nodes for i_b in range(max(n_envs, 1)) for vgeom in fem.vgeoms)


@pytest.mark.required
def test_maxvolume(box_obj_path, show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)

    # Mesh without any maximum-element-volume constraint
    fem1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            verbose=1,
        ),
        material=gs.materials.FEM.Muscle(),
    )

    # Mesh with maximum element volume limited to 0.01
    fem2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            maxvolume=0.01,
            verbose=1,
        ),
        material=gs.materials.FEM.Muscle(),
    )

    assert len(fem1.elems) < len(fem2.elems), (
        f"Mesh with maxvolume=0.01 generated {len(fem2.elems)} elements; "
        f"expected more than {len(fem1.elems)} elements without a volume limit."
    )


@pytest.mark.required
def test_offset_pos(box_obj_path, show_viewer):
    POS = (0.2, -0.1, 0.3)
    OFFSET_POS = (0.05, 0.0, 0.1)

    scene = gs.Scene(show_viewer=show_viewer)
    box = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            pos=POS,
            offset_pos=OFFSET_POS,
        ),
        material=gs.materials.FEM.Elastic(),
    )
    box_no_offset = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            pos=POS,
        ),
        material=gs.materials.FEM.Elastic(),
    )
    scene.build()

    offset = box.get_state().pos - box_no_offset.get_state().pos
    assert_allclose(offset, OFFSET_POS, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize(
    "coupler_type, material_model",
    [
        (gs.options.SAPCouplerOptions, "linear"),
        (gs.options.SAPCouplerOptions, "linear_corotated"),
        (gs.options.LegacyCouplerOptions, "linear"),
    ],
)
def test_implicit_falling_sphere_box(coupler_type, material_model, show_viewer):
    SPHERE_POS = (0.4, -0.1, 0.1)
    SPHERE_RADIUS = 0.1
    SPHERE_RHO = 500.0
    BOX_POS = (0.0, 0.1, 0.3)
    BOX_SIZE = 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=3 if coupler_type == gs.options.SAPCouplerOptions else 2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=coupler_type(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.0, 0.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.4, -0.1, 0.10),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            rho=SPHERE_RHO,
            model=material_model,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BOX_SIZE * 2, BOX_SIZE * 2, BOX_SIZE * 2),
            pos=BOX_POS,
        ),
        material=gs.materials.FEM.Elastic(),
    )

    # Build the scene
    scene.build()

    sphere_mass_ref = (4 / 3 * np.pi * SPHERE_RADIUS**3) * SPHERE_RHO
    elems_mass_scaled = scene.fem_solver.elements_i.mass_scaled.to_numpy()
    sphere_elems_mass_scaled = elems_mass_scaled[sphere.el_start : (sphere.el_start + sphere.n_elements)]
    sphere_mass_1 = sphere_elems_mass_scaled.sum() / scene.fem_solver.vol_scale
    verts_mass = scene.fem_solver.elements_v_info.mass.to_numpy()
    sphere_verts_mass = verts_mass[sphere.v_start : (sphere.v_start + sphere.n_vertices)]
    sphere_mass_2 = sphere_verts_mass.sum()
    assert_allclose(sphere_mass_1, sphere_mass_2, tol=gs.EPS)
    assert_allclose(sphere_mass_ref, sphere_mass_1, rtol=0.01)  # Large tolerance due to tessellation

    # Run simulation
    n_steps = 40 if coupler_type == gs.options.SAPCouplerOptions else 150
    for _ in range(n_steps):
        scene.step()

    for entity, init_pos, entity_halfsize_ref in zip(scene.entities, (SPHERE_POS, BOX_POS), (SPHERE_RADIUS, BOX_SIZE)):
        # Not moving anymore
        state = entity.get_state()
        assert_allclose(state.vel, 0.0, tol=0.025)

        # Landed vertically
        pos = tensor_to_array(state.pos[0])
        BV, *_ = igl.bounding_box(pos)
        entity_center = 0.5 * (BV[0] + BV[-1])
        if coupler_type == gs.options.SAPCouplerOptions:
            tol = 1e-2 if material_model == "linear_corotated" else 1e-3
        else:
            tol = 5e-3
        assert_allclose(entity_center[:2], init_pos[:2], tol=tol)

        # Reasonable deformation if possible
        # FIXME: Compute theoretical deformation to be able to reduce the tolerance
        if coupler_type == gs.options.SAPCouplerOptions:
            entity_halfsize = 0.5 * (BV[0] - BV[-1])
            assert_allclose(entity_halfsize, entity_halfsize_ref, tol=5e-3)

        # Reasonable penetration depth
        # FIXME: Compute theoretical penetration depth to be able to reduce the tolerance
        penetration_depth_ref = 0.0
        tol = 1e-3 if coupler_type == gs.options.SAPCouplerOptions else 0.05
        assert_allclose(-state.pos[..., 2].min(), penetration_depth_ref, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("use_implicit_solver", [False, True])
@pytest.mark.parametrize("precision", ["64"])
def test_hard_constraint(use_implicit_solver, show_viewer):
    DT = 0.01
    HEIGHT = 2.0  # It must be height enough to avoid hitting the ground when dropping the box at the end
    BOX_SIZE = 0.1
    MOTION_RADIUS = 0.1
    MOTION_SPEED = 0.005
    VERTICES_IDX = [5, 7]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2 if use_implicit_solver else 100,
            gravity=(0.0, 0.0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=use_implicit_solver,
            enable_vertex_constraints=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, HEIGHT + 0.5),
            camera_lookat=(0.0, 0.0, HEIGHT),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    # Place the tested entity after another FEM entity so its local and solver indices differ.
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.5, 0.0, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated" if use_implicit_solver else "stable_neohookean",
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-BOX_SIZE / 2, BOX_SIZE / 2 + MOTION_RADIUS, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated" if use_implicit_solver else "stable_neohookean",
        ),
    )
    anchor = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(10.0, 0.0, HEIGHT),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    scene.build(n_envs=2)

    assert_equal(box.get_el2v(), box.elems)

    target_poss = box.get_state().pos[..., VERTICES_IDX, :]
    box.set_vertex_constraints(VERTICES_IDX)
    scene.step()
    assert_allclose(box.get_state().pos[..., VERTICES_IDX, :], target_poss, tol=1e-8)

    # Simulate
    n_steps = int(0.5 * math.pi / MOTION_SPEED)
    for it in range(n_steps):
        # Update the position of the fixed vertices
        target_poss = torch.zeros((2, 3), dtype=gs.tc_float, device=gs.device)
        target_poss[0, 0] = MOTION_RADIUS * math.sin(MOTION_SPEED * it)
        target_poss[0, 1] = MOTION_RADIUS * math.cos(MOTION_SPEED * it)
        target_poss[0, 2] = HEIGHT + BOX_SIZE
        target_poss[1, 0] = (MOTION_RADIUS + BOX_SIZE) * math.sin(MOTION_SPEED * it)
        target_poss[1, 1] = (MOTION_RADIUS + BOX_SIZE) * math.cos(MOTION_SPEED * it)
        target_poss[1, 2] = HEIGHT + BOX_SIZE
        box.set_vertex_constraints(VERTICES_IDX, target_poss)

        # Do one simulation step
        scene.step(update_visualizer=False)

        # Check that the constrained vertices are at their respective target positions
        corners = box.get_state().pos[..., :8, :]
        assert_allclose(corners[..., VERTICES_IDX, :], target_poss, tol=1e-8)

        # Check that the box is not more or less a box
        e_z = corners[..., 5, :] - corners[..., 4, :]
        e_z /= torch.linalg.norm(e_z, dim=-1, keepdim=True)
        e_y = corners[..., 6, :] - corners[..., 4, :]
        e_y /= torch.linalg.norm(e_y, dim=-1, keepdim=True)
        e_x = torch.cross(e_y, e_z, dim=-1)
        e_x /= torch.linalg.norm(e_x, dim=-1, keepdim=True)
        R = torch.stack((e_x, e_y, e_z), dim=-1)
        corners_local = (corners - corners[..., [0], :]) @ R + box.init_positions[0]
        assert_allclose(corners_local, box.init_positions[:8], tol=0.01)

        # Update the viewer if requested
        if show_viewer:
            if it % max(int(1e-3 / (MOTION_SPEED * DT)), 1) == 0:
                scene.visualizer.context.draw_debug_spheres(
                    poss=target_poss, radius=0.005, color=(1, 0, 1, 0.8), persistent=True
                )
            scene.visualizer.update(force=False, auto=True)

    # Disable constraints
    box.remove_vertex_constraints()

    # Check that the box has been free-falling
    n_steps = 50
    com_pos_z_0 = box.get_state().pos[..., 8, 2]
    for _ in range(n_steps):
        scene.step()
    com_pos_z_f = box.get_state().pos[..., 8, 2]
    com_pos_delta = -0.5 * 9.81 * (n_steps * DT) ** 2
    assert_allclose(com_pos_z_f - com_pos_z_0, com_pos_delta, tol=0.05)

    # Distinct positions and link poses per environment make cross-environment index mixups observable.
    box.set_position(((0.0, 0.0, 0.0), (0.0, 0.0, 0.2)))
    box.set_velocity((0.0, 0.0, 0.0))
    anchor.set_pos(((10.0, 0.0, HEIGHT), (20.0, 0.0, HEIGHT)), relative=False)
    anchor.set_quat(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), relative=False)
    target_poss = box.get_state().pos[..., VERTICES_IDX, :]
    box.set_vertex_constraints(VERTICES_IDX, envs_idx=[1])
    scene.step()
    assert_allclose(box.get_state().pos[1, VERTICES_IDX, :], target_poss[1], tol=1e-8)

    box.remove_vertex_constraints(envs_idx=[1])
    box.set_vertex_constraints(VERTICES_IDX, link=anchor.base_link, envs_idx=[1])
    scene.step()
    assert_allclose(box.get_state().pos[1, VERTICES_IDX, :], target_poss[1], tol=1e-8)

    box.remove_vertex_constraints(envs_idx=[1])
    box.set_position((0.0, 0.0, 0.0))
    box.set_velocity((0.0, 0.0, 0.0))
    target_poss = box.get_state().pos[..., VERTICES_IDX, :]
    box.set_vertex_constraints(VERTICES_IDX)
    target_poss = target_poss + torch.tensor(
        (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.02),)), dtype=gs.tc_float, device=gs.device
    )
    box.update_constraint_targets(VERTICES_IDX, target_poss[1], envs_idx=[1])
    scene.step()
    assert_allclose(box.get_state().pos[..., VERTICES_IDX, :], target_poss, tol=1e-6)

    box.remove_vertex_constraints(envs_idx=[1])
    target_poss = target_poss + torch.tensor((0.0, 0.0, 0.05), dtype=gs.tc_float, device=gs.device)
    box.update_constraint_targets(VERTICES_IDX, target_poss)
    scene.step()
    constrained_poss = box.get_state().pos[..., VERTICES_IDX, :]
    assert_allclose(constrained_poss[0], target_poss[0], tol=1e-6)
    assert (torch.linalg.norm(constrained_poss[1] - target_poss[1], dim=-1) > 0.01).all()


@pytest.mark.required
@pytest.mark.xfail(raises=AssertionError, reason="Constraint dynamics inconsistent with analytical formula")
@pytest.mark.parametrize("precision", ["64"])
def test_explicit_legacy_coupler_soft_constraint_box(show_viewer):
    DT = 0.01
    BOX_SIZE = 0.1
    CONSTRAINT_STIFFNESS = 1e1
    BOX_VELOCITY = torch.tensor([0.2, 0.0, 0.0])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=False,
            enable_vertex_constraints=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.FEM.Elastic(
            rho=1.0 / BOX_SIZE**3,  # Unit mass
        ),
        surface=gs.surfaces.Default(
            color=(1, 1, 1, 0.5),
        ),
    )
    scene.build()

    verts_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    target_poss = box.init_positions[verts_idx]
    box.set_vertex_constraints(verts_idx, target_poss, is_soft_constraint=True, stiffness=CONSTRAINT_STIFFNESS)
    if show_viewer:
        scene.draw_debug_spheres(poss=target_poss, radius=0.01, color=(1, 0, 1, 1))

    # Initialize box velocity to non-zero value
    box.set_velocity(BOX_VELOCITY)

    # Check that the box has a spring dynamics
    omega = math.sqrt(8 * CONSTRAINT_STIFFNESS)
    for i in range(2000):
        pos = box.get_state().pos[..., :8, :].sum(dim=-2)
        pos_ref = BOX_VELOCITY * (i * DT) * math.exp(-omega * (i * DT))
        assert_allclose(pos, pos_ref, tol=1e-3)
        scene.step()


# This test cannot be flagged as required because it takes 250s to run on CPU.
# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_implicit_sap_coupler_collide_sphere_box(show_viewer):
    SPHERE_RADIUS = 0.1
    BOX_SIZE = 0.015

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.45),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, SPHERE_RADIUS),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            friction_mu=1.0,
            model="linear_corotated",
        ),
    )
    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    box = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            pos=(0.0, 0.0, 2 * SPHERE_RADIUS + BOX_SIZE),
            scale=BOX_SIZE,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e4,
            rho=50.0,
            friction_mu=1.0,
            model="linear_corotated",
        ),
    )
    scene.build()

    # Run simulation
    for _ in range(40):
        scene.step()

    for entity, init_height in zip(scene.entities, (SPHERE_RADIUS, 2 * SPHERE_RADIUS + BOX_SIZE)):
        # Barely moving
        state = entity.get_state()
        assert_allclose(state.vel, 0.0, tol=0.05)

        # More or less at the initial position
        pos = tensor_to_array(state.pos[0])
        BV, *_ = igl.bounding_box(pos)
        entity_center = 0.5 * (BV[0] + BV[-1])
        assert_allclose(entity_center[:2], 0.0, tol=0.02)
        assert_allclose(entity_center[2], init_height, tol=5e-3)


# This test cannot be flagged as required because it takes 400s to run on CPU.
# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_implicit_sap_coupler_hard_constraint_and_collision(show_viewer):
    DT = 0.01
    HEIGHT = 2.0  # It must be height enough to avoid hitting the ground when dropping the box at the end
    BOX_SIZE = 0.1
    SPHERE_RADIUS = 0.03
    MOTION_RADIUS = 0.1
    MOTION_SPEED = 0.005
    VERTICES_IDX = [4, 6]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, HEIGHT + 0.5),
            camera_lookat=(0.0, 0.0, HEIGHT),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-BOX_SIZE / 2, BOX_SIZE / 2 + MOTION_RADIUS, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated",
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(MOTION_RADIUS + BOX_SIZE, -SPHERE_RADIUS, HEIGHT - BOX_SIZE / 2),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
        ),
    )
    scene.build()

    # Attach the sphere to its center
    sphere_poss = sphere.get_state().pos[0]
    sphere_poss -= torch.tensor(sphere.morph.pos)
    sphere_center_idx = int(torch.argmin(torch.linalg.norm(sphere_poss, dim=-1)))
    sphere_target_poss = sphere.init_positions[sphere_center_idx]
    sphere.set_vertex_constraints(sphere_center_idx, sphere_target_poss)

    # Simulate
    n_steps = int(0.5 * math.pi / MOTION_SPEED)
    box_pos_y_min = torch.tensor(float("inf"), dtype=gs.tc_float, device=gs.device)
    for it in range(n_steps):
        # Update the position of the fixed vertices
        target_poss = torch.zeros((2, 3), dtype=gs.tc_float, device=gs.device)
        target_poss[0, 0] = MOTION_RADIUS * math.sin(MOTION_SPEED * it)
        target_poss[0, 1] = MOTION_RADIUS * math.cos(MOTION_SPEED * it)
        target_poss[0, 2] = HEIGHT
        target_poss[1, 0] = (MOTION_RADIUS + BOX_SIZE) * math.sin(MOTION_SPEED * it)
        target_poss[1, 1] = (MOTION_RADIUS + BOX_SIZE) * math.cos(MOTION_SPEED * it)
        target_poss[1, 2] = HEIGHT
        box.set_vertex_constraints(VERTICES_IDX, target_poss)

        # Do one simulation step
        scene.step(update_visualizer=False)

        # Check that the constrained vertices are at their respective target positions
        corners = box.get_state().pos[..., :8, :]
        assert_allclose(corners[..., VERTICES_IDX, :], target_poss, tol=1e-8)
        sphere_center = sphere.get_state().pos[..., sphere_center_idx, :]
        assert_allclose(sphere_center, sphere_target_poss, tol=1e-8)

        # Check that the box is more or less a box
        e_z = corners[..., 5, :] - corners[..., 4, :]
        e_z /= torch.linalg.norm(e_z, dim=-1, keepdim=True)
        e_y = corners[..., 6, :] - corners[..., 4, :]
        e_y /= torch.linalg.norm(e_y, dim=-1, keepdim=True)
        e_x = torch.cross(e_y, e_z, dim=-1)
        e_x /= torch.linalg.norm(e_x, dim=-1, keepdim=True)
        R = torch.stack((e_x, e_y, e_z), dim=-1)
        corners_local = (corners - corners[..., [0], :]) @ R + box.init_positions[0]
        assert_allclose(corners_local, box.init_positions[:8], tol=0.02)

        # Check that the sphere is more or less a sphere
        sphere_poss = sphere.get_state().pos
        sphere_poss -= torch.tensor(sphere.morph.pos)
        sphere_dist_max = torch.linalg.norm(sphere_poss, dim=-1).max(dim=-1).values
        assert_allclose(sphere_dist_max, SPHERE_RADIUS, tol=0.01)

        # Check that the box is not going to far along y-axis due to collision with the sphere
        box_pos_y_min = torch.minimum(corners[..., [2, 3, 7], 1].min(dim=-1).values, box_pos_y_min)
        assert (box_pos_y_min > -0.05).all()

        # Update the viewer if requested
        if show_viewer:
            if it % max(int(1e-3 / (MOTION_SPEED * DT)), 1) == 0:
                scene.visualizer.context.draw_debug_spheres(
                    poss=target_poss, radius=0.005, color=(1, 0, 1, 0.8), persistent=True
                )
            scene.visualizer.update(force=False, auto=True)

    # Wait for a few extra steps
    for _ in range(10):
        scene.step()

    # Disable box constraints only
    box.remove_vertex_constraints(VERTICES_IDX)

    # Simulate for a while
    n_steps = 40
    com_pos_z_0 = box.get_state().pos[..., 8, 2]
    for _ in range(n_steps):
        # Do one simulation step
        scene.step()

        # Check that the box is not moving much further along y-axis despite removing constraints
        box_pos = box.get_state().pos
        assert (box_pos[..., 1].min(dim=-1).values > box_pos_y_min - 0.01).all()

    # Check that the box has been free-falling, but not the sphere
    com_pos_z_f = box_pos[..., 8, 2]
    com_pos_delta = -0.5 * 9.81 * (n_steps * DT) ** 2
    assert_allclose(com_pos_z_f - com_pos_z_0, com_pos_delta, tol=0.05)
    sphere_center = sphere.get_state().pos[..., sphere_center_idx, :]
    assert_allclose(sphere_center, sphere_target_poss, tol=1e-8)
