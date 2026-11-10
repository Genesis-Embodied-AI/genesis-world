import math

import numpy as np
import pytest
import torch

import igl

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

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

    vertices = tensor_to_array(fem.get_verts_pos())
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
    vertices = tensor_to_array(fem.get_verts_pos())
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

    offset = box.get_verts_pos() - box_no_offset.get_verts_pos()
    assert_allclose(offset, OFFSET_POS, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
def test_halt_on_nan(show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.3),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    scene.build(n_envs=2)

    # The explicit integration of a stiff sphere at the default time step diverges until its positions overflow, and
    # every env is flagged at the very step its positions stop being finite
    for _ in range(10):
        scene.step()
        is_finite = sphere.get_verts_pos().isfinite().all(dim=(-2, -1))
        assert_equal(scene.get_error_envs_mask(), ~is_finite)
        if not is_finite.any():
            break
    assert not is_finite.any()

    # The error raises at the beginning of the next step on CPU, and at worst of the one after it on GPU
    with pytest.raises(gs.GenesisException, match="NaN detected in FEM states"):
        for _ in range(1 if gs.backend == gs.cpu else 2):
            scene.step()

    # Writing some of the vertices leaves the error flagged, since the other vertices stay non-finite
    sphere.set_verts_pos(sphere.init_positions[:1], verts_idx_local=[0])
    assert scene.get_error_envs_mask().all()

    # Resetting the scene clears the error, so the simulation proceeds
    scene.reset()
    scene.step()
    assert sphere.get_verts_pos().isfinite().all()
    assert not scene.get_error_envs_mask().any()


@pytest.mark.required
@pytest.mark.parametrize(
    "coupler_type, material_model, fem_floor_contact_type",
    [
        # The SAP coupler supports 64 bits precision alone
        pytest.param(gs.options.SAPCouplerOptions, "linear", "tet", marks=pytest.mark.precision("64")),
        pytest.param(gs.options.SAPCouplerOptions, "linear_corotated", "vert", marks=pytest.mark.precision("64")),
        (gs.options.LegacyCouplerOptions, "linear", None),
    ],
)
def test_implicit_falling_sphere_box(coupler_type, material_model, fem_floor_contact_type, show_viewer):
    SPHERE_POS = (0.4, -0.1, 0.1)
    SPHERE_RADIUS = 0.1
    SPHERE_RHO = 500.0
    BOX_POS = (0.0, 0.1, 0.3)
    BOX_SIZE = 0.05
    GRAVITY = 9.81
    BAR_LENGTH = 0.5
    BAR_WIDTH = 0.05
    BAR_RHO = 1000.0
    BAR_E = 1e8

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=3 if coupler_type == gs.options.SAPCouplerOptions else 2,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
        ),
        coupler_options=(
            coupler_type(fem_floor_contact_type=fem_floor_contact_type)
            if coupler_type == gs.options.SAPCouplerOptions
            else coupler_type()
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, 0.0, 0.5),
            camera_lookat=(0.0, -0.1, 0.3),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=SPHERE_POS,
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
    # A bar hanging clear of the ground and of the other bodies, whose stiffness is far beyond what the explicit solver
    # keeps stable at this time step.
    bar = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -0.4, 0.5),
            size=(BAR_WIDTH, BAR_WIDTH, BAR_LENGTH),
        ),
        material=gs.materials.FEM.Elastic(
            E=BAR_E,
            nu=0.0,
            rho=BAR_RHO,
            model="linear",
        ),
    )

    # Build the scene
    scene.build()

    # Clamp the top face of the bar
    bar_pos_0 = bar.init_positions
    bar.set_vertex_constraints(torch.nonzero(bar_pos_0[:, 2] > bar_pos_0[:, 2].max() - gs.EPS)[:, 0])

    sphere_mass_ref = (4 / 3 * np.pi * SPHERE_RADIUS**3) * SPHERE_RHO
    elems_mass_scaled = qd_to_numpy(scene.fem_solver.fem_info.elements.mass_scaled)
    sphere_elems_mass_scaled = elems_mass_scaled[sphere.el_start : (sphere.el_start + sphere.n_elements)]
    sphere_mass_1 = sphere_elems_mass_scaled.sum() / scene.fem_solver.vol_scale
    verts_mass = qd_to_numpy(scene.fem_solver.fem_info.vertices.mass)
    sphere_verts_mass = verts_mass[sphere.v_start : (sphere.v_start + sphere.n_vertices)]
    sphere_mass_2 = sphere_verts_mass.sum()
    assert_allclose(sphere_mass_1, sphere_mass_2, tol=gs.EPS)
    assert_allclose(sphere_mass_ref, sphere_mass_1, rtol=0.01)  # Large tolerance due to tessellation

    # Run simulation
    n_steps = 40 if coupler_type == gs.options.SAPCouplerOptions else 150
    for _ in range(n_steps):
        scene.step()

    for entity, init_pos, entity_halfsize_ref in zip((sphere, box), (SPHERE_POS, BOX_POS), (SPHERE_RADIUS, BOX_SIZE)):
        # Not moving anymore
        # FIXME: The SAP solve neither converges nor reproduces across CPU thread counts, so the sphere keeps a residual
        # velocity under vertex floor contact. The rework of the SAP coupler fixes both, and lifts this exception.
        if entity is box or fem_floor_contact_type != "vert":
            assert_allclose(entity.get_verts_vel(), 0.0, tol=0.025)

        # Landed vertically
        pos = tensor_to_array(entity.get_verts_pos())
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
        assert_allclose(-entity.get_verts_pos()[..., 2].min(), penetration_depth_ref, tol=tol)

    # Without Poisson contraction, the bar stretches uniaxially under its own weight, and its free end comes to rest
    # rho * g * L^2 / (2 * E) below its initial height.
    assert_allclose(bar.get_verts_vel(), 0.0, atol=2e-14 if gs.np_float == np.float64 else 2e-5)
    is_bottom = bar_pos_0[:, 2] < bar_pos_0[:, 2].min() + gs.EPS
    bar_sag = (bar_pos_0[is_bottom, 2] - bar.get_verts_pos()[..., is_bottom, 2]).mean()
    rtol = 2e-3 if gs.np_float == np.float64 else 1e-2
    assert_allclose(bar_sag, BAR_RHO * GRAVITY * BAR_LENGTH**2 / (2 * BAR_E), rtol=rtol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_legacy_coupler_rigid_velocity_correction(show_viewer):
    DT = 0.01
    GRAVITY = 9.81
    BOX_SIZE = 0.1
    GAP = 0.001
    VEL_Z = -0.5
    DROP_HEIGHT = 0.02
    # Soft enough for the explicit integration to stay stable at this time step
    SOFT_E = 1e2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.8, 0.3),
            camera_lookat=(0.25, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    ground = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box_thrown = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, 0.5 * BOX_SIZE + GAP),
        ),
        material=gs.materials.FEM.Elastic(
            E=SOFT_E,
        ),
    )
    box_dropped = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.5, 0.0, 0.5 * BOX_SIZE + DROP_HEIGHT),
        ),
        material=gs.materials.FEM.Elastic(
            E=SOFT_E,
        ),
    )
    scene.build(n_envs=2)

    # An entity translates rigidly until its first contact with the ground, so the elastic forces vanish and the
    # coupling with the ground alone alters the velocity of its vertices. Wherever the influence of the ground exceeds
    # its threshold, each surface triangle scales the normal velocity of each of its vertices by
    # '1 - influence * (1 + restitution)', once per incident triangle.
    box_thrown.set_velocity((0.0, 0.0, VEL_Z))
    # The thrown box touches the ground at the first step. The lowest vertex of the dropped box falls by
    # 'g * dt^2 * n * (n + 1) / 2' over n steps, so it stays out of range for the largest n with
    # 'n * (n + 1) <= fall_ratio', and the contact starts at the step after the next one.
    z_range = -ground.material.coup_softness * math.log(0.1)
    fall_ratio = 2.0 * (box_dropped.init_positions[:, 2].min() - z_range) / (GRAVITY * DT**2)
    n_steps = math.floor(0.5 * (math.sqrt(1.0 + 4.0 * fall_ratio) - 1.0)) + 2
    entities = (box_thrown, box_dropped)
    for i_step in range(n_steps):
        entities_pos_vel = [(entity.get_verts_pos(), entity.get_verts_vel()) for entity in entities]
        scene.step()
        for entity, (verts_pos_prev, verts_vel_prev) in zip(entities, entities_pos_vel):
            verts_influence = torch.clamp(torch.exp(-verts_pos_prev[..., 2] / ground.material.coup_softness), max=1.0)
            verts_n_tris = sum(
                [
                    torch.bincount(verts_idx, minlength=entity.n_vertices)
                    for verts_idx in torch.as_tensor(entity.surface_triangles, device=gs.device).T
                ]
            )
            verts_factor = (1.0 - verts_influence * (1.0 + ground.material.coup_restitution)) ** verts_n_tris
            verts_scale = torch.where(verts_influence > 0.1, verts_factor, 1.0)
            verts_vel_z_ref = (verts_vel_prev[..., 2] - GRAVITY * DT) * verts_scale
            verts_vel = entity.get_verts_vel()
            assert_allclose(verts_vel[..., :2], 0.0, tol=gs.EPS)
            assert_allclose(verts_vel[..., 2], verts_vel_z_ref, tol=gs.EPS)
        assert (verts_influence > 0.1).any() == (i_step == n_steps - 1)
        entities = (box_dropped,)


@pytest.mark.required
@pytest.mark.parametrize("use_implicit_solver", [False, True])
def test_hard_constraint(use_implicit_solver, show_viewer):
    DT = 0.01
    GRAVITY = 9.81
    DAMPING = 0.5
    DAMPING_BETA = 5e-4
    N_SUBSTEPS = 2 if use_implicit_solver else 6
    # The free fall at the end crosses z=0, where a scene without ground plane has no floor to stop it
    HEIGHT = 0.5
    BOX_SIZE = 0.1
    RHO = 1000.0
    MOTION_RADIUS = 0.1
    MOTION_SPEED = 0.005
    VERTICES_IDX = [5, 7]
    MODE_AMPLITUDE = 1e-3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=N_SUBSTEPS,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=use_implicit_solver,
            pcg_threshold=1e-20,
            # Each solver reads its own damping option
            damping=DAMPING,
            damping_alpha=DAMPING,
            damping_beta=DAMPING_BETA,
            enable_vertex_constraints=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, HEIGHT + 0.5),
            camera_lookat=(0.0, 0.0, HEIGHT),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    # Place the tested entity after two FEM entities sharing a material of another model, so that its local and solver
    # indices differ, and so do the indices of its entity and of its material. Those two free-fall from rest all along.
    free_material = gs.materials.FEM.Elastic(
        E=1e5,
        nu=0.45,
        rho=RHO,
        model="linear",
    )
    free_boxes = [
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(pos_x, 0.0, HEIGHT + BOX_SIZE / 2),
                size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            ),
            material=free_material,
        )
        for pos_x in (-0.5, -1.0)
    ]
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-BOX_SIZE / 2, BOX_SIZE / 2 + MOTION_RADIUS, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.45,
            rho=RHO,
            model="linear_corotated" if use_implicit_solver else "stable_neohookean",
        ),
    )
    vibrating_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-1.5, 0.0, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=free_material,
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
    free_boxes_pos_0 = torch.stack([free_box.get_verts_pos() for free_box in free_boxes])
    substep_dt = DT / N_SUBSTEPS

    # Each vertex carries a quarter of the rest mass of every element it belongs to
    boxes_elems = torch.stack([entity.get_el2v() for entity in (box, vibrating_box)])
    boxes_verts_rest = torch.stack([entity.init_positions for entity in (box, vibrating_box)])
    boxes_elems_verts = torch.eye(box.n_vertices, dtype=gs.tc_float, device=gs.device)[boxes_elems]
    boxes_elems_pos = boxes_elems_verts @ boxes_verts_rest[:, None]
    boxes_edges_rest = boxes_elems_pos[..., :3, :] - boxes_elems_pos[..., 3:, :]
    boxes_elems_vol = torch.linalg.det(boxes_edges_rest).abs() / 6.0
    boxes_verts_mass = torch.einsum("beam,be->bm", boxes_elems_verts, 0.25 * RHO * boxes_elems_vol)
    elems, _ = boxes_elems
    edges_rest, vibrating_edges_rest = boxes_edges_rest
    elems_vol, vibrating_elems_vol = boxes_elems_vol
    verts_mass, vibrating_verts_mass = boxes_verts_mass

    # The stiffness matrix of the linear model assembles V * (lam * g_a g_b^T + mu * g_b g_a^T + mu * (g_a . g_b) * I)
    # over the vertices a and b of each element, g_a being the gradient of the shape function of vertex a.
    shape_grads = torch.linalg.inv(vibrating_edges_rest)
    shape_grads = torch.cat((shape_grads, -shape_grads.sum(dim=-1, keepdim=True)), dim=-1)
    elems_grads_outer = torch.einsum("eia,ejb->eabij", shape_grads, shape_grads)
    elems_grads_dot = elems_grads_outer.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    elems_stiffness = vibrating_elems_vol[:, None, None, None, None] * (
        free_material.lam * elems_grads_outer
        + free_material.mu * elems_grads_outer.transpose(-1, -2)
        + free_material.mu * elems_grads_dot[..., None, None] * torch.eye(3, dtype=gs.tc_float, device=gs.device)
    )
    stiffness = torch.einsum("eam,ebn,eabij->minj", boxes_elems_verts[1], boxes_elems_verts[1], elems_stiffness)

    # The lowest elastic mode follows the six rigid modes, normalized to a unit modal mass
    dofs_mass_inv_sqrt = vibrating_verts_mass.repeat_interleave(3).rsqrt()
    stiffness = stiffness.reshape(3 * vibrating_box.n_vertices, 3 * vibrating_box.n_vertices)
    omegas_sqr, modes = np.linalg.eigh(tensor_to_array(dofs_mass_inv_sqrt[:, None] * stiffness * dofs_mass_inv_sqrt))
    omega_sqr = omegas_sqr[6]
    mode = dofs_mass_inv_sqrt * torch.tensor(modes[:, 6], dtype=gs.tc_float, device=gs.device)
    mode = mode.reshape(vibrating_box.n_vertices, 3)
    vibrating_box.set_position(vibrating_box.init_positions + MODE_AMPLITUDE * mode)

    assert_equal(box.get_el2v(), box.elems)

    target_poss = box.get_verts_pos()[..., VERTICES_IDX, :]
    box.set_vertex_constraints(VERTICES_IDX)
    scene.step()
    assert_allclose(box.get_verts_pos()[..., VERTICES_IDX, :], target_poss, tol=1e-8)

    if use_implicit_solver:
        # The box released in its lowest elastic mode keeps vibrating in it, the gravity moving its translation alone.
        # Each substep of the implicit solver solves (1 + h * c + h^2 * omega^2) * v' = v - h * omega^2 * q for the
        # modal velocity, with c = alpha + beta * omega^2 the Rayleigh damping of the mode, then q' = q + h * v'.
        mode_pos, mode_vel = MODE_AMPLITUDE, 0.0
        for _ in range(N_SUBSTEPS):
            mode_vel = (mode_vel - substep_dt * omega_sqr * mode_pos) / (
                1.0 + substep_dt * (DAMPING + DAMPING_BETA * omega_sqr) + substep_dt**2 * omega_sqr
            )
            mode_pos = mode_pos + substep_dt * mode_vel
        modal_mass = vibrating_verts_mass[:, None] * mode
        vibrating_box_disp = vibrating_box.get_verts_pos() - vibrating_box.init_positions
        rtol = 1e-12 if gs.np_float == np.float64 else 2e-4
        assert_allclose((vibrating_box_disp * modal_mass).sum((-2, -1)), mode_pos, rtol=rtol)
        assert_allclose((vibrating_box.get_verts_vel() * modal_mass).sum((-2, -1)), mode_vel, rtol=rtol)

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
        corners = box.get_verts_pos()[..., :8, :]
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
                scene.draw_debug_spheres(poss=target_poss, radius=0.005, color=(1, 0, 1, 0.8))
            scene.visualizer.update(force=False, auto=True)

    # Disable constraints
    box.remove_vertex_constraints()

    # Check that the box has been free-falling
    n_steps = 50
    fall_state = scene.get_state()
    box_pos_0, box_vel_0 = box.get_verts_pos(), box.get_verts_vel()
    com_pos_z_0 = box_pos_0[..., 8, 2]
    for _ in range(n_steps):
        scene.step()
    com_pos_z_f = box.get_verts_pos()[..., 8, 2]
    com_pos_delta = -0.5 * GRAVITY * (n_steps * DT) ** 2
    assert_allclose(com_pos_z_f - com_pos_z_0, com_pos_delta, tol=0.05)

    # The undeformed boxes follow the discrete free fall exactly: the gravity kick of each substep, followed by the
    # Rayleigh mass damping under the implicit solver and preceded by the exponential velocity decay under the explicit
    # one, then the position update by the new velocity.
    vel_z, pos_z = 0.0, 0.0
    for _ in range((1 + int(0.5 * math.pi / MOTION_SPEED) + n_steps) * N_SUBSTEPS):
        if use_implicit_solver:
            vel_z = (vel_z - GRAVITY * substep_dt) / (1.0 + DAMPING * substep_dt)
        else:
            vel_z = vel_z * math.exp(-DAMPING * substep_dt) - GRAVITY * substep_dt
        pos_z = pos_z + substep_dt * vel_z
    free_boxes_pos = torch.stack([free_box.get_verts_pos() for free_box in free_boxes])
    tol = 5e-10 if gs.np_float == np.float64 else (1e-3 if use_implicit_solver else 1e-4)
    assert_allclose(free_boxes_pos - free_boxes_pos_0, (0.0, 0.0, pos_z), tol=tol)

    # The internal forces of the deformed box add up to zero, so its center of mass falls like the free boxes
    com_vel = verts_mass @ box_vel_0 / verts_mass.sum()
    com_pos = verts_mass @ box_pos_0 / verts_mass.sum()
    for _ in range(n_steps * N_SUBSTEPS):
        if use_implicit_solver:
            com_vel[..., 2] -= GRAVITY * substep_dt
            com_vel /= 1.0 + DAMPING * substep_dt
        else:
            com_vel *= math.exp(-DAMPING * substep_dt)
            com_vel[..., 2] -= GRAVITY * substep_dt
        com_pos += substep_dt * com_vel
    atol = 5e-12 if gs.np_float == np.float64 else (2e-5 if use_implicit_solver else 2e-6)
    assert_allclose(verts_mass @ box.get_verts_pos() / verts_mass.sum(), com_pos, atol=atol)

    # The internal forces are the opposite of the gradient of the elastic energy of the box with respect to the
    # positions of its vertices. The linear corotated model takes the rotation of each element as given.
    mu, lam = box.material.mu, box.material.lam
    pos = box.get_verts_pos().requires_grad_()
    F = (torch.linalg.inv(edges_rest) @ (pos[:, elems[:, :3]] - pos[:, elems[:, 3:]])).transpose(-1, -2)
    if use_implicit_solver:
        U, _, Vh = torch.linalg.svd(F.detach())
        F_rot = (U @ Vh).transpose(-1, -2) @ F
        eps = 0.5 * (F_rot + F_rot.transpose(-1, -2)) - torch.eye(3, dtype=gs.tc_float, device=gs.device)
        energy_density = mu * eps.square().sum((-2, -1)) + 0.5 * lam * eps.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2
    else:
        F_norm_sqr = F.square().sum((-2, -1))
        J_shift = torch.linalg.det(F) - 1.0 - 0.75 * mu / lam
        energy_density = 0.5 * mu * (F_norm_sqr - 3.0 - (F_norm_sqr + 1.0).log()) + 0.5 * lam * J_shift**2
    energy = (elems_vol * energy_density).sum()
    forces = scene.fem_solver.get_forces()[:, box.v_start : box.v_start + box.n_vertices]
    tol = 1e-11 if gs.np_float == np.float64 else (5e-3 if use_implicit_solver else 5e-4)
    assert_allclose(forces, -torch.autograd.grad(energy, pos)[0], tol=tol)

    if use_implicit_solver:
        # Replaying the free fall reproduces it bitwise
        fall_pos = box.get_verts_pos()
        scene.reset(state=fall_state)
        for _ in range(n_steps):
            scene.step()
        assert_equal(box.get_verts_pos(), fall_pos)

    # Distinct positions and link poses per environment make cross-environment index mixups observable.
    box.set_position(((0.0, 0.0, 0.0), (0.0, 0.0, 0.2)))
    box.set_velocity((0.0, 0.0, 0.0))
    anchor.set_pos(((10.0, 0.0, HEIGHT), (20.0, 0.0, HEIGHT)), relative=False)
    anchor.set_quat(((1.0, 0.0, 0.0, 0.0), (0.5**0.5, 0.0, 0.0, 0.5**0.5)), relative=False)
    target_poss = box.get_verts_pos()[..., VERTICES_IDX, :]
    box.set_vertex_constraints(VERTICES_IDX, envs_idx=[1])
    scene.step()
    assert_allclose(box.get_verts_pos()[1, VERTICES_IDX, :], target_poss[1], tol=1e-8)

    box.remove_vertex_constraints(envs_idx=[1])
    box.set_vertex_constraints(VERTICES_IDX, link=anchor.base_link, envs_idx=[1])
    scene.step()
    tol = 1e-8 if gs.np_float == np.float64 else 2e-6
    assert_allclose(box.get_verts_pos()[1, VERTICES_IDX, :], target_poss[1], tol=tol)

    box.remove_vertex_constraints(envs_idx=[1])
    box.set_position((0.0, 0.0, 0.0))
    box.set_velocity((0.0, 0.0, 0.0))
    target_poss = box.get_verts_pos()[..., VERTICES_IDX, :]
    box.set_vertex_constraints(VERTICES_IDX)
    target_poss = target_poss + torch.tensor(
        (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.02),)), dtype=gs.tc_float, device=gs.device
    )
    box.update_constraint_targets(VERTICES_IDX, target_poss[1], envs_idx=[1])
    scene.step()
    assert_allclose(box.get_verts_pos()[..., VERTICES_IDX, :], target_poss, tol=1e-6)

    box.remove_vertex_constraints(envs_idx=[1])
    target_poss = target_poss + torch.tensor((0.0, 0.0, 0.05), dtype=gs.tc_float, device=gs.device)
    box.update_constraint_targets(VERTICES_IDX, target_poss)
    scene.step()
    constrained_poss = box.get_verts_pos()[..., VERTICES_IDX, :]
    assert_allclose(constrained_poss[0], target_poss[0], tol=1e-6)
    assert (torch.linalg.norm(constrained_poss[1] - target_poss[1], dim=-1) > 0.01).all()


@pytest.mark.required
def test_soft_constraint(asset_tmp_path, show_viewer):
    DT = 0.01
    TET_SIZE = 0.05
    RHO = 1000.0
    OMEGA = 10.0
    VELOCITY = 0.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -0.8, 0.5),
            camera_lookat=(0.15, 0.15, 0.0),
        ),
        show_viewer=show_viewer,
    )
    tet_verts = TET_SIZE * np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    tet_faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
    )
    tet_path = str(asset_tmp_path / "regular_tetrahedron.obj")
    igl.write_triangle_mesh(tet_path, tet_verts, tet_faces)
    # A material softer than the default keeps the explicit integration of these small elements stable
    tet_fixed = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=tet_path,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            rho=RHO,
        ),
    )
    tet_link = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=tet_path,
            pos=(0.3, 0.0, 0.0),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            rho=RHO,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.3, 0.3, 0.0),
            size=(0.05, 0.05, 0.05),
        ),
    )
    scene.build(n_envs=2)

    # The stiffness of each spring is proportional to the mass of its vertex, a quarter of the rest mass of every
    # element it belongs to, so that the vertices share the rate sqrt(k / m) and the tetrahedra translate undeformed
    verts = tet_fixed.init_positions
    elems = tet_fixed.get_el2v()
    elems_mass = RHO * torch.linalg.det(verts[elems[:, :3]] - verts[elems[:, 3:]]).abs() / 6.0
    verts_mass = torch.zeros_like(verts[:, 0])
    for i in range(4):
        verts_mass.index_add_(0, elems[:, i], 0.25 * elems_mass)
    for i_v, vert_mass in enumerate(verts_mass):
        stiffness = OMEGA**2 * vert_mass.item()
        tet_fixed.set_vertex_constraints(i_v, is_soft_constraint=True, stiffness=stiffness)
        tet_link.set_vertex_constraints(i_v, link=box.base_link, is_soft_constraint=True, stiffness=stiffness)

    # The first tetrahedron starts moving away from its fixed targets, while the targets of the second one start moving
    # away with the box. The offset of each to its targets follows the same critically damped oscillator.
    tet_fixed.set_velocity((VELOCITY, 0.0, 0.0))
    box.set_dofs_velocity((VELOCITY, 0.0, 0.0, 0.0, 0.0, 0.0))
    box_pos_0 = box.get_pos()
    for i in range(50):
        offset = VELOCITY * (i * DT) * math.exp(-OMEGA * (i * DT))
        assert_allclose(tet_fixed.get_verts_pos() - tet_fixed.init_positions, (offset, 0.0, 0.0), atol=5e-5)
        tet_link_offset = tet_link.get_verts_pos() - tet_link.init_positions - (box.get_pos() - box_pos_0)[:, None]
        assert_allclose(tet_link_offset, (-offset, 0.0, 0.0), atol=5e-5)
        scene.step()


# This test cannot be flagged as required because it takes 250s to run on CPU.
# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_implicit_sap_coupler_collide_sphere_box(show_viewer):
    SPHERE_RADIUS = 0.1
    BOX_SIZE = 0.015
    CUBE_HALFSIZE = 0.05
    CUBE_RHO = 1000.0
    CUBE_HYDROELASTIC_MODULUS = 1e5
    GRAVITY = 9.81

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
            gravity=(0.0, 0.0, -GRAVITY),
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
    scene.add_entity(
        morph=gs.morphs.Plane(),
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
    # A stack of two cubes resting on each other, clear of the sphere and the box
    bottom_cube, top_cube = [
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.3, -0.3, (2 * i + 1) * CUBE_HALFSIZE),
                size=(2 * CUBE_HALFSIZE, 2 * CUBE_HALFSIZE, 2 * CUBE_HALFSIZE),
            ),
            material=gs.materials.FEM.Elastic(
                rho=CUBE_RHO,
                hydroelastic_modulus=CUBE_HYDROELASTIC_MODULUS,
            ),
        )
        for i in range(2)
    ]
    scene.build()

    # Run simulation
    for _ in range(40):
        scene.step()

    for entity, init_height in zip((sphere, box), (SPHERE_RADIUS, 2 * SPHERE_RADIUS + BOX_SIZE)):
        # Barely moving
        assert_allclose(entity.get_verts_vel(), 0.0, tol=0.05)

        # More or less at the initial position
        pos = tensor_to_array(entity.get_verts_pos())
        BV, *_ = igl.bounding_box(pos)
        entity_center = 0.5 * (BV[0] + BV[-1])
        assert_allclose(entity_center[:2], 0.0, tol=0.02)
        assert_allclose(entity_center[2], init_height, tol=5e-3)

    # The pressure field of each cube is E * d / h at depth d below its surface, so the equal-pressure plane splits the
    # overlap evenly. The mesh joins each face to the cube center, so the contact face has the side 2 * h - overlap at
    # that depth, and the pressure E * overlap / (2 * h) over it balances the top cube weight rho * (2 * h)^3 * g.
    # One fixed-point iteration from the unshrunk face solves this equation well within the tolerance, which covers the
    # residual settling motion of the stack over the horizon.
    cubes_overlap = bottom_cube.get_verts_pos()[..., 2].max() - top_cube.get_verts_pos()[..., 2].min()
    cubes_overlap_ref = 4.0 * CUBE_RHO * GRAVITY * CUBE_HALFSIZE**2 / CUBE_HYDROELASTIC_MODULUS
    cubes_overlap_ref *= (2.0 * CUBE_HALFSIZE / (2.0 * CUBE_HALFSIZE - cubes_overlap_ref)) ** 2
    assert_allclose(cubes_overlap, cubes_overlap_ref, rtol=0.05)


# This test cannot be flagged as required because it takes 400s to run on CPU.
# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("model_name", ["coupled_sliders"])
def test_implicit_sap_coupler_hard_constraint_and_collision(xml_path, tol, show_viewer):
    DT = 0.01
    # The free fall at the end crosses z=0, where a scene without ground plane has no floor to stop it
    HEIGHT = 0.5
    BOX_SIZE = 0.1
    SPHERE_RADIUS = 0.03
    MOTION_RADIUS = 0.1
    MOTION_SPEED = 0.005
    VERTICES_IDX = [4, 6]
    SLIDER_FORCE = 0.01

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
    anchor = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(10.0, 0.0, HEIGHT),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    sliders = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml_path,
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    scene.build()

    # Attach the sphere to a distant rigid anchor by its center
    sphere_poss = sphere.get_verts_pos()
    sphere_poss -= torch.tensor(sphere.morph.pos, device=gs.device)
    sphere_center_idx = int(torch.argmin(torch.linalg.norm(sphere_poss, dim=-1)))
    sphere_target_poss = sphere.init_positions[sphere_center_idx]
    sphere.set_vertex_constraints(sphere_center_idx, link=anchor.base_link)

    # Push the heavy slider alone, while the light one is tied to it by the joint equality
    sliders.control_dofs_force([SLIDER_FORCE, 0.0])

    # The center of the sphere follows the anchor, while the box is still far from it
    for shift in (-0.05, 0.0):
        anchor.set_pos((10.0, shift, HEIGHT))
        scene.step()
        sphere_center = sphere.get_verts_pos()[..., sphere_center_idx, :]
        shift_pos = torch.tensor((0.0, shift, 0.0), dtype=gs.tc_float, device=gs.device)
        assert_allclose(sphere_center, sphere_target_poss + shift_pos, tol=tol)

    # The sliders move at the ratio of the joint equality while nothing is in contact yet. With the heavy slider
    # displacement being 'ratio' times the light one, the light slider has the inertia 'ratio^2 * M0 + M1'.
    ratio = sliders.equalities[0].eq_data[1]
    dofs_inertia = sliders.get_mass_mat().diagonal()
    light_slider_acc = SLIDER_FORCE * ratio / (ratio**2 * dofs_inertia[0] + dofs_inertia[1])
    slider_vel = torch.tensor((ratio, 1.0), dtype=gs.tc_float, device=gs.device) * light_slider_acc * 2 * DT
    assert_allclose(sliders.get_dofs_velocity(), slider_vel, tol=tol)

    # Move the box against the sphere, wait for a few extra steps, then release the box and let it fall
    n_steps_moving = int(0.5 * math.pi / MOTION_SPEED)
    n_steps_waiting = 10
    n_steps_falling = 40
    for it in range(n_steps_moving + n_steps_waiting + n_steps_falling):
        is_moving = it < n_steps_moving
        if is_moving:
            # Update the position of the fixed vertices
            target_poss = torch.zeros((2, 3), dtype=gs.tc_float, device=gs.device)
            target_poss[0, 0] = MOTION_RADIUS * math.sin(MOTION_SPEED * it)
            target_poss[0, 1] = MOTION_RADIUS * math.cos(MOTION_SPEED * it)
            target_poss[0, 2] = HEIGHT
            target_poss[1, 0] = (MOTION_RADIUS + BOX_SIZE) * math.sin(MOTION_SPEED * it)
            target_poss[1, 1] = (MOTION_RADIUS + BOX_SIZE) * math.cos(MOTION_SPEED * it)
            target_poss[1, 2] = HEIGHT
            box.set_vertex_constraints(VERTICES_IDX, target_poss)
        if it == n_steps_moving + n_steps_waiting:
            # Disable box constraints only
            box.remove_vertex_constraints(VERTICES_IDX)
            com_pos_0 = box.get_verts_pos()[..., 8, :]

        # Do one simulation step
        scene.step(update_visualizer=False)

        corners = box.get_verts_pos()[..., :8, :]
        sphere_center = sphere.get_verts_pos()[..., sphere_center_idx, :]
        assert_allclose(sphere_center, sphere_target_poss, tol=tol)

        # The frame of the box from its corners
        e_z = corners[..., 5, :] - corners[..., 4, :]
        e_z /= torch.linalg.norm(e_z, dim=-1, keepdim=True)
        e_y = corners[..., 6, :] - corners[..., 4, :]
        e_y /= torch.linalg.norm(e_y, dim=-1, keepdim=True)
        e_x = torch.cross(e_y, e_z, dim=-1)
        e_x /= torch.linalg.norm(e_x, dim=-1, keepdim=True)
        R = torch.stack((e_x, e_y, e_z), dim=-1)

        # Check that the sphere does not enter the box further than the compliance of the contact, which a missing
        # contact would let it cross entirely
        sphere_local = (sphere.get_verts_pos() - corners[..., [0], :]) @ R + box.init_positions[0]
        box_lower, box_upper = box.init_positions[:8].min(dim=0).values, box.init_positions[:8].max(dim=0).values
        sphere_depth = torch.minimum(sphere_local - box_lower, box_upper - sphere_local).min(dim=-1).values
        assert (sphere_depth < 0.01).all()

        if it >= n_steps_moving + n_steps_waiting:
            # Gravity acts along z and the contact only repels, so the released box cannot move back toward the sphere
            assert (box.get_verts_pos()[..., 8, 1] > com_pos_0[..., 1] - 2e-4).all()

        if is_moving:
            # Check that the constrained vertices are at their respective target positions
            assert_allclose(corners[..., VERTICES_IDX, :], target_poss, tol=tol)

            # Check that the box is more or less a box
            corners_local = (corners - corners[..., [0], :]) @ R + box.init_positions[0]
            assert_allclose(corners_local, box.init_positions[:8], tol=0.02)

            # Check that the sphere is more or less a sphere
            sphere_poss = sphere.get_verts_pos()
            sphere_poss -= torch.tensor(sphere.morph.pos, device=gs.device)
            sphere_dist_max = torch.linalg.norm(sphere_poss, dim=-1).max(dim=-1).values
            assert_allclose(sphere_dist_max, SPHERE_RADIUS, tol=0.01)

            # Check that the box is not going to far along y-axis due to collision with the sphere
            assert (corners[..., [2, 3, 7], 1] > -0.05).all()

        # Update the viewer if requested
        if show_viewer:
            if is_moving and it % max(int(1e-3 / (MOTION_SPEED * DT)), 1) == 0:
                scene.draw_debug_spheres(poss=target_poss, radius=0.005, color=(1, 0, 1, 0.8))
            scene.visualizer.update(force=False, auto=True)

    # Check that the box has been free-falling, but not the sphere
    com_pos_z_f = box.get_verts_pos()[..., 8, 2]
    com_pos_delta = -0.5 * 9.81 * (n_steps_falling * DT) ** 2
    assert_allclose(com_pos_z_f - com_pos_0[..., 2], com_pos_delta, tol=0.05)
    sphere_center = sphere.get_verts_pos()[..., sphere_center_idx, :]
    assert_allclose(sphere_center, sphere_target_poss, tol=tol)
