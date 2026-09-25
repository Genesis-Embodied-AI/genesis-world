import pytest
import numpy as np
import torch

import genesis as gs

from ..utils.assertions import assert_allclose


# Note that "session" scope must NOT be used because the material while be altered without copy when building the scene
@pytest.fixture(scope="function")
def pbd_material():
    """Fixture for common FEM material properties"""
    return gs.materials.PBD.Elastic()


@pytest.mark.required
def test_maxvolume(pbd_material, show_viewer, box_obj_path):
    scene = gs.Scene(
        pbd_options=gs.options.PBDOptions(
            particle_size=0.1,
        ),
        show_viewer=show_viewer,
    )

    # Mesh without any maximum-element-volume constraint
    pbd1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            verbose=1,
        ),
        material=pbd_material,
    )

    # Mesh with maximum element volume limited to 0.001
    pbd2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            maxvolume=0.001,
            verbose=1,
        ),
        material=pbd_material,
    )

    assert pbd1.n_elems < pbd2.n_elems, (
        f"Mesh with maxvolume=0.01 generated {pbd2.n_elems} elements; "
        f"expected more than {pbd1.n_elems} elements without a volume limit."
    )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_get_mass(n_envs, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 0, 0.5),
            scale=1.0,
        ),
        material=gs.materials.PBD.Cloth(),
    )
    scene.build(n_envs=n_envs)

    mass = cloth.get_mass()
    expected_n = max(n_envs, 1)
    assert mass.shape == (expected_n,), f"Expected shape ({expected_n},), got {mass.shape}"
    assert (mass > 0).all(), f"Expected positive mass, got {mass}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("material_type", [gs.materials.PBD.Cloth])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cloth_attach_fixed_point(n_envs, material_type, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cloth_1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 2.0, 0.5),
            scale=1.0,
        ),
        material=material_type(),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 0.2, 1.0),
            vis_mode="visual",
        ),
    )
    cloth_2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 0, 0.1),
            scale=2.0,
        ),
        material=material_type(),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 0.2, 1.0),
            vis_mode="visual",
        ),
    )
    scene.build(n_envs=n_envs)

    # Make sure that 'get_position', 'get_velocity' is working
    for cloth in (cloth_1, cloth_2):
        init_com = torch.tensor([2.0, 0.0, 1.0])
        cloth.set_position(init_com)
        cloth.process_input()
        poss = cloth.get_particles_pos().clone()
        assert_allclose(poss.mean(dim=-2), init_com, tol=1e-2)
        poss += torch.tensor([0.5, -1.0, -0.5])
        vels = torch.rand_like(poss)
        cloth.set_position(poss)
        cloth.set_velocity(vels)
        assert_allclose(cloth.get_particles_vel(), 0.0, tol=tol)
        cloth.process_input()
        assert_allclose(cloth.get_particles_pos(), poss, tol=tol)
        assert_allclose(cloth.get_particles_vel(), vels, tol=tol)
    scene.reset()

    # Simulate for a while
    for _ in range(40):
        scene.step()

    # Make sure that the cloth is landing perfectly vertically and laying on the ground without moving
    poss = cloth_2.get_particles_pos()
    assert_allclose(poss[..., :2], cloth_2._mesh.verts[..., :2], tol=tol)
    assert_allclose(poss[..., 2], cloth_2._particle_size / 2, tol=tol)
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels, 0.0, tol=tol)

    # Attach top-left corner and simulate for a while
    particle_idx = cloth_2.find_closest_particle((-1, -1, 0))
    particle_pos_ref = (-0.5, -0.5, 0.05)
    cloth_2.set_particles_pos(particle_pos_ref, particle_idx)
    cloth_2.fix_particles(particle_idx)
    for i in range(60):
        scene.step()

    # Make sure that the corner is at the target position, some points are still on the ground, and none are moving
    poss = cloth_2.get_particles_pos()
    if scene.n_envs > 0:
        particle_pos = poss[torch.arange(scene.n_envs), particle_idx]
    else:
        particle_pos = poss[particle_idx]
    assert_allclose(particle_pos, particle_pos_ref, tol=tol)
    assert_allclose(poss[..., 2].min(dim=-1).values, cloth_2._particle_size / 2, tol=tol)
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels[..., 2].mean(dim=-1), 0.0, tol=1e-3)

    # Release cloth
    cloth_2.release_particle(particle_idx)
    for i in range(30):
        scene.step()

    # Make sure that the cloth is laying on the ground without moving
    poss = cloth_2.get_particles_pos()
    assert_allclose(poss[..., 2], cloth_2._particle_size / 2, tol=tol)
    assert -0.6 < poss[..., :1].min() and poss[..., :1].max() < 0.6
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels[..., 2].mean(dim=-1), 0.0, tol=tol)


@pytest.mark.required
def test_cloth_attach_rigid_link(show_viewer):
    particle_size = 0.01
    box_height = 2.25

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-2,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=particle_size,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 2.0, 4.0),
            camera_lookat=(-1.0, -1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.25, 0.25, box_height),
            size=(0.25, 0.25, 0.25),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0.25, 0.25, box_height + 0.125 + particle_size),
            scale=0.5,
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )
    scene.build(n_envs=2)

    particles_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    box_link_idx = box.links[0].idx
    cloth.fix_particles_to_link(box_link_idx, particles_idx_local=particles_idx)

    # leftward velocity for both envs on base linear DOFs [3,4,5]
    vel = np.array([[-0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    box.set_dofs_velocity(vel, dofs_idx_local=[0, 1, 2])

    cloth_pos0 = cloth.get_particles_pos()[:, particles_idx].clone()
    link_pos0 = scene.rigid_solver.links[box_link_idx].get_pos().clone()

    for _ in range(20):
        scene.step()

    # wait for the box to stop
    box.set_dofs_velocity(np.zeros_like(vel), dofs_idx_local=[0, 1, 2])
    for _ in range(10):
        scene.step()

    # Check that the attached particles followed the link displacement per env
    cloth_pos1 = cloth.get_particles_pos()[:, particles_idx].clone()
    link_pos1 = scene.rigid_solver.links[box_link_idx].get_pos().clone()
    assert_allclose((cloth_pos1 - cloth_pos0).movedim(0, -2), link_pos1 - link_pos0, atol=2e-5)

    # Release cloth and restore box's speed
    box.set_dofs_velocity(vel, dofs_idx_local=[0, 1, 2])
    cloth.release_particle(particles_idx)
    for _ in range(15):
        scene.step()

    # Make sure that the cloth is laying on the ground without moving
    cloth_pos2 = cloth.get_particles_pos()[:, particles_idx]
    link_pos2 = scene.rigid_solver.links[box_link_idx].get_pos()
    link_disp = link_pos2 - link_pos1
    cloth_disp = cloth_pos2 - cloth_pos1
    assert ((cloth_disp.movedim(0, -2) - link_disp).norm(dim=-1) > 0.2).all()
