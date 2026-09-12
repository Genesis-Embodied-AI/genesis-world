import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy


@pytest.mark.required
@pytest.mark.parametrize(
    "n_envs, pressure_solver",
    [
        (0, "WCSPH"),
        (0, "DFSPH"),
        pytest.param(2, "WCSPH", marks=pytest.mark.slow),  # ~150s
        pytest.param(2, "DFSPH", marks=pytest.mark.slow),  # ~300s
    ],
)
def test_rigid_flotation_follows_density_ratio(n_envs, pressure_solver, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            pressure_solver=pressure_solver,
            lower_bound=(-0.25, -0.25, 0.0),
            upper_bound=(0.25, 0.25, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.5, 0.5, 0.5),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    light_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(-0.12, 0.0, 0.25),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    heavy_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.12, 0.0, 0.25),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=1500.0,
        ),
    )
    raft = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.12, 0.25),
            size=(0.1, 0.1, 0.04),
            euler=(30.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    scene.build(n_envs=n_envs)

    raft_rolls = []
    for _ in range(50):
        scene.step()
        raft_rolls.append(gu.quat_to_xyz(raft.get_quat(), rpy=True, degrees=True)[..., 0])
    raft_rolls = torch.stack(raft_rolls)

    # The static fluid pressure must push the light ball (rho well below the fluid rest density) up toward the
    # surface, while the heavy ball (rho well above) must keep sinking: flotation discriminates on the density
    # ratio, which is Archimedes' principle.
    light_z = light_ball.get_pos()[..., 2]
    heavy_z = heavy_ball.get_pos()[..., 2]
    assert (light_z > 0.28).all(), f"Light ball must rise under buoyancy, got z={light_z}"
    assert (heavy_z < 0.22).all(), f"Heavy ball must sink, got z={heavy_z}"
    # The pressure on the tilted raft acts off its center of mass, so the buoyancy torque rocks it about its roll
    # axis. That torque reaches the rigid body through the coupling wrench alone.
    assert (raft_rolls.amax(dim=0) - raft_rolls.amin(dim=0) > 2.0).all(), f"Raft must rock, got rolls={raft_rolls}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_coupling_force_wakes_hibernated_link(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-0.55, -0.25, 0.0),
            upper_bound=(0.55, 0.25, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.5, 1.0),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.35, 0.0, 0.15),
            size=(0.3, 0.4, 0.3),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.35, 0.0, 0.06),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=200.0,
        ),
    )
    # The island partition, which hibernation builds on, only engages when the scene holds at least two free
    # bodies; this bystander rests dry on the floor for the whole scenario.
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.06),
            radius=0.06,
        ),
        material=gs.materials.Rigid(
            rho=2000.0,
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(15):
        scene.step()

    # The dry resting ball must have hibernated, otherwise the scenario would validate flotation instead of the
    # wake-up on coupling forces.
    is_ball_hibernated = qd_to_numpy(scene.rigid_solver.dyn_state.links.is_hibernated, transpose=True)
    assert is_ball_hibernated[..., ball.links[0].idx].all()

    # Teleporting the fluid over the hibernated ball must wake it through the coupling forces alone: there is no
    # awake body around to collide with, so with the ball left asleep the fluid pressure would be silently
    # discarded and the ball would stay frozen on the floor instead of floating up.
    liquid.set_position((0.35, 0.0, 0.15))
    for _ in range(40):
        scene.step()
    ball_z = ball.get_pos()[..., 2]
    assert (ball_z > 0.15).all(), f"Submerged ball must wake up and float, got z={ball_z}"
