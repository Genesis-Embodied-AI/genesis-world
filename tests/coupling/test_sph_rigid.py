import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@pytest.mark.parametrize(
    "n_envs, pressure_solver",
    [
        pytest.param(0, "WCSPH", marks=pytest.mark.required),
        (0, "DFSPH"),
        pytest.param(2, "WCSPH", marks=pytest.mark.slow),  # ~150s
        pytest.param(2, "DFSPH", marks=pytest.mark.slow),  # ~300s
    ],
)
def test_flotation_follows_density_ratio(n_envs, pressure_solver, show_viewer):
    PARTICLE_SIZE = 0.02
    BLOCK_SIZE = 0.1
    BLOCK_DENSITY_RATIO = 0.5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=PARTICLE_SIZE,
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
    liquid = scene.add_entity(
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
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -0.13, 0.51 + 0.5 * BLOCK_SIZE),
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            rho=BLOCK_DENSITY_RATIO * gs.materials.SPH.Liquid().rho,
        ),
    )
    scene.build(n_envs=n_envs)

    # The light ball first sinks with the settling fluid before buoyancy lifts it above the surface
    raft_rolls, block_drafts = [], []
    for i_step in range(100):
        scene.step()
        raft_rolls.append(gu.quat_to_xyz(raft.get_quat(), rpy=True, degrees=True)[..., 0])

        # The liquid never enters the FEM block, whose surface holds it back like a wall. The particles next to a face
        # stand half a particle size off it, so none lies within the block shrunk by that much.
        verts_pos, particles_pos = block.get_verts_pos(), liquid.get_particles_pos()
        block_center = verts_pos.mean(dim=-2, keepdim=True)
        assert ((particles_pos - block_center).abs().amax(dim=-1) > 0.5 * (BLOCK_SIZE - PARTICLE_SIZE)).all()

        # The draft of the block below the free surface of the liquid around it, the surface lying half a particle size
        # above the centers of the topmost particles
        if i_step >= 80:
            offsets = (particles_pos[..., :2] - block_center[..., :2]).abs().amax(dim=-1)
            is_around = (offsets > 0.5 * BLOCK_SIZE + PARTICLE_SIZE) & (offsets < 0.5 * BLOCK_SIZE + 3 * PARTICLE_SIZE)
            heights_around = torch.where(is_around, particles_pos[..., 2], float("-inf"))
            level = heights_around.topk(20, dim=-1).values.mean(dim=-1) + 0.5 * PARTICLE_SIZE
            block_drafts.append(level - verts_pos[..., 2].amin(dim=-1))

            # The block rides the waves the other bodies raise, moving vertically with the liquid around it
            verts_lo, verts_hi = verts_pos.amin(dim=-2, keepdim=True), verts_pos.amax(dim=-2, keepdim=True)
            offsets_out = torch.maximum(verts_lo - particles_pos, particles_pos - verts_hi).clamp(min=0.0)
            dists = offsets_out.norm(dim=-1)
            is_near = (dists > 0.0) & (dists < 2 * scene.sph_solver.support_radius)
            particles_vel_z = liquid.get_particles_vel()[..., 2]
            liquid_vel_z = torch.where(is_near, particles_vel_z, 0.0).sum(dim=-1) / is_near.sum(dim=-1)
            assert ((block.get_verts_vel()[..., 2].mean(dim=-1) - liquid_vel_z).abs() < 1e-1).all()
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
    # The FEM block floats with the draft at which the weight of the liquid it displaces balances its own, up to the
    # smoothing of the surface by the kernel over half a particle size.
    assert_allclose(torch.stack(block_drafts).mean(dim=0), BLOCK_DENSITY_RATIO * BLOCK_SIZE, atol=0.5 * PARTICLE_SIZE)


@pytest.mark.parametrize("n_envs", [0, pytest.param(2, marks=pytest.mark.required)])
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
