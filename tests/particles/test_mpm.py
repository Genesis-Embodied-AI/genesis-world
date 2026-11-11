import numpy as np
import pytest
import torch
import trimesh

import genesis as gs

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_particle_constraints(show_viewer, tol):
    GRAVITY = 9.81
    N_STEPS = 5
    # A density that single precision cannot represent
    RHO = 1000.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=20,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.2, 1.0),
            camera_lookat=(0.0, 0.3, 0.35),
            camera_fov=32,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    rigid_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.55),
            size=(0.12, 0.12, 0.05),
            fixed=True,
        ),
    )
    mpm_cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=5e4,
            nu=0.3,
            rho=RHO,
        ),
        morph=gs.morphs.Box(
            pos=(0, 0, 0.35),
            size=(0.15, 0.15, 0.15),
        ),
    )
    # Two copies of one box offset along its local y axis, each placed with its own member pose
    box = trimesh.creation.box(
        extents=(0.04, 0.12, 0.04), transform=trimesh.transformations.translation_matrix((0.0, 0.1, 0.0))
    )
    mpm_mesh_set = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(box, box),
            poss=((0.5, 0.5, 0.3), (-0.5, 0.5, 0.3)),
            eulers=((0.0, 0.0, 90.0), (90.0, 0.0, 0.0)),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    moving_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -0.5, 0.85),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    mpm_follower = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -0.5, 0.7),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    # A block held by its middle slab, the particles on either side mirroring each other about it
    mpm_held_block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, -0.5, 0.3),
            size=(0.1, 0.05, 0.05),
        ),
        material=gs.materials.MPM.Elastic(
            sampler="regular",
        ),
    )
    scene.build(n_envs=2)

    # The mass of the particles follows the density at the precision of the simulation
    assert_allclose(mpm_cube.get_mass(), mpm_cube.n_particles * scene.mpm_solver.particle_volume_real * RHO, tol=tol)

    # Each member is rotated about its own position: the box offset lands at -x for the member rotated about z and at
    # +z for the one rotated about x, and the long axis follows. A regular lattice spans the box minus one particle.
    particles = mpm_mesh_set.init_particles
    particle_size = mpm_mesh_set.particle_size
    long_span, short_span = 0.12 - particle_size, 0.04 - particle_size
    particles_z90 = particles[particles[:, 0] > 0.0]
    particles_x90 = particles[particles[:, 0] < 0.0]
    assert_allclose(particles_z90.mean(axis=0), (0.4, 0.5, 0.3), tol=tol)
    assert_allclose(np.ptp(particles_z90, axis=0), (long_span, short_span, short_span), tol=tol)
    assert_allclose(particles_x90.mean(axis=0), (-0.5, 0.5, 0.4), tol=tol)
    assert_allclose(np.ptp(particles_x90, axis=0), (short_span, short_span, long_span), tol=tol)
    # The visual mesh carries the same member poses
    assert_allclose(mpm_mesh_set.vmesh.verts.min(axis=0), (-0.52, 0.48, 0.28), tol=tol)
    assert_allclose(mpm_mesh_set.vmesh.verts.max(axis=0), (0.46, 0.52, 0.46), tol=tol)
    # Positioning the set as a whole moves the mean member position there, keeping the members' layout
    mpm_mesh_set.set_position((0.0, 0.5, 0.5))
    assert_allclose(mpm_mesh_set.get_particles_pos(), particles + (0.0, 0.0, 0.2), tol=tol)

    # Test get_particles_in_bbox - returns (n_envs, n_particles) mask
    mask = mpm_cube.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
    assert mask.shape == (2, mpm_cube.n_particles), "mask should be (n_envs, n_particles)"
    assert mask.any(), "bbox should select some particles"
    assert not mask.all(), "bbox should not select all particles"

    # A checkpoint made before any particle is constrained
    checkpoint_unconstrained = scene.sim.__getstate__()

    # Attach and test following
    link_idx = rigid_box.links[0].idx
    mpm_cube.set_particle_constraints(mask, link_idx, stiffness=1e5)
    initial_rigid_pos = rigid_box.get_pos()
    initial_mpm_x = mpm_cube.get_particles_pos()[:, mask[0], 0].mean()

    pos_diff = torch.tensor([0.2, 0, 0], device=gs.device)
    rigid_box.set_pos(initial_rigid_pos + pos_diff, zero_velocity=False)

    # Particles attached to a link translating at the velocity they share move rigidly with it
    mpm_follower.set_particle_constraints(True, moving_box.links[0].idx, stiffness=1e3)
    moving_box.set_dofs_velocity((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    mpm_follower.set_velocity((1.0, 0.0, 0.0))
    follower_pos_0 = mpm_follower.get_particles_pos()
    moving_box_pos_0 = moving_box.get_pos()

    held_block_offset_x = mpm_held_block.init_particles[:, 0] - 0.5
    mpm_held_block.set_free(np.abs(held_block_offset_x) > mpm_held_block.particle_size)

    for _ in range(30):
        scene.step()

    # The attached particles follow the link, the springs keeping every particle finite whatever their stiffness
    assert mpm_cube.get_particles_vel().isfinite().all()
    mpm_diff = mpm_cube.get_particles_pos()[:, mask[0], 0].mean() - initial_mpm_x
    # The springs deflect under the pull of the rest of the block, so the attached particles lag the link by a physical
    # offset that precision does not bound
    assert_allclose(mpm_diff, pos_diff[0], tol=1e-4)

    # The followers keep their offset to the link along its motion, gravity only sagging them vertically
    moving_box_disp = moving_box.get_pos() - moving_box_pos_0
    assert_allclose(moving_box_disp, (30 * scene.dt, 0.0, 0.0), tol=tol)
    follower_offset = mpm_follower.get_particles_pos() - follower_pos_0 - moving_box_disp[:, None]
    assert_allclose(follower_offset[..., :2], 0.0, tol=tol)

    # The held particles zero the cells they touch once every particle scattered into them, whatever the order of the
    # particles, so the sides of the block sag alike
    held_block_pos = mpm_held_block.get_particles_pos()
    assert_allclose(
        held_block_pos[:, held_block_offset_x > 0.0, 2].mean(dim=-1),
        held_block_pos[:, held_block_offset_x < 0.0, 2].mean(dim=-1),
        tol=tol,
    )

    # Restoring the checkpoint brings back unconstrained particles, which fall freely
    scene.sim.__setstate__(checkpoint_unconstrained)
    for _ in range(N_STEPS):
        scene.step()
    assert_allclose(mpm_follower.get_particles_vel()[..., 2], -GRAVITY * N_STEPS * scene.dt, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("variant", ["svd", "no_svd"])
def test_perf_dispatch(variant, show_viewer):
    if variant == "svd":
        elastic_model = "corotation"
        liquid_viscous = True
    else:
        elastic_model = "neohooken"
        liquid_viscous = False

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=64,
            lower_bound=(-0.4, -0.4, -0.05),
            upper_bound=(0.4, 0.4, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    elastic = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.1, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Elastic(
            E=3e5,
            nu=0.2,
            rho=200.0,
            model=elastic_model,
        ),
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
        material=gs.materials.MPM.Liquid(
            viscous=liquid_viscous,
        ),
    )
    scene.build(n_envs=2)

    # Aggregate SVD flag must match the variant for the dispatch path under test to actually run.
    assert scene.sim.mpm_solver.mpm_config.is_svd_needed == (variant == "svd")

    init_elastic_pos = elastic.get_particles_pos()
    init_liquid_pos = liquid.get_particles_pos()

    for _ in range(100):
        scene.step()

    final_elastic_pos = elastic.get_particles_pos()
    final_liquid_pos = liquid.get_particles_pos()

    # Gravity acted: particles moved down on average.
    assert init_elastic_pos[..., 2].mean() - final_elastic_pos[..., 2].mean() > 0.05
    assert init_liquid_pos[..., 2].mean() - final_liquid_pos[..., 2].mean() > 0.05

    # No ground penetration (plane is at z=0). Non-viscous liquid spreads thin enough that a sub-grid-cell penetration
    # is normal for the MPM coupling, so allow up to 2mm.
    assert final_elastic_pos[..., 2].min() > -1e-3
    assert final_liquid_pos[..., 2].min() > -2e-3


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
def test_halt_on_nan(show_viewer):
    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.3),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            E=1e9,
        ),
    )
    scene.build(n_envs=2)

    # The explicit integration of a stiff block diverges, the block of the second env first since it hits the floor
    # faster. The diverging env is flagged and frozen before its state stops being finite, while the other proceeds.
    block.set_velocity(((0.0, 0.0, 0.0), (0.0, 0.0, -5.0)))
    for _ in range(10):
        scene.step()
        assert block.get_particles_pos().isfinite().all()
        if scene.get_error_envs_mask().any():
            break
    assert_equal(scene.get_error_envs_mask(), (False, True))

    # The error raises at the beginning of the next step on CPU, and at worst of the one after it on GPU
    with pytest.raises(gs.GenesisException, match="Invalid MPM particle velocities"):
        for _ in range(1 if gs.backend == gs.cpu else 2):
            scene.step()

    # Resetting the scene clears the error, so the simulation proceeds
    scene.reset()
    scene.step()
    assert block.get_particles_pos().isfinite().all()
    assert not scene.get_error_envs_mask().any()


@pytest.mark.required
def test_implicit_particle_constraints(show_viewer):
    GRAVITY = 9.81
    LINK_SHIFT = 0.2
    VELOCITY = 1.0
    N_STEPS = 30
    N_STEPS_FALL = 5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            use_implicit_solver=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, -1.5, 1.1),
            camera_lookat=(0.15, -0.25, 0.55),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    rigid_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.55),
            size=(0.12, 0.12, 0.05),
            fixed=True,
        ),
        vis_mode="collision",
    )
    mpm_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.35),
            size=(0.15, 0.15, 0.15),
        ),
        material=gs.materials.MPM.Elastic(),
        vis_mode="particle",
    )
    moving_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -0.5, 0.85),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
        vis_mode="collision",
    )
    mpm_follower = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -0.5, 0.7),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.MPM.Elastic(),
        vis_mode="particle",
    )
    scene.build(n_envs=2)
    checkpoint_unconstrained = scene.sim.__getstate__()

    # The top of the cube hangs from a fixed link by stiff springs, the link then moved aside at once
    mask = mpm_cube.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
    mpm_cube.set_particle_constraints(mask, rigid_box.links[0].idx, stiffness=1e5)
    attached_x_0 = mpm_cube.get_particles_pos()[:, mask[0], 0].mean(dim=-1)
    link_shift = torch.tensor((LINK_SHIFT, 0.0, 0.0), device=gs.device)
    rigid_box.set_pos(rigid_box.get_pos() + link_shift, zero_velocity=False)

    # The follower hangs from a link moving at the velocity they share
    mpm_follower.set_particle_constraints(True, moving_box.links[0].idx, stiffness=1e3)
    moving_box.set_dofs_velocity((VELOCITY, 0.0, 0.0, 0.0, 0.0, 0.0))
    mpm_follower.set_velocity((VELOCITY, 0.0, 0.0))
    follower_pos_0 = mpm_follower.get_particles_pos()
    moving_box_pos_0 = moving_box.get_pos()

    for _ in range(N_STEPS):
        scene.step()

    # The attached particles follow the link however stiff their springs, the cube neither left behind nor blown up
    assert mpm_cube.get_particles_vel().isfinite().all()
    attached_x = mpm_cube.get_particles_pos()[:, mask[0], 0].mean(dim=-1)
    assert_allclose(attached_x - attached_x_0, LINK_SHIFT, tol=1e-2 * LINK_SHIFT)

    # The follower keeps its offset to its link, which keeps its velocity, where a spring target lagging a step behind
    # the link would leave it a step behind
    step_length = VELOCITY * scene.dt
    moving_box_disp = moving_box.get_pos() - moving_box_pos_0
    follower_offset = mpm_follower.get_particles_pos() - follower_pos_0 - moving_box_disp[:, None]
    assert_allclose(moving_box_disp, (N_STEPS * step_length, 0.0, 0.0), tol=0.1 * step_length)
    assert_allclose(follower_offset[..., :2], 0.0, tol=0.1 * step_length)

    # Restoring the checkpoint brings back unconstrained particles, which fall freely where attached ones would hang
    scene.sim.__setstate__(checkpoint_unconstrained)
    for _ in range(N_STEPS_FALL):
        scene.step()
    fall_vel = GRAVITY * N_STEPS_FALL * scene.dt
    assert_allclose(mpm_follower.get_particles_vel()[..., 2], -fall_vel, tol=0.1 * fall_vel)


@pytest.mark.required
def test_implicit_halt_on_nan(show_viewer):
    IMPACT_VELOCITY = 5.0
    N_STEPS = 50

    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            use_implicit_solver=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 0.5, 0.4),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=35,
        ),
        show_viewer=show_viewer,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.3),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(
            E=1e9,
            sampler="regular",
        ),
        vis_mode="particle",
    )
    # A rigid box falling beside the block, whose dofs join the Newton solve
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.3, 0.0, 0.9),
            size=(0.05, 0.05, 0.05),
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=2)
    pos_0 = block.get_particles_pos()
    span_0 = pos_0.amax(dim=-2) - pos_0.amin(dim=-2)
    particle_size = block.particle_size

    # A block as stiff as hard plastic falls onto the floor of the domain, thrown at it in the second env. It comes to
    # rest on the floor as undeformed as it started, without any env flagged.
    block.set_velocity(((0.0, 0.0, 0.0), (0.0, 0.0, -IMPACT_VELOCITY)))
    for _ in range(N_STEPS):
        scene.step()
    assert not scene.get_error_envs_mask().any()
    pos = block.get_particles_pos()
    assert_allclose(block.get_particles_vel(), 0.0, tol=1e-2)
    assert_allclose(pos[..., 2].amin(dim=-1), scene.mpm_solver.boundary_lower[2], tol=0.5 * particle_size)
    assert_allclose(pos.amax(dim=-2) - pos.amin(dim=-2), span_0, tol=0.1 * particle_size)

    # A non-finite velocity fails the Newton solve of its env, which is flagged and frozen, its particles and the rigid
    # box alike, while the other proceeds
    block.set_velocity(((0.0, 0.0, 1.0), (0.0, 0.0, float("nan"))))
    pos_free_0 = block.get_particles_pos(envs_idx=[0])
    pos_failed_0 = block.get_particles_pos(envs_idx=[1])
    box_pos_free_0 = box.get_pos(envs_idx=[0])
    box_qpos_failed_0 = box.get_qpos(envs_idx=[1])
    box_dofs_vel_failed_0 = box.get_dofs_velocity(envs_idx=[1])
    scene.step()
    assert_equal(scene.get_error_envs_mask(), (False, True))
    assert (block.get_particles_pos(envs_idx=[0])[..., 2] > pos_free_0[..., 2]).all()
    assert_equal(block.get_particles_pos(envs_idx=[1]), pos_failed_0)
    assert (box.get_pos(envs_idx=[0])[..., 2] < box_pos_free_0[..., 2]).all()
    assert_equal(box.get_qpos(envs_idx=[1]), box_qpos_failed_0)
    assert_equal(box.get_dofs_velocity(envs_idx=[1]), box_dofs_vel_failed_0)

    # The error raises at the beginning of the next step on CPU, and at worst of the one after it on GPU
    with pytest.raises(gs.GenesisException, match="Newton solve of the implicit MPM solver"):
        for _ in range(1 if gs.backend == gs.cpu else 2):
            scene.step()

    # Resetting the scene clears the error, so the simulation proceeds
    scene.reset()
    scene.step()
    assert block.get_particles_pos().isfinite().all()
    assert not scene.get_error_envs_mask().any()


@pytest.mark.required
def test_implicit_emitted_particles(show_viewer):
    NOZZLE_HEIGHT = 0.5
    DROPLET_SIZE = 0.1
    N_DROPLETS = 5
    N_STEPS = 50

    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 1.0),
            use_implicit_solver=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, -0.7, 0.55),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    emitter = scene.add_emitter(
        material=gs.materials.MPM.Liquid(),
        max_particles=5000,
    )
    scene.build(n_envs=2)

    # Droplets shot down at the plane join the simulation as they are emitted
    for _ in range(N_DROPLETS):
        emitter.emit(droplet_shape="circle", droplet_size=DROPLET_SIZE, pos=(0.0, 0.0, NOZZLE_HEIGHT), speed=2.0)
        scene.step()
    is_emitted = emitter.entity.get_particles_active()
    for _ in range(N_STEPS):
        scene.step()

    # They land on the plane and spread over it, which holds them above it
    pos = emitter.entity.get_particles_pos()[is_emitted]
    assert pos.isfinite().all()
    assert (pos[:, 2] > -emitter.entity.particle_size).all()
    assert (pos[:, 2] < 0.2 * NOZZLE_HEIGHT).all()
    assert ((pos[:, :2].amax(dim=0) - pos[:, :2].amin(dim=0)) > 2.0 * DROPLET_SIZE).all()


@pytest.mark.required
@pytest.mark.parametrize("partner", ["SPH", "PBD", "FEM"])
def test_implicit_solver_coupling_refusal(partner):
    scene = gs.Scene(
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            use_implicit_solver=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.2, 0.0, 0.2),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.2, 0.0, 0.2),
            size=(0.1, 0.1, 0.1),
        ),
        material={
            "SPH": gs.materials.SPH.Liquid,
            "PBD": gs.materials.PBD.Liquid,
            "FEM": gs.materials.FEM.Elastic,
        }[partner](),
    )
    with pytest.raises(gs.GenesisException, match=f"Coupling MPM with {partner} requires the explicit MPM solver"):
        scene.build()


@pytest.mark.required
@pytest.mark.parametrize(
    "unsupported",
    [
        "requires_grad",
        # The SAP coupler refuses single precision first
        pytest.param("coupler", marks=pytest.mark.precision("64")),
        "cpic",
        "elliptic_cone",
        "cg_solver",
        "hibernation",
        "noslip",
        "custom_material",
    ],
)
def test_implicit_solver_refusal(unsupported):
    if unsupported == "cpic":
        with pytest.raises(gs.GenesisException, match="'enable_CPIC' requires the explicit solver"):
            gs.options.MPMOptions(use_implicit_solver=True, enable_CPIC=True)
        return

    # A material of a stress of its own, which the explicit solver runs and the implicit one has no energy for
    class CustomMaterial(gs.materials.MPM.Base):
        pass

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            requires_grad=unsupported == "requires_grad",
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic if unsupported == "elliptic_cone" else gs.friction_cone.pyramidal,
            constraint_solver=gs.constraint_solver.CG if unsupported == "cg_solver" else gs.constraint_solver.Newton,
            use_hibernation=unsupported == "hibernation",
            noslip_iterations=5 if unsupported == "noslip" else 0,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions() if unsupported == "coupler" else None,
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.1, 0.1, 0.1),
        ),
        material=CustomMaterial() if unsupported == "custom_material" else gs.materials.MPM.Elastic(),
    )
    # A free rigid body, whose dofs the Newton solve takes when it touches the particles
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.4),
            size=(0.1, 0.1, 0.1),
        ),
    )
    error_message = {
        "requires_grad": "Differentiable simulation",
        "coupler": "requires LegacyCouplerOptions",
        "elliptic_cone": "friction_cone=gs.friction_cone.pyramidal",
        "cg_solver": "constraint_solver=gs.constraint_solver.Newton",
        "hibernation": "use_hibernation=False",
        "noslip": "noslip_iterations=0",
        "custom_material": "supports the Elastic, ElastoPlastic, Liquid, Muscle, Sand and Snow materials alone",
    }[unsupported]
    with pytest.raises(gs.GenesisException, match=error_message):
        scene.build()
