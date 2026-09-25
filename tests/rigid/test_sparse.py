import pytest

import genesis as gs
import genesis.utils.geom as gu

from ..utils.assertions import assert_allclose


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_noslip_resting_stability(show_viewer):
    TABLE_Z = 0.762

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=128,
            noslip_iterations=2,
            sparse_solve=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.7, -1.1, 1.4),
            camera_lookat=(0.4, 0.0, 0.75),
        ),
        show_viewer=show_viewer,
    )

    franka = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/panda_bullet/panda.urdf",
            pos=(0, 0, TABLE_Z),
            fixed=True,
        )
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0, TABLE_Z / 2),
            size=(0.8, 0.6, TABLE_Z / 2),
            fixed=True,
        )
    )

    boxes = []
    for i in range(16):
        # Keep the grid clear of the arm: boxes dropped closer to the base land on the panda instead of the table.
        x = 0.4 + 0.12 * (i % 4)
        y = -0.25 + 0.12 * (i // 4)
        box = scene.add_entity(
            material=gs.materials.Rigid(
                friction=0.5,
            ),
            # Drop from just above the resting height: a high drop makes the landing chaotic across precisions,
            # while the point here is the stability of the resting contacts.
            morph=gs.morphs.Box(
                pos=(x, y, 0.75 * TABLE_Z + 0.025),
                size=(0.04, 0.04, 0.04),
            ),
            surface=gs.surfaces.Default(
                color=(0.3 + 0.7 * (i % 4) / 3, 0.3 + 0.7 * (i // 4) / 3, 0.5, 1.0),
            ),
            visualize_contact=True,
        )
        boxes.append(box)

    scene.build()

    # Hold the arm at its initial configuration, otherwise it collapses under gravity and sweeps the boxes off.
    franka.control_dofs_position(franka.get_qpos())

    for _ in range(80):
        scene.step()

    # The boxes must come to rest flat on the table top, neither penetrating nor bouncing, and noslip must prevent
    # any sideways creep or spin away from the drop pose.
    assert_allclose([box.get_pos()[2] for box in boxes], 0.75 * TABLE_Z + 0.02, tol=2e-4)
    assert_allclose([box.get_pos()[:2] for box in boxes], [box.morph.pos[:2] for box in boxes], tol=1e-5)
    assert_allclose([gu.quat_to_xyz(box.get_quat()) for box in boxes], 0.0, tol=2e-5)
    assert_allclose([box.get_vel() for box in boxes], 0.0, tol=1e-4)
    assert_allclose([box.get_ang() for box in boxes], 0.0, tol=5e-4)


@pytest.mark.required
def test_self_collision_sparse_dense_consistency(tol):
    # Pose folding the arm into link2 / left_finger contact: rows coupling two links of the same kinematic tree carry
    # both ancestor chains in their dof support, which the sparse consumers must treat as a set.
    vels = []
    for sparse_solve in (False, True):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                noslip_iterations=2,
                sparse_solve=sparse_solve,
            ),
            show_viewer=False,
        )
        franka = scene.add_entity(
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
        )
        scene.build()
        franka.set_qpos([-0.2326, -1.6055, 1.7372, -2.7745, 0.1091, 0.9083, 0.4493, 0.0384, 0.0258])

        scene.step()

        vels.append(franka.get_dofs_velocity())

    assert_allclose(*vels, tol=tol)
