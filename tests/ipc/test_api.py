from typing import TYPE_CHECKING, cast

import pytest

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)


import genesis as gs
from genesis.engine.couplers.ipc_coupler.data import COUPLING_TYPE

from .utils import (
    get_ipc_rigid_links_idx,
)

if TYPE_CHECKING:
    from genesis.engine.couplers import IPCCoupler


@pytest.mark.required
def test_needs_coup():
    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )
    scene.build()
    assert scene.sim.coupler._coup_type_by_entity == {}
    assert not scene.sim.coupler.has_any_rigid_coupling


@pytest.mark.required
def test_auto_coup_type_from_dofs():
    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            two_way_coupling=True,
        ),
        show_viewer=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    fixed_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
        ),
    )
    free_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0.0, 0.5),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0.0, 0.5, 0.2),
            fixed=True,
        ),
    )
    scene.build()

    coupler = cast("IPCCoupler", scene.sim.coupler)
    assert coupler._coup_type_by_entity[plane] == COUPLING_TYPE.IPC_ONLY
    assert coupler._coup_type_by_entity[fixed_box] == COUPLING_TYPE.IPC_ONLY
    assert coupler._coup_type_by_entity[free_box] == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT
    assert coupler._coup_type_by_entity[robot] == COUPLING_TYPE.EXTERNAL_ARTICULATION
    assert plane.n_joints == 1 and plane.n_dofs == 0
    assert fixed_box.n_joints == 1 and fixed_box.n_dofs == 0
    assert free_box.n_dofs > 0 and not free_box.base_link.is_fixed
    assert robot.n_dofs > 0 and robot.base_link.is_fixed


@pytest.mark.required
def test_link_filter_strict():
    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        show_viewer=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_type="two_way_soft_constraint",
            coup_links=("moving",),
        ),
    )

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    base_link = robot.get_link("base")
    moving_link = robot.get_link("moving")

    assert robot in coupler._coup_links
    assert coupler._coup_links[robot] == {moving_link}

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link.idx in ipc_links_idx
    assert base_link.idx not in ipc_links_idx

    assert moving_link in coupler._abd_slots_by_link
    assert base_link not in coupler._abd_slots_by_link


@pytest.mark.required
@pytest.mark.parametrize("coup_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("merge_fixed_links", [True, False])
def test_find_target_links(coup_type, merge_fixed_links, show_viewer):
    from genesis.engine.couplers.ipc_coupler.utils import find_target_link_for_fixed_merge

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            newton_tolerance=1e-2,
            newton_translation_tolerance=1e-2,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            coup_type="ipc_only",
        ),
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/panda_bullet/panda_nohand.urdf",
            pos=(0, 0, 1),
            fixed=True,
            merge_fixed_links=merge_fixed_links,
        ),
        material=gs.materials.Rigid(
            coup_type=coup_type,
        ),
    )

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    # panda_nohand has joint8 (fixed: link7 -> attachment).
    # With merge_fixed_links=True, attachment is merged into link7.
    # With merge_fixed_links=False, attachment stays separate but IPC should still group them.
    link7 = robot.get_link("link7")
    assert link7 in coupler._abd_slots_by_link

    if not merge_fixed_links:
        attachment = robot.get_link("attachment")
        # attachment exists as separate link but shares ABD body with link7
        target = find_target_link_for_fixed_merge(attachment)
        assert target == link7
        # attachment is NOT in _abd_slots_by_link — only the target link gets a slot entry
        assert attachment not in coupler._abd_slots_by_link

    if coup_type == "external_articulation":
        art_data = coupler._articulation_data_by_entity[robot]
        assert len(art_data.articulation_slots) == 1
        # All 7 revolute joints should be present (fixed joint is skipped)
        assert len(art_data.joints_child_link) == 7


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("enable_rigid_ground_contact", [True, False])
@pytest.mark.parametrize("coup_type", ["ipc_only", "two_way_soft_constraint"])
def test_collision_delegation_ipc_vs_rigid(coup_type, enable_rigid_ground_contact):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_ground_contact=enable_rigid_ground_contact,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )

    # Non-IPC box — always handled by rigid solver
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(1.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )

    if coup_type == "two_way_soft_constraint":
        entity = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
            ),
            material=gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_links=("left_finger", "right_finger"),
            ),
        )

        ipc_excluded_geoms = {geom.idx for name in entity.material.coup_links for geom in entity.get_link(name).geoms}
    else:
        with pytest.raises(gs.GenesisException):
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/go2/urdf/go2.urdf",
                    pos=(0.0, 0.0, 1.0),
                ),
                material=gs.materials.Rigid(
                    coup_type="ipc_only",
                ),
            )

        entity = scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=(0.0, 0.0, 0.6),
            ),
            material=gs.materials.Rigid(
                coup_type="ipc_only",
            ),
        )

        ipc_excluded_geoms = {geom.idx for geom in entity.geoms}

    scene.build()
    assert scene.sim is not None
    assert scene.sim.rigid_solver.collider is not None

    pair_idx = scene.sim.rigid_solver.collider._collision_pair_idx

    # Collect geom indices for entities that should retain rigid solver pairs
    rigid_kept_geoms = {geom.idx for geom in entity.geoms} - ipc_excluded_geoms
    ground_geoms = {plane.geoms[0].idx}
    box_geoms = {box.geoms[0].idx}

    # Non-IPC box always has rigid solver ground pairs
    assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in box_geoms for b in ground_geoms)

    # Pairs between IPC-excluded geoms must have no rigid solver pairs (handled by IPC)
    for i_ga in ipc_excluded_geoms:
        for i_gb in ipc_excluded_geoms:
            if i_ga < i_gb:
                assert pair_idx[i_ga, i_gb] == -1

    if coup_type == "two_way_soft_constraint":
        # Mixed pairs (IPC-excluded ↔ non-IPC) must be kept in rigid solver
        for i_ga in ipc_excluded_geoms:
            for i_gb in box_geoms:
                a, b = min(i_ga, i_gb), max(i_ga, i_gb)
                assert pair_idx[a, b] >= 0

        # IPC-excluded geom ↔ ground must be kept in rigid solver (ground is not IPC-excluded)
        for i_ga in ipc_excluded_geoms:
            for i_gb in ground_geoms:
                a, b = min(i_ga, i_gb), max(i_ga, i_gb)
                assert pair_idx[a, b] >= 0
    else:
        # ipc_only: ALL pairs involving the entity are excluded (IPC fully controls pose)
        for i_ga in ipc_excluded_geoms:
            for i_gb in box_geoms:
                a, b = min(i_ga, i_gb), max(i_ga, i_gb)
                assert pair_idx[a, b] == -1
            for i_gb in ground_geoms:
                a, b = min(i_ga, i_gb), max(i_ga, i_gb)
                assert pair_idx[a, b] == -1

    # Non-excluded rigid geoms (if any) keep rigid solver ground and self-collision pairs
    if rigid_kept_geoms:
        assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in rigid_kept_geoms for b in ground_geoms)
        assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in rigid_kept_geoms for b in rigid_kept_geoms if a < b)


@pytest.mark.required
def test_coup_collision_links():
    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        show_viewer=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_type="two_way_soft_constraint",
            coup_collision_links=("moving",),
        ),
    )
    scene.build()
    assert scene.sim is not None

    # Verify the collision settings were applied
    coupler = cast("IPCCoupler", scene.sim.coupler)
    collision_settings = coupler._coupling_collision_settings[robot]
    base_link = robot.get_link("base")
    moving_link = robot.get_link("moving")

    # "base" should be disabled (not in coup_collision_links), "moving" should not be in the disabled dict
    assert base_link in collision_settings
    assert collision_settings[base_link] is False
    assert moving_link not in collision_settings
