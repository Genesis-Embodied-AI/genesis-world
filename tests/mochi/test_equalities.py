import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.8), **mochi_kwargs):
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=gs.options.MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _write_mjcf(path, worldbody, equality):
    path.write_text(
        f"""<mujoco>
  <worldbody>
    {worldbody}
  </worldbody>
  <equality>
    {equality}
  </equality>
</mujoco>
"""
    )
    return str(path)


def _anchor_gap(entity, equality):
    """World distance between the two anchors of a connect or weld constraint."""
    data = np.asarray(equality.eq_data, dtype=np.float64)
    anchors = (data[0:3], data[3:6])
    if equality.type == gs.EQUALITY_TYPE.WELD:
        anchors = (data[3:6], data[0:3])
    points = []
    for link_idx, anchor in zip((equality.eq_obj1id, equality.eq_obj2id), anchors):
        link = entity.links[link_idx - entity.link_start]
        pos = tensor_to_array(link.get_pos())
        quat = tensor_to_array(link.get_quat())
        points.append(pos + gu.transform_by_quat(anchor, quat))
    return np.linalg.norm(points[0] - points[1])


@pytest.mark.precision("64")
def test_connect_equality(tmp_path, show_viewer):
    xml_path = _write_mjcf(
        tmp_path / "connect.xml",
        """<body name="a" pos="0 0 0.5"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>
    <body name="b" pos="0.3 0 0.5"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>""",
        '<connect body1="a" body2="b" anchor="0.15 0 0"/>',
    )
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    pair = scene.add_entity(gs.morphs.MJCF(file=xml_path), material=gs.materials.Mochi.Rigid())
    scene.build()
    assert pair.n_equalities == 1
    assert scene.mochi_solver.mochi_config.has_equalities
    equality = pair.equalities[0]
    # Kick the second box upwards: the hinge point must follow the first box while the pair falls and lands.
    vel = np.zeros(pair.n_dofs)
    vel[6:9] = (0.0, 0.0, 1.0)
    pair.set_dofs_velocity(vel)
    gaps = []
    for _ in range(150):
        scene.step()
        gaps.append(_anchor_gap(pair, equality))
    assert max(gaps) < 2e-3
    assert gaps[-1] < 2e-4
    assert_allclose(pair.get_dofs_velocity(), 0.0, atol=2e-3)
    # The two boxes rest on the ground in contact with each other through the hinge.
    for link in pair.links:
        assert_allclose(float(tensor_to_array(link.get_pos())[2]), 0.05, atol=2e-3)


@pytest.mark.precision("64")
def test_weld_equality(tmp_path, show_viewer):
    xml_path = _write_mjcf(
        tmp_path / "weld.xml",
        """<body name="a" pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>
    <body name="b" pos="0.3 0 1"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>""",
        '<weld body1="a" body2="b" anchor="0.15 0 0"/>',
    )
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, gravity=(0.0, 0.0, 0.0), n_newton_iterations=8)
    pair = scene.add_entity(gs.morphs.MJCF(file=xml_path), material=gs.materials.Mochi.Rigid())
    scene.build()
    equality = pair.equalities[0]
    assert equality.type == gs.EQUALITY_TYPE.WELD
    quat_rel_0 = gu.transform_quat_by_quat(
        tensor_to_array(pair.links[1].get_quat()), gu.inv_quat(tensor_to_array(pair.links[0].get_quat()))
    )
    # Spin the first box: the welded pair tumbles as one rigid body.
    vel = np.zeros(pair.n_dofs)
    vel[3:6] = (0.0, 2.0, 1.0)
    pair.set_dofs_velocity(vel)
    for _ in range(90):
        scene.step()
        assert _anchor_gap(pair, equality) < 2e-3
    quat_rel = gu.transform_quat_by_quat(
        tensor_to_array(pair.links[1].get_quat()), gu.inv_quat(tensor_to_array(pair.links[0].get_quat()))
    )
    assert np.linalg.norm(gu.quat_to_rotvec(gu.transform_quat_by_quat(quat_rel, gu.inv_quat(quat_rel_0)))) < 2e-2
    # Both boxes rotate together; the spin of the first box is shared with the whole (much larger) inertia of the pair.
    ang = tensor_to_array(pair.get_dofs_velocity())
    assert_allclose(ang[3:6], ang[9:12], atol=5e-2)
    assert np.linalg.norm(ang[3:6]) > 0.1


@pytest.mark.precision("64")
def test_joint_equality(tmp_path, show_viewer):
    xml_path = _write_mjcf(
        tmp_path / "joint.xml",
        """<body name="l1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
      <body name="l2" pos="0 0 -0.3">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1"/>
      </body>
    </body>""",
        '<joint joint1="j2" joint2="j1" polycoef="0 1 0 0 0"/>',
    )
    scene = _mochi_scene(show_viewer, 1.0 / 120.0, n_newton_iterations=8)
    pendulum = scene.add_entity(gs.morphs.MJCF(file=xml_path), material=gs.materials.Mochi.Rigid())
    scene.build()
    assert pendulum.equalities[0].type == gs.EQUALITY_TYPE.JOINT
    pendulum.set_qpos(np.array([0.6, 0.6]))
    q_max = 0.0
    for _ in range(240):
        scene.step()
        q = tensor_to_array(pendulum.get_qpos())
        # The coupling keeps the second joint at the angle of the first one while the pendulum swings.
        assert abs(q[1] - q[0]) < 5e-3
        q_max = max(q_max, abs(q[0]))
    assert q_max > 0.3
