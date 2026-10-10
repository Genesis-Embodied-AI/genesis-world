"""Loop closure in the MochiSolver: a four-bar linkage closed by a connect equality constraint and driven at the crank,
next to a pair of boxes welded together.

Equality constraints are enforced as stiff penalties of the implicit Newton solve (`MochiOptions.equality_stiffness`),
so the closing joint of the linkage holds within a fraction of a millimeter while the mechanism turns.
"""

import argparse
import os
import tempfile

import numpy as np

import genesis as gs

FOUR_BAR = """<mujoco>
  <worldbody>
    <body name="crank" pos="0 0 1">
      <joint name="crank_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.012" mass="0.2"/>
      <body name="coupler" pos="0 0 0.1">
        <joint name="coupler_hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0.1" size="0.01" mass="0.4"/>
      </body>
    </body>
    <body name="rocker" pos="0.4 0 1">
      <joint name="rocker_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.012" mass="0.3"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="coupler" body2="rocker" anchor="0.4 0 0.1"/>
  </equality>
</mujoco>
"""

WELDED_BOXES = """<mujoco>
  <worldbody>
    <body name="a" pos="0 0.6 0.5"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>
    <body name="b" pos="0.3 0.6 0.5" euler="0 30 0"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>
  </worldbody>
  <equality>
    <weld body1="a" body2="b" anchor="0.15 0 0"/>
  </equality>
</mujoco>
"""


def write_model(directory, name, xml):
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(xml)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-n", "--n_steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="64")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 120.0, gravity=(0.0, 0.0, -9.8)),
        mochi_options=gs.options.MochiOptions(n_newton_iterations=8),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.5, -2.5, 1.5), camera_lookat=(0.2, 0.3, 0.7)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
    with tempfile.TemporaryDirectory() as directory:
        linkage = scene.add_entity(
            gs.morphs.MJCF(file=write_model(directory, "four_bar.xml", FOUR_BAR)),
            material=gs.materials.Mochi.Rigid(),
            surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)),
        )
        welded = scene.add_entity(
            gs.morphs.MJCF(file=write_model(directory, "welded.xml", WELDED_BOXES)),
            material=gs.materials.Mochi.Rigid(),
            surface=gs.surfaces.Default(color=(0.9, 0.5, 0.3)),
        )
        if args.record:
            cam = scene.add_camera(
                res=(640, 360),
                pos=(1.5, -2.5, 1.5),
                lookat=(0.2, 0.3, 0.7),
            )
        scene.build()

    equality = linkage.equalities[0]
    coupler, rocker = (
        linkage.links[equality.eq_obj1id - linkage.link_start],
        linkage.links[equality.eq_obj2id - linkage.link_start],
    )
    anchors = np.asarray(equality.eq_data, dtype=np.float64)
    crank_dof = linkage.get_joint("crank_hinge").dofs_idx_local

    if args.record:
        cam.start_recording(save_to_filename="loop_closure.mp4", fps=30)
    for i_step in range(args.n_steps):
        # Constant torque at the crank turns the mechanism.
        linkage.control_dofs_force(np.array([0.3]), crank_dof)
        scene.step()
        if i_step % 60 == 0:
            p_a = coupler.get_pos().cpu().numpy() + gs.utils.geom.transform_by_quat(
                anchors[0:3], coupler.get_quat().cpu().numpy()
            )
            p_b = rocker.get_pos().cpu().numpy() + gs.utils.geom.transform_by_quat(
                anchors[3:6], rocker.get_quat().cpu().numpy()
            )
            angle = float(linkage.get_qpos()[0])
            print(
                f"step {i_step:4d}: crank angle={angle:7.3f} rad, closing joint gap={np.linalg.norm(p_a - p_b):.2e} m, "
                f"welded pair z={float(welded.links[0].get_pos()[2]):.3f} m"
            )
        if args.record:
            cam.render()
    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    main()
