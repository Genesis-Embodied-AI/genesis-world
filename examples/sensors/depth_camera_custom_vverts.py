"""Depth cameras attached to rigid and kinematic links, with per-frame set_vverts on a deforming sphere."""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.utils.image_exporter import FrameImageExporter
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser(description="Multi-solver depth camera demo")
    parser.add_argument("-v", "--vis", action="store_true", help="Open Genesis 3D viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-b", "--num-envs", type=int, default=0, help="Number of parallel envs (0 = unbatched)")
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("--save-every", type=int, default=10, help="Save depth PNG every N steps")
    parser.add_argument("-o", "--output-dir", type=str, default="out/depth", help="Directory for saved depth PNGs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("depth_*.png"):
        old.unlink()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
        ),
        show_viewer=args.vis,
    )

    # Rigid entities (RigidSolver)
    plane = scene.add_entity(gs.morphs.Plane())

    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.42),
        ),
    )

    # Kinematic entities (KinematicSolver)
    # 1) Deforming sphere - vertices are pushed every frame via set_vverts. Pre-translated to the robot's right so the
    # mesh sits at the desired world pose with morph.pos at the origin (the morph applies its own offset on top).
    sphere_tri = trimesh.creation.icosphere(subdivisions=3, radius=0.25)
    sphere_verts = sphere_tri.vertices.astype(np.float32) + np.array([1.0, 0.0, 0.5], dtype=np.float32)
    sphere_tri = trimesh.Trimesh(vertices=sphere_verts, faces=sphere_tri.faces, process=False)

    kin_sphere = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(sphere_tri,),
            pos=(0, 0, 0),
            fixed=True,
            enable_custom_vverts=True,
        ),
        material=gs.materials.Kinematic(
            use_visual_raycasting=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.8, 0.4),
        ),
    )

    # 2) Static box - FK-driven, visible to both depth cameras like the sphere.
    box_tri = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    box_tri.vertices += np.array([0.0, 1.0, 0.15], dtype=np.float32)

    kin_box = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(box_tri,),
            pos=(0, 0, 0),
            fixed=True,
        ),
        material=gs.materials.Kinematic(
            use_visual_raycasting=True,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.3, 0.3),
        ),
    )

    # Depth camera sensors - on different solvers
    max_range = 5.0
    cam_res = (96, 72)

    cams = {
        # Camera 1: mounted on Go2 base link (RigidSolver entity)
        "robot": scene.add_sensor(
            gs.sensors.DepthCamera(
                pattern=gs.sensors.DepthCameraPattern(
                    res=cam_res,
                    fov_horizontal=90.0,
                ),
                entity_idx=go2.idx,
                link_idx_local=0,
                pos_offset=(0.3, 0.0, 0.1),
                euler_offset=(0.0, 0.0, 0.0),
                max_range=max_range,
                return_world_frame=True,
            ),
        ),
        # Camera 2: mounted on the plane (RigidSolver, world-fixed, third-person)
        "world": scene.add_sensor(
            gs.sensors.DepthCamera(
                pattern=gs.sensors.DepthCameraPattern(
                    res=cam_res,
                    fov_horizontal=60.0,
                ),
                entity_idx=plane.idx,
                link_idx_local=0,
                pos_offset=(-1.5, 0.0, 1.5),
                euler_offset=(0.0, 45.0, 0.0),
                max_range=max_range,
                return_world_frame=True,
            ),
        ),
    }

    if args.num_envs > 0:
        scene.build(n_envs=args.num_envs)
    else:
        scene.build()

    # Set Go2 to a standing pose
    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    standing_angles = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]
    dofs_idx = [go2.get_joint(name).dofs_idx_local[0] for name in joint_names]
    go2.set_dofs_position(standing_angles, dofs_idx)

    exporter = FrameImageExporter(str(out_dir), depth_clip_max=max_range)

    B = max(1, args.num_envs)
    print("=" * 65)
    print("Scene entities:")
    print(f"  [RIGID]      plane       idx={plane.idx}")
    print(f"  [RIGID]      go2         idx={go2.idx}  (articulated, {go2.n_links} links)")
    print(f"  [KINEMATIC]  kin_sphere   idx={kin_sphere.idx}  (deforms each step)")
    print(f"  [KINEMATIC]  kin_box      idx={kin_box.idx}")
    print()
    print("Depth cameras:")
    for i, name in enumerate(cams):
        print(f"  cam{i}={name}")
    print()
    print(f"Output: {out_dir}/depth_cam<i>_env<env>_<step>.png")
    print(f"  (view live:  feh --reload 0.1 {out_dir})")
    print("=" * 65)
    print()

    # Wave-deform parameters; per-env time offset gives each environment a distinct animation phase.
    amplitude, freq = 0.08, 4.0

    try:
        for step in range(args.steps):
            t = step * 0.05
            ts = t + 0.5 * np.arange(B, dtype=np.float32)
            deformed = np.broadcast_to(sphere_verts, (B, *sphere_verts.shape)).copy()
            deformed[..., 2] += (
                amplitude
                * np.sin(freq * sphere_verts[:, 0] + ts[:, None])
                * np.cos(freq * sphere_verts[:, 1] + 0.7 * ts[:, None])
            )
            deformed[..., 0] += amplitude * 0.5 * np.sin(freq * sphere_verts[:, 2] + 1.3 * ts[:, None])
            kin_sphere.set_vverts(deformed if args.num_envs > 0 else deformed[0])

            scene.step()

            depth_imgs = {name: cam.read_image() for name, cam in cams.items()}

            if step % args.save_every == 0:
                for i, name in enumerate(cams):
                    exporter.export_frame_single_camera(step, i, depth=tensor_to_array(depth_imgs[name]))

            if step % 50 == 0:
                hits = {
                    name: int((torch.isfinite(img) & (img < max_range)).sum().item())
                    for name, img in depth_imgs.items()
                }
                summary = "  ".join(f"cam_{name} hits={n:5d}" for name, n in hits.items())
                print(f"  step {step:4d}:  {summary}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)

    print(f"\nDone! Depth PNGs at: {out_dir}")


if __name__ == "__main__":
    main()
