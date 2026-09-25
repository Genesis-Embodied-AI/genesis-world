![Genesis World teaser](https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/HeroShot_Final.png)

# Genesis World

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI Downloads](https://static.pepy.tech/badge/genesis-world)](https://pepy.tech/projects/genesis-world)
[![Documentation](https://app.readthedocs.org/projects/genesis-world/badge/?version=latest)](https://genesis-world.readthedocs.io/en/latest/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/genesis-world)](https://github.com/Genesis-Embodied-AI/genesis-world/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/genesis-world)](https://github.com/Genesis-Embodied-AI/genesis-world/discussions)



**Genesis World** is a simulation platform for physical AI developments. It combines a unified multi-physics engine, a photo-realistic renderer ([Nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx)), and a cross-platform compiler ([Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)) behind a Pythonic simulation interface. Genesis World is designed to scale from a single laptop kernel to datacenter-grade GPUs, while remaining easy to read, extend, and embed in research code.

It was previously named **Genesis** and started as an academic project since Dec 2024, and its development is now officially supported by [Genesis AI](https://www.genesis.ai/).

For more technical details, refer to our [blog post](https://genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward).

## Table of Contents

1. [What is Genesis World?](#what-is-genesis-world)
2. [Catalogue](#catalogue)
3. [Quick Installation](#quick-installation)
4. [Contribution](#contributing-to-genesis)
5. [Support](#support)
6. [License and Acknowledgments](#license-and-acknowledgments)
7. [Citation](#citation)

## What is Genesis World?

![Genesis World stack](https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/diagram_white_lum.png)

Genesis World occupies the four layers inside the dashed box. Above sits whatever you build (robotics environments, ML pipelines, data generation, agentic simulation); below sits whatever compute backend you have.

- **Simulation Interface** — the user-facing API: asset parsing (URDF, MJCF, OBJ, GLB, USD, …), entity accessors, controllers, sensors, parallel and heterogeneous environments, and a built-in GUI.
- **Physics** — a unified multi-physics engine integrating Rigid, FEM, MPM, Particle (PBD / SPH), [uipc](https://github.com/spiriMirror/libuipc), an explicit coupler, and SAP, all sharing one scene and one state.
- **Render** — three rendering paths plug in as camera sensors: **[Nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx)** (our in-house renderer designed for robotics), Luisa (DSL ray tracer), and Pyrender (rasterizer).
- **Compiler** — **[Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)** lowers Python kernel code to CUDA, AMD ROCm, Apple Metal, Vulkan, x86, and ARM64. It carries Genesis's autodiff, GPU graphs, and fastcache machinery.

### Documentation
- [Genesis World](https://genesis-world.readthedocs.io/en/latest/)
- [Quadrants](https://genesis-embodied-ai.github.io/quadrants/index.html)
- [Nyx](https://genesis-embodied-ai.github.io/genesis-nyx/latest/)

## Catalogue

Three sections, mirroring the Genesis layers that ship runnable demos: **Physics** (solvers and multi-solver coupling), **Rendering** (in-repo camera setups plus the Nyx walkthroughs hosted in [genesis-nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx)), and **Simulation Interface** (sensors, GUI, controllers, parallel/heterogeneous envs, and tutorials). Most scripts run end-to-end after `pip install -e ".[dev]"`; demos that depend on optional backends (e.g. the IPC and Nyx examples) need the extras listed in [Optional extras](#optional-extras).

### Physics

| | | |
|---|---|---|
| [Rigid: franka cube](./examples/rigid/franka_cube.py) | [Rigid: collision tower](./examples/collision/tower.py) | [Rigid: contype](./examples/collision/contype.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rigid_franka_cube.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/collision_tower.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/collision_contype.webp" width="240"> |
| [FEM: hard & soft constraint](./examples/deformable/fem_hard_and_soft_constraint.py) | [MPM: tutorial](./examples/tutorials/mpm.py) | [MPM: sand wheel](./examples/coupling/sand_wheel.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/fem_hard_and_soft_constraint.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_mpm.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_sand_wheel.webp" width="240"> |
| [SPH: rigid](./examples/coupling/sph_rigid.py) | [SPH: + MPM](./examples/coupling/sph_mpm.py) | [PBD: liquid](./examples/deformable/pbd_liquid.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_sph_rigid.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_sph_mpm.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/pbd_liquid.webp" width="240"> |
| [PBD: cloth](./examples/tutorials/pbd_cloth.py) | [Stable Fluid: smoke](./examples/fluid/smoke.py) | [IPC: robot cloth teleop](./examples/ipc/ipc_robot_cloth_teleop.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_pbd_cloth.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/smoke.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/ipc_cloth_teleop.webp" width="240"> |
| [Coupler: cloth on rigid](./examples/coupling/cloth_on_rigid.py) | [Coupler: rigid + MPM](./examples/coupling/rigid_mpm_attachment.py) | [Coupler: cut dragon](./examples/coupling/cut_dragon.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_cloth_on_rigid.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_rigid_mpm_attachment.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_cut_dragon.webp" width="240"> |
| [Coupler: water wheel](./examples/coupling/water_wheel.py) | [Coupler: flush cubes](./examples/coupling/flush_cubes.py) | [SAP: Franka grasp rigid cube](./examples/sap_coupling/franka_grasp_rigid_cube.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_water_wheel.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_flush_cubes.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sap_franka_grasp_rigid_cube.webp" width="240"> |
| [Rigid: contact patch](./examples/collision/contact_manifold.py) | | |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/collision_contact_manifold.webp" width="240"> | | |

### Rendering

Genesis exposes three rendering paths as camera sensors: built-in (Nyx / Luisa / Pyrender) and detailed Nyx walkthroughs hosted in [genesis-nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx/tree/main/examples).

| | | |
|---|---|---|
| [Follow entity](./examples/rendering/follow_entity.py) | [Animated camera](./examples/rendering/moving_camera.py) | [Nyx: hello](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/01_hello_nyx.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rendering_follow_entity.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rendering_moving_camera.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_01_hello_nyx.png" width="240"> |
| [Nyx: attached camera](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/02_attached_camera.py) | [Nyx: PBR materials](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/03_materials.py) | [Nyx: light types](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/04_light_types.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_02_attached_camera.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_03_materials.png" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_04_light_types.png" width="240"> |
| [Nyx: 3D Gaussian splat](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/05_gaussian_splat.py) | [Nyx: object picking](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/06_object_picking.py) | [Nyx: multi-cam multi-env](https://github.com/Genesis-Embodied-AI/genesis-nyx/blob/main/examples/07_multi_camera_multi_env.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_05_gaussian_splat.png" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_06_object_picking.png" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/nyx_07_multi_camera_multi_env.png" width="240"> |

### Simulation Interface

| | | |
|---|---|---|
| [Controlling a robot](./examples/tutorials/control_your_robot.py) | [GUI: ImGui joint control](./examples/gui/imgui_joint_control.py) | [Heterogeneous envs](./examples/rigid/heterogeneous_simulation.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_control_your_robot.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/gui_imgui_joint_control.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/het_articulated.webp" width="240"> |
| [Domain randomization](./examples/rigid/domain_randomization.py) | [Sensor: depth camera](./examples/sensors/depth_camera_custom_vverts.py) | [Sensor: IMU](./examples/sensors/imu_franka.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rigid_domain_randomization.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_depth_camera_custom_vverts.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_imu_franka.webp" width="240"> |
| [Sensor: lidar](./examples/sensors/lidar_teleop.py) | [Sensor: tactile sandbox](./examples/sensors/tactile_sandbox.py) | [Sensor: contact force](./examples/sensors/contact_force_go2.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_lidar_teleop.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_tactile_sandbox.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_contact_force_go2.webp" width="240"> |
| [Sensor: surface distance](./examples/sensors/surface_distance_shadowhand.py) | [Sensor: temperature grid](./examples/sensors/temperature_grid.py) | [GUI: debug drawing](./examples/tutorials/draw_debug.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_surface_distance_shadowhand.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sensors_temperature_grid.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_draw_debug.webp" width="240"> |
| [GUI: mesh point picker](./examples/viewer_plugin/mesh_point_selector.py) | [GUI: mouse interaction](./examples/viewer_plugin/mouse_interaction.py) | [Diff-IK controller](./examples/rigid/diffik_controller.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/viewer_mesh_point_selector.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/viewer_mouse_interaction.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rigid_diffik_controller.webp" width="240"> |
| [Batched IK](./examples/tutorials/batched_IK.py) | [Drone](./examples/drone/hover_train.py) | [Advanced: worm](./examples/tutorials/advanced_worm.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_batched_IK.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/drone_hover_train.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_advanced_worm.webp" width="240"> |

## Quick Installation

### Using pip

Install **PyTorch** first following the [official instructions](https://pytorch.org/get-started/locally/).

Then, install Genesis via PyPI:
```bash
pip install genesis-world  # Requires Python>=3.10,<3.14;
```

For the latest version to date, make sure that `pip` is up-to-date via `pip install --upgrade pip`, then run command:
```bash
pip install git+https://github.com/Genesis-Embodied-AI/genesis-world.git
```
Note that the package must still be updated manually to sync with main branch.

Users seeking to contribute are encouraged to install Genesis in editable mode. First, make sure that `genesis-world` has been uninstalled, then clone the repository and install locally:
```bash
git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
cd genesis-world
pip install -e ".[dev]"
```
It is recommended to systematically execute `pip install -e ".[dev]"` after moving HEAD to make sure that all dependencies and entrypoints are up-to-date.

### Optional extras

| | |
|---|---|
| IPC solver (uipc backend) | `pip install pyuipc` *(Linux / Windows x86, NVIDIA GPU)* |
| Nyx renderer | `pip install gs-nyx` — see [genesis-nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx) |

Quadrants is bundled with Genesis automatically; no extra install. The standalone wheel (`pip install quadrants`) is documented at [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) for users who want the compiler outside Genesis.

### Using uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.

**Install uv:**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Quick start with uv:**
```bash
git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
cd genesis-world
uv sync
```

Then install PyTorch for your platform:

```bash
# NVIDIA GPU (CUDA 12.6 as an example)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only (Linux/Windows)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (Metal/MPS)
uv pip install torch
```

Run an example:
```bash
uv run examples/rigid/single_franka.py
```

## Contributing to Genesis

The Genesis project is an open and collaborative effort. We welcome all forms of contributions from the community, including:

- **Pull requests** for new features or bug fixes.
- **Bug reports** through GitHub Issues.
- **Suggestions** to improve Genesis's usability.

Refer to our [contribution guide](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/.github/contributing/PULL_REQUESTS.md) for more details.

## Support

- Report bugs or request features via GitHub [Issues](https://github.com/Genesis-Embodied-AI/genesis-world/issues).
- Join discussions or ask questions on GitHub [Discussions](https://github.com/Genesis-Embodied-AI/genesis-world/discussions).

## License and Acknowledgments

The Genesis source code is licensed under Apache 2.0.

Genesis's development has been made possible thanks to these open-source projects:

- [Taichi](https://github.com/taichi-dev/taichi): the original compiler that [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) forked from in June 2025. Kudos to the Taichi team for their technical support over the years.
- [libuipc](https://github.com/spiriMirror/libuipc): IPC solver backend.
- [FluidLab](https://github.com/zhouxian/FluidLab): Reference MPM solver implementation.
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi): Reference SPH solver implementation.
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) and [PBF3D](https://github.com/WASD4959/PBF3D): Reference PBD solver implementations.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Reference for rigid body dynamics.
- [libccd](https://github.com/danfis/libccd): Reference for collision detection.
- [PyRender](https://github.com/mmatl/pyrender): Rasterization-based renderer.
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [LuisaRender](https://github.com/LuisaGroup/LuisaRender): Ray-tracing DSL.
- [Madrona](https://github.com/shacklettbp/madrona) and [Madrona-mjx](https://github.com/shacklettbp/madrona_mjx): Batch renderer backend

## Citation

If you use Genesis in your research, please consider citing:

```bibtex
@article{
   genesis2026genesisworld,
   author = {Genesis AI Team},
   title = {The Role of Simulation in Scalable Robotics, Genesis World 1.0, and the Path Forward},
   journal = {Genesis AI Blog},
   month = {May},
   year = {2026},
   url = {https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward},
}
```
```bibtex
@misc{
  Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Generative and Universal Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/genesis-world}
}
```
<!--
Catalogue entries pruned from the Physics grid. Kept here as a reference so
they can be reinstated later. The links and thumbnail paths are all still
valid in the repo; just paste any pair of rows back into the Physics table.

| [Rigid: grasp bottle](./examples/rigid/grasp_bottle.py) | [Rigid: collision pyramid](./examples/collision/pyramid.py) | [FEM: elastic dragon](./examples/deformable/elastic_dragon.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rigid_grasp_bottle.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/collision_pyramid.png" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/elastic_dragon.webp" width="240"> |
| [FEM: SAP fixed constraint](./examples/sap_coupling/fem_fixed_constraint.py) | [SPH: liquid](./examples/tutorials/sph_liquid.py) | [Coupler: grasp soft cube](./examples/coupling/grasp_soft_cube.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sap_fem_fixed_constraint.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_sph_liquid.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_grasp_soft_cube.webp" width="240"> |
| [Coupler: cloth + rigid](./examples/coupling/cloth_attached_to_rigid.py) | [SAP: Franka grasp FEM sphere](./examples/sap_coupling/franka_grasp_fem_sphere.py) | [SAP: FEM sphere + cube](./examples/sap_coupling/fem_sphere_and_cube.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/coupling_cloth_attached_to_rigid.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sap_franka_grasp_fem_sphere.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/sap_fem_sphere_and_cube.webp" width="240"> |

Pruned from Simulation Interface (same logic — labels/paths still valid):

| [Entity name](./examples/tutorials/entity_name.py) | [Select rendered envs](./examples/tutorials/selecting_rendered_envs.py) | [GUI: keyboard teleop](./examples/viewer_plugin/keyboard_teleop.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_entity_name.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_selecting_rendered_envs.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/keyboard_teleop.webp" width="240"> |
| [Control franka](./examples/rigid/control_franka.py) | [Position control comparison](./examples/tutorials/position_control_comparison.py) | [IK + motion planning](./examples/tutorials/IK_motion_planning_grasp.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rigid_control_franka.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_position_control_comparison.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_IK_motion_planning_grasp.webp" width="240"> |
| [Close kinematic chain](./examples/rigid/closed_loop.py) | [Advanced: muscle](./examples/tutorials/advanced_muscle.py) | [Advanced: hybrid robot](./examples/tutorials/advanced_hybrid_robot.py) |
| <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/rigid_closed_loop.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_advanced_muscle.webp" width="240"> | <img src="https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/readme-assets/videos/tutorials_advanced_hybrid_robot.webp" width="240"> |
-->
