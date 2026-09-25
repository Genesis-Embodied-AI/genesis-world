import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
            enable_rigid_rigid_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.3, 0.6),
            camera_lookat=(0.2, 0.0, 0.3),
        ),
        show_viewer=args.vis,
    )

    # Both FEM and Rigid bodies will be added to IPC for unified contact simulation
    # FEM bodies use StableNeoHookean constitution, Rigid bodies use ABD constitution
    # FEM entities (added to IPC as deformable bodies)
    blob = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.3, 0.0, 0.4),
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean",
        ),
    )

    # Rigid bodies (added to both Genesis rigid solver AND IPC as ABD objects)
    # This enables contact between rigid bodies and FEM bodies through IPC
    rigid_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.4),
            size=(0.1, 0.1, 0.1),
            euler=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            rho=1000,
            friction=0.3,
            coup_type="two_way_soft_constraint",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.2, 0.2, 0.8),
        ),
    )
    scene.build()

    rigid_cube.set_dofs_velocity((4.0, 0, 0, 0, 0, 0))  # Initial velocity in x direction

    # Storage for previous positions to compute velocity
    fem_prev_pos = None
    rigid_prev_pos = None

    # Get FEM blob total mass from material properties
    # For a sphere: mass = (4/3) * pi * r^3 * rho
    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    blob_mass = (4.0 / 3.0) * np.pi * (blob_radius**3) * blob_rho
    cube_mass = tensor_to_array(rigid_cube.get_mass())

    initial_momentum = 4.0 * cube_mass

    # Calculate rigid and fem linear momentum
    def compute_rigid_fem_linear_momentum():
        """Compute total linear momentum of the system."""
        nonlocal fem_prev_pos

        rigid_solver = scene.sim.rigid_solver

        # Rigid body linear momentum.
        cube_vel = tensor_to_array(
            rigid_cube.get_links_vel(links_idx_local=0, ref=gs.link_ref_frame.link_COM)[..., 0, :]
        )
        rigid_linear_momentum = cube_mass * cube_vel

        # FEM body linear momentum.
        # Get FEM vertex data from IPC using libuipc API
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge
        from uipc import builtin

        # Find FEM geometry in IPC scene
        fem_vertex_positions, fem_vertex_masses = None, None
        visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
        for geom_slot in visitor.geometries():
            if isinstance(geom_slot, SimplicialComplexSlot):
                geom = geom_slot.geometry()
                if geom.dim() == 3:  # FEM is 3D
                    meta_attrs = geom.meta()
                    solver_type_attr = meta_attrs.find("solver_type")
                    if not solver_type_attr:
                        continue
                    (solver_type,) = solver_type_attr.view()

                    if solver_type == "fem":
                        # Get geometry (merge instances if needed)
                        if geom.instances().size() >= 1:
                            geom = merge(apply_transform(geom))

                        fem_vertex_positions = geom.positions().view().squeeze(axis=-1)

                        # Get vertex masses (mass = volume * mass_density)
                        volume_attr = geom.vertices().find(builtin.volume)
                        mass_density_attr = geom.vertices().find(builtin.mass_density)
                        if volume_attr and mass_density_attr:
                            volumes = volume_attr.view().reshape(-1)
                            mass_densities = mass_density_attr.view().reshape(-1)
                            fem_vertex_masses = volumes * mass_densities
                        else:
                            # Fallback: uniform mass distribution
                            n_vertices = len(fem_vertex_positions)
                            fem_vertex_masses = np.ones(n_vertices) * (blob_mass / n_vertices)

                        break

        # Compute vertex velocities using finite difference
        if fem_prev_pos is not None:
            fem_vertex_velocities = (fem_vertex_positions - fem_prev_pos) / scene.options.sim.dt
        else:
            # First step: zero velocity
            fem_vertex_velocities = np.zeros_like(fem_vertex_positions)

        # Store current positions for next step
        fem_prev_pos = fem_vertex_positions

        # Compute linear momentum: P = sum(m_i * v_i)
        fem_linear_momentum = np.sum(fem_vertex_masses[:, np.newaxis] * fem_vertex_velocities, axis=0)

        # Also compute average velocity
        assert abs(np.sum(fem_vertex_masses) - blob_mass) < gs.EPS
        fem_vel = fem_linear_momentum / blob_mass

        return rigid_linear_momentum, fem_linear_momentum, cube_vel, fem_vel, cube_mass, blob_mass

    print("\n=== Linear Momentum Conservation Test (Zero Gravity) ===")
    print(f"Rigid cube mass: {cube_mass:.4f} kg")
    print(f"FEM blob mass: {blob_mass:.4f} kg (estimated from V*rho)")
    print("Rigid initial velocity: [4.0, 0, 0, 0, 0, 0] m/s")
    print("FEM initial velocity: [0, 0, 0] m/s")
    print(f"Expected total momentum: [{initial_momentum:.4f}, 0, 0] kg*m/s")

    # Storage for plotting
    time_history = []
    rigid_p_history = []
    fem_p_history = []
    total_p_history = []
    rigid_v_history = []
    fem_v_history = []

    test_time = 0.30  # seconds
    n_steps = int(test_time / scene.options.sim.dt) if "PYTEST_VERSION" not in os.environ else 10
    for i_step in range(n_steps):
        # Compute momentum at every step
        (rigid_p, fem_p, rigid_v, fem_v, rigid_m, fem_m) = compute_rigid_fem_linear_momentum()
        total_p = rigid_p + fem_p

        # Save data for plotting
        time_history.append(i_step * scene.options.sim.dt)
        rigid_p_history.append(rigid_p)
        fem_p_history.append(fem_p)
        total_p_history.append(total_p)
        rigid_v_history.append(rigid_v)
        fem_v_history.append(fem_v)

        # Print every 100 steps
        if i_step % (n_steps // 10) == 0:
            print(f"\n{'=' * 70}")
            print(f"Step {i_step:4d}: t = {i_step * scene.options.sim.dt:.3f}s")
            print(f"{'-' * 70}")
            print(f"Rigid mass:   {rigid_m:8.4f} kg")
            print(f"Rigid velocity: [{rigid_v[0]:9.5f}, {rigid_v[1]:9.5f}, {rigid_v[2]:9.5f}] m/s")
            print(f"Rigid momemtum: [{rigid_p[0]:9.5f}, {rigid_p[1]:9.5f}, {rigid_p[2]:9.5f}] kg*m/s")
            print("")
            print(f"FEM   mass:   {fem_m:8.4f} kg")
            print(f"FEM   velocity: [{fem_v[0]:9.5f}, {fem_v[1]:9.5f}, {fem_v[2]:9.5f}] m/s")
            print(f"FEM   momemtum: [{fem_p[0]:9.5f}, {fem_p[1]:9.5f}, {fem_p[2]:9.5f}] kg*m/s")
            print("")
            print(f"TOTAL momemtum: [{total_p[0]:9.5f}, {total_p[1]:9.5f}, {total_p[2]:9.5f}] kg*m/s")
            print(f"|p_total|: {np.linalg.norm(total_p):.6f} kg*m/s")

        scene.step()

    # Convert lists to numpy arrays for plotting
    time_history = np.stack(time_history, axis=0)
    rigid_p_history = np.stack(rigid_p_history, axis=0)
    fem_p_history = np.stack(fem_p_history, axis=0)
    total_p_history = np.stack(total_p_history, axis=0)
    rigid_v_history = np.stack(rigid_v_history, axis=0)
    fem_v_history = np.stack(fem_v_history, axis=0)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Momentum Conservation Test (Zero Gravity)", fontsize=16)

    # Plot 1: Linear Momentum X-component
    ax = axes[0, 0]
    ax.plot(time_history, rigid_p_history[:, 0], color="tab:red", linestyle="-", label="Rigid px", linewidth=2)
    ax.plot(time_history, fem_p_history[:, 0], color="tab:blue", linestyle="-", label="FEM px", linewidth=2)
    ax.plot(time_history, total_p_history[:, 0], "k", linestyle="--", label="Total px", linewidth=2)
    ax.axhline(y=initial_momentum, color="tab:green", linestyle=":", label=f"Expected: {initial_momentum:.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum (kg*m/s)")
    ax.set_title("Linear Momentum X-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total Momentum Magnitude
    ax = axes[0, 1]
    total_p_mag = np.linalg.norm(total_p_history, axis=1)
    ax.plot(time_history, total_p_mag, "k-", linewidth=2, label="|p_total|")
    ax.axhline(y=initial_momentum, color="tab:red", linestyle="--", label=f"Expected: {initial_momentum:.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|p| (kg*m/s)")
    ax.set_title("Total Momentum Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Velocity X-component
    ax = axes[1, 0]
    ax.plot(time_history, rigid_v_history[:, 0], color="tab:red", linestyle="-", label="Rigid vx", linewidth=2)
    ax.plot(time_history, fem_v_history[:, 0], color="tab:blue", linestyle="-", label="FEM vx", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity X-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Momentum Conservation Error
    ax = axes[1, 1]
    momentum_err_rel = np.linalg.norm(total_p_history - (initial_momentum, 0.0, 0.0), axis=1) / initial_momentum
    ax.plot(time_history, momentum_err_rel, color="tab:red", linestyle="-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (relative)")
    ax.set_title("Momentum Conservation Error (Relative)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Showing the figure blocks until the window is closed, so skip it when nobody is there to close it.
    if "PYTEST_VERSION" not in os.environ:
        plt.show()


if __name__ == "__main__":
    main()
