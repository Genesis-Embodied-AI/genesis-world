import os
import subprocess
import sys

import numpy as np
import pytest

# A multi-threaded quadrants CPU runtime is configured at gs.init, so each thread count runs in its own process:
# the script below steps a cloth draped over a box with the pipeline step kernel (whose item loops parallelize at
# more than one thread) and dumps the trajectory. Parallel loops accumulate atomics in a nondeterministic order,
# so the trajectories agree to a tolerance, not bit-exactly.
_SCRIPT = """
import sys

import numpy as np
import trimesh

import genesis as gs

out_path, cpu_threads = sys.argv[1], int(sys.argv[2])
gs.init(backend=gs.cpu, precision="64", logging_level="warning", cpu_threads=cpu_threads)

n_cells, size = 12, 0.8
axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
X, Y = np.meshgrid(axis, axis, indexing="ij")
verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
faces = []
for i in range(n_cells):
    for j in range(n_cells):
        a = i * (n_cells + 1) + j
        faces.append([a, a + 1, a + n_cells + 2])
        faces.append([a, a + n_cells + 2, a + n_cells + 1])
obj_path = out_path + ".obj"
trimesh.Trimesh(verts, np.array(faces), process=False).export(obj_path)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
    mochi_options=gs.options.MochiOptions(n_newton_iterations=8, step_kernel="pipeline"),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane(), material=gs.materials.Mochi.Rigid())
scene.add_entity(gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 0.15), fixed=True), material=gs.materials.Mochi.Rigid())
cloth = scene.add_entity(
    gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 0.6)),
    material=gs.materials.Mochi.Shell(
        E=2e4, nu=0.3, rho=200.0, thickness=2e-3, friction=0.6, collider_radius=0.02, penalty_coefficient=1e7
    ),
)
scene.build()
expected = gs.PARA_LEVEL.PARTIAL if cpu_threads > 1 else gs.PARA_LEVEL.NEVER
assert scene.mochi_solver._resolve_para_level() == expected
positions = []
for _ in range(40):
    scene.step()
    positions.append(cloth.get_vertices_position().cpu().numpy())
np.save(out_path, np.array(positions))
"""


@pytest.mark.required
@pytest.mark.precision("64")
def test_threaded_pipeline_matches_single_thread(tmp_path, backend):
    if backend != "cpu":
        pytest.skip("CPU threading only exists on the CPU backend")
    script_path = os.path.join(tmp_path, "run_threads.py")
    with open(script_path, "w") as handle:
        handle.write(_SCRIPT)
    trajectories = {}
    for cpu_threads in (1, 8):
        out_path = os.path.join(tmp_path, f"traj_{cpu_threads}.npy")
        env = {key: value for key, value in os.environ.items() if key != "QD_NUM_THREADS"}
        subprocess.run(
            [sys.executable, script_path, out_path, str(cpu_threads)],
            check=True,
            env=env,
            timeout=1200,
        )
        trajectories[cpu_threads] = np.load(out_path)
    np.testing.assert_allclose(trajectories[8], trajectories[1], rtol=1e-6, atol=1e-6)
