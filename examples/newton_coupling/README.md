# Newton Coupling Examples

The first-version runtime couples `FEM.QCloth`, the existing `RigidSolver`,
and Consistent IPC contact behind the normal `Scene.step()` interface:

```python
scene = gs.Scene(
    engine_options=gs.options.NewtonEngineOptions(
        contact_d_hat=1e-3,
        contact_friction_mu=1.0,
        contact_resistance=1e4,
    )
)
```

It requires a GPU, double precision, the Quadrants ndarray backend, one
unbatched environment, and one substep. The engine is constructed and compiled
by `Scene.build()`. Post-build qpos, controller, and QCloth constraint changes
are synchronized into that existing engine before the next step. Cloth-only
scenes use `FEMOptions.floor_height` as an analytical
halfplane. Mixed Rigid–QCloth scenes should author fixed rigid support geometry
so the halfplane does not also collide with the robot.

```powershell
python examples/newton_coupling/franka_cloth_grasp.py -v
python examples/newton_coupling/cloth_stack.py -v
```

`franka_cloth_grasp.py` controls the center between the two fingertip pads.
The Panda `hand` link origin is approximately 10.34 cm behind that point and
is converted internally before every IK solve.

The base package retains the published `quadrants==1.3.3` wheel so Genesis
remains installable on every supported platform. The Newton examples currently
require the checkpoint-contained graph-parallel patch validated at
[Quadrants `73920ffc`](https://github.com/alanray-tech/quadrants/commit/73920ffc15ee7dfc55227bbf4283d2a81c9a3aa4);
use a compatible prebuilt wheel or local Quadrants build when running this
experimental engine. Benchmark reports must record the exact compiler commit
used; no checkout name or directory layout is assumed.
