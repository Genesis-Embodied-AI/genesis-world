# Newton Coupling Examples

The first-version runtime couples `FEM.QCloth`, the existing `RigidSolver`,
and Consistent IPC contact behind the normal `Scene.step()` interface:

```python
scene = gs.Scene(
    coupler_options=gs.options.NewtonCouplerOptions(
        contact_d_hat=1e-3,
        contact_friction_mu=1.0,
        contact_resistance=1e4,
    )
)
```

It requires a GPU, double precision, the Quadrants ndarray backend, one
unbatched environment, and one substep. The engine is built lazily on the
first `Scene.step()`, after post-build qpos, controller, and QCloth constraint
setup. Cloth-only scenes use `FEMOptions.floor_height` as an analytical
halfplane. Mixed Rigid–QCloth scenes should author fixed rigid support geometry
so the halfplane does not also collide with the robot.

```powershell
python examples/newton_coupling/franka_cloth_grasp.py -v
python examples/newton_coupling/cloth_stack.py -v
```

`franka_cloth_grasp.py` controls the center between the two fingertip pads.
The Panda `hand` link origin is approximately 10.34 cm behind that point and
is converted internally before every IK solve.

The first version pins
[Quadrants `974a8975b`](https://github.com/alanray-tech/quadrants/commit/974a8975b24518e44618c0c071d9202df6d9f693)
for checkpoint-contained graph parallel regions. Published
`quadrants==1.3.3` cannot compile this pipeline. Benchmark reports must record
the exact compiler commit used; no checkout name or directory layout is
assumed.
