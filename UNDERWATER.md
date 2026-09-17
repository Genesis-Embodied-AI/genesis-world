# Opt-in Akinci SPH boundary particles

Adds `LegacyCouplerOptions.sph_akinci_boundary` (**default `False`**): WCSPH
rigid–fluid coupling via [Akinci et al. 2012](https://doi.org/10.1145/2185520.2185558)
boundary particles.

## Why this is optional

Modern Genesis `main` already has hydrostatic buoyancy via **#2857** in
`LegacyCoupler` (always on for non-fixed links). Akinci is a **second path**
inside `SPHSolver`. With both on, forces **add** and can **over-buoy** dense
bodies. Keep the flag off unless you need boundary particles for research or demos.

| Path | Location | Default |
|------|----------|---------|
| #2857 plane-integral pressure | `legacy_coupler.py` | always on |
| Akinci boundary particles | `sph_solver.py` | **off** |

`legacy_coupler.py` is not modified.

## Enable

```python
import genesis as gs

scene = gs.Scene(
    coupler_options=gs.options.LegacyCouplerOptions(
        rigid_sph=True,
        sph_akinci_boundary=True,
    ),
)
```

## Install from this branch

```bash
# The branch is maintained on the project fork until it is accepted upstream.
git clone https://github.com/robotlearning123/genesis-world.git
cd genesis-world
git switch release/akinci-boundary-upstream-20260721

# Env (Python 3.10–3.13, NVIDIA GPU)
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128  # pick your CUDA
uv pip install -e .
uv pip install nvidia-ml-py setproctitle syrupy pytest-xdist pytest-forked pytest-timeout pytest-print
```

`nvidia-ml-py` is required by Genesis `tests/conftest.py` GPU checks.

## Verify

```bash
export CUDA_VISIBLE_DEVICES=0
python -m pytest \
  tests/coupling/test_sph_akinci_boundary.py \
  tests/coupling/test_sph_rigid.py -m "required and not slow" \
  --backend gpu -n0 -v
# GPU gate: expect 9 passed
```

Demo:

```bash
python examples/underwater/buoyancy_demo.py
```

## Fidelity (honest)

- Surface-area-scaled magnitude (best near body span ≈ 7 particles); does **not**
  converge to Archimedes under refinement.
- Per-geom boundary volume (multi-geom seams can over-buoy).
- WCSPH only (no Akinci force path in the DFSPH substep).
- Not a quantitative hydrodynamics oracle; thrusters should stay analytic.

## Files

| Path | Role |
|------|------|
| `genesis/options/solvers.py` | flag |
| `genesis/engine/solvers/sph_solver.py` | sampling + forces |
| `tests/coupling/test_sph_akinci_boundary.py` | required tests |
| `examples/underwater/` | demos |

## Provenance

Clean-room from Akinci 2012; cross-checked against SPlisHSPlasH (MIT). Ported to
Genesis main after #2857 (`func_apply_coupling_force` order + `dyn_state` access).
