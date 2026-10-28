# Coding Conventions

## Import Pattern

```python
import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
```

## Naming Conventions

- Classes: `PascalCase` (e.g., `RigidEntity`, `SimOptions`)
- Functions/methods: `snake_case` (e.g., `add_entity`, `get_dofs_position`)
- Constants: `UPPER_CASE` (e.g., `EPS`)
- Private: `_leading_underscore` (e.g., `_initialized`)

## Configuration via Pydantic Options

All configuration uses Pydantic models in `genesis/options/`:

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
    rigid_options=gs.options.RigidOptions(enable_collision=True),
    vis_options=gs.options.VisOptions(show_world_frame=True),
)
```

## Tensor Operations

Genesis uses PyTorch tensors on `gs.device`:

```python
# Always specify device
positions = torch.zeros(n_envs, n_dofs, device=gs.device)

# Convert numpy to torch
torch_tensor = torch.from_numpy(numpy_array).to(gs.device)
```

## Error Handling

```python
# Use Genesis exception for domain errors
gs.raise_exception("Invalid parameter value")

# Use warnings for non-critical issues
gs.logger.warning("Deprecated feature used")
```

## Build Pattern

Always call `scene.build()` before simulation:

```python
scene = gs.Scene()
scene.add_entity(...)  # Add all entities first
scene.build(n_envs=1)  # Compile kernels
scene.step()           # Now safe to step
```

## Backend Selection

```python
import genesis as gs

# CPU backend (default for debug mode)
gs.init(backend=gs.cpu)

# GPU backend (auto-selects CUDA/ROCm/Metal)
gs.init(backend=gs.gpu)

# Explicit precision
gs.init(backend=gs.gpu, precision="32")  # or "64"
```

## Common API Patterns

### Basic Simulation Loop

```python
import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

scene.build()

for _ in range(1000):
    scene.step()
```

### Parallel Environments

```python
import genesis as gs
import torch

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# Build with multiple environments
n_envs = 100
scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

# Control all environments at once
target = torch.zeros(n_envs, robot.n_dofs, device=gs.device)
robot.control_dofs_position(target)
```
