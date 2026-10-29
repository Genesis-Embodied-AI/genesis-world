# Genesis Architecture

## Project Structure

```
Genesis/
├── genesis/                    # Main source code
│   ├── __init__.py            # Entry point, gs.init(), global state
│   ├── engine/                # Core simulation engine
│   │   ├── scene.py           # Scene class - main API entry point
│   │   ├── simulator.py       # Manages all solvers
│   │   ├── entities/          # Entity types (RigidEntity, MPMEntity, etc.)
│   │   ├── solvers/           # Physics solvers
│   │   │   ├── rigid/         # Rigid body solver
│   │   │   ├── mpm_solver.py  # Material Point Method
│   │   │   ├── sph_solver.py  # Smoothed Particle Hydrodynamics
│   │   │   ├── fem_solver.py  # Finite Element Method
│   │   │   ├── pbd_solver.py  # Position Based Dynamics
│   │   │   └── sf_solver.py   # Stable Fluid
│   │   └── couplers/          # Inter-solver coupling
│   ├── options/               # Configuration classes (Pydantic models)
│   │   ├── materials.py       # Physical laws, independent of the solver
│   │   ├── morphs.py          # Shape definitions (Box, Mesh, URDF, MJCF)
│   │   ├── solvers.py         # Solver configuration options
│   │   └── surfaces.py        # Surface properties
│   ├── vis/                   # Visualization (Visualizer, Camera, Viewer)
│   ├── sensors/               # Sensor systems (camera, IMU, etc.)
│   └── utils/                 # Utilities (mesh, geometry, etc.)
├── tests/                     # Test files
├── examples/                  # Example scripts
└── genesis/assets/            # Built-in meshes, URDFs, textures
```

## Core Components Flow

```
gs.init() → Scene → Simulator → Solvers → Entities
                 ↓
            Visualizer → Viewer / Cameras
```

## Entities

Entities are physical objects in the simulation:

| Entity Type | Solver | Use Case |
|-------------|--------|----------|
| `RigidEntity` | Rigid | Robots, rigid objects |
| `MPMEntity` | MPM | Deformable solids, granular materials |
| `SPHEntity` | SPH | Liquids, fluids |
| `FEMEntity` | FEM | Finite element deformable bodies |
| `PBD2DEntity`, `PBD3DEntity` | PBD | Cloth, soft bodies |
| `DroneEntity` | Rigid | Quadcopters with aerodynamics |

Location: `genesis/engine/entities/`

### Descriptions

An entity is created from a description: a dataclass holding everything its solver simulates it with, defined beside the entity that consumes it (`genesis/engine/entities/rigid_entity/description.py` for rigid and kinematic entities). Creation is two steps, each owned by the class it concerns:

- the description class resolves the morphs, material and surface into a description (`RigidEntityDescription.resolve`), which is where assets are read and post-processed;
- the entity class builds itself from the description alone (`RigidEntity.__init__`), so a description loaded from a file and one resolved from assets take the same path.

`Scene.export` writes the description of every entity, obtained through `Entity.desc`, beside the scene options. `Scene.load` creates each entity from its description through the solver its entity options name (`Solver.add_entity` given `desc`), attachments included, since a description names only entities created before it. A kind of entity gains export support by defining its description class, returning it from `desc`, and having its solver's `add_entity` build from a given description, with no change to the scene. The serialization facility alone reduces every absolute path a value holds to the name of the asset, so no option or mesh redacts anything itself.

## Morphs

Morphs define geometry and initial pose (solver-agnostic):

```python
# Primitives
gs.morphs.Box(size=(1, 1, 1), pos=(0, 0, 0.5))
gs.morphs.Sphere(radius=0.5, pos=(0, 0, 1))
gs.morphs.Plane()

# Robot descriptions
gs.morphs.URDF(file="path/to/robot.urdf", fixed=True)
gs.morphs.MJCF(file="path/to/robot.xml")
```

Location: `genesis/options/morphs.py`

## Materials

A material states a physical law and its parameters, whichever solver simulates it. Each entity options class admits the materials its solver simulates, and the numerical choices of that solver for the entity go in its `solver_options`. A material field left to None takes the default of the solver, which each entity options class states once (such as `FEMEntityOptions.material_params`), and the entity options raise on a field their solver cannot honour.

```python
gs.materials.Material(density=1000)
gs.materials.Elastic(youngs_modulus=1e5, poisson_ratio=0.3)
gs.materials.Liquid(viscosity=0.01)
gs.materials.Cloth(thickness=0.001)
```

Location: `genesis/options/materials.py`

## Entity options

An entity is constructed from typed options, whose class decides the kind of entity and the solver simulating it: each class names the options class of its solver (`EntityOptions._solver_options_cls`), and holds the options of that solver specific to the entity (`solver_options`, such as `gs.options.MPMEntitySolverOptions`). `HybridEntityOptions` names no solver, since a hybrid entity adds a rigid entity and an MPM entity of their own. The flattened arguments of `Scene.add_entity` are a shorthand for `RigidEntityOptions`, so the material never changes the kind of entity.

```python
scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))  # rigid
scene.add_entity(options=gs.options.KinematicEntityOptions(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1))))
scene.add_entity(options=gs.options.MPMEntityOptions(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)), material=gs.materials.Elastic()))
scene.add_entity(options=gs.options.SPHEntityOptions(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)), solver_options=gs.options.SPHEntitySolverOptions(sampler="pbs")))
```

Rigid and kinematic entity options hold `links`, rigid ones `geoms` as well, and FEM entity options hold `vertices`: mappings from a selection (`gs.LinkIndex`, `gs.GeomIndex`, `gs.VertexIndex`) to `LinkOptions`, `GeomOptions` or `VertexOptions`. They carry only the settings the solver already holds at that granularity: the density and friction of a geom, the coupling and the visual raycasting of a link, and the fixed vertices of an FEM entity. A geom option overrides a link option, which overrides the entity, and two selections of equal specificity giving one field of one target raise an error. A `GeomIndex` counts the collision geoms of a link as the asset states them, so the rigid description applies the options before post-processing splits or fuses the geoms.

```python
scene.add_entity(
    options=gs.options.RigidEntityOptions(
        morph=gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf"),
        links={"arm2": gs.options.LinkOptions(material=gs.materials.Material(friction=1.2))},
    )
)
```

Location: `genesis/options/entities.py`, `genesis/options/indices.py`

## Solvers

| Solver | Options Class | Purpose |
|--------|--------------|---------|
| Rigid | `gs.options.RigidOptions` | Articulated rigid body dynamics |
| MPM | `gs.options.MPMOptions` | Continuum mechanics |
| SPH | `gs.options.SPHOptions` | Fluid simulation |
| FEM | `gs.options.FEMOptions` | Finite element deformation |
| PBD | `gs.options.PBDOptions` | Fast soft body simulation |
| SF | `gs.options.SFOptions` | Eulerian fluid/smoke |

Location: `genesis/engine/solvers/`

## Key Files Reference

| File | Purpose |
|------|---------|
| `genesis/__init__.py` | Package entry, `gs.init()`, global state |
| `genesis/engine/scene.py` | `Scene` class - main user interface |
| `genesis/engine/simulator.py` | `Simulator` - manages all solvers |
| `genesis/options/morphs.py` | Shape/geometry definitions |
| `genesis/options/solvers.py` | Solver option classes |
