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
│   │   ├── materials/         # Material definitions per solver
│   │   └── couplers/          # Inter-solver coupling
│   ├── options/               # Configuration classes (Pydantic models)
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
| `ToolEntity` | Tool | Cutting/interaction tools |

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

Materials define physical properties. Each entity options class admits the materials of its own family.

```python
gs.materials.Rigid(rho=1000)
gs.materials.MPM.Elastic(E=1e5, nu=0.3)
gs.materials.SPH.Liquid(sampler="pbs")
gs.materials.PBD.Cloth(stretch_compliance=0.0)
```

Location: `genesis/engine/materials/`

## Entity options

An entity is constructed from typed options, whose class decides the kind of entity and the solver simulating it: each class names the options class of its solver (`EntityOptions.scene_solver_options_cls`). The flattened arguments of `Scene.add_entity` are a shorthand for `RigidEntityOptions`, so the material never changes the kind of entity.

```python
scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))  # rigid
scene.add_entity(options=gs.options.KinematicEntityOptions(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1))))
scene.add_entity(options=gs.options.MPMEntityOptions(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)), material=gs.materials.MPM.Elastic()))
```

Location: `genesis/options/entities.py`

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

### System data lifecycle

`Simulator` owns an internal `SystemDataArray`. Each record has a concrete type, an owning solver and an index: 0 for the aggregate record of a solver, the scene index of the entity for a per-entity record and the solver index of the geometry for a per-geometry record. The owning solver decides whether its record exists, describes it and binds it, and the record lives as long as the scene. Solvers, couplers and entities look records up during build and keep references for runtime access, and a coupler writes into the fields of the records it couples.

The build phases are:

1. `describe()` reads the entity ranges that `add_entity` assigned, resolves the static configuration, and returns the dimensions and inputs of the shared allocation, or None when the solver allocates nothing.
2. The simulator allocates every description into its data collection.
3. `register_data()` binds the allocated record and registers the range records that index into it. The simulator calls it only for a solver that allocated a record.
4. The collection closes registration.
5. `build()` initializes state and private workspaces.

| Family | Solver description | Record | Range records | Kept by the solver |
|--------|--------------------|--------|---------------|--------------------|
| Kinematic, rigid | `ArticulatedDescription` | `ArticulatedData`: dynamic info and state | `LinksData`, `JointsData` per entity, `VisualGeomData` per visual geometry, `RigidGeomData` per collision geometry of the rigid solver | `RigidInfo`, scratch and adjoint workspaces |
| FEM | `FEMDescription` | `FEMData`: time-indexed state, element and vertex info, surface fields | `FEMEntityData` per entity | Iterative solve, rendering and constraint workspaces |
| MPM | `MPMDescription` | `MPMData`: particle history, particle info, coupling grid | `ParticleEntityData` per entity | Grid reset bookkeeping, render buffers |
| SPH | `SPHDescription` | `SPHData`: particle state and info, in entity order and in spatial hash order | `ParticleEntityData` per entity, in entity order | Spatial hash, render buffers |
| PBD | `PBDDescription` | `PBDData`: particle state and info, in entity order and in spatial hash order, rest topology | `PBDEntityData` per entity: particle range and topology ranges | Spatial hash, visual vertex supports, render buffers |
| Tool | `ToolDescription` | `ToolData`: the `ToolEntityData` of each entity | `ToolEntityData` per entity: pose history, signed distance field | Boundary |
| SF | `SFDescription` | `SFData`: grid of cell velocity, pressure and jet concentrations, shared by every environment | None | Swap fields of the pressure projection |

A range record holds the aggregate record it indexes and its ranges into it. A tool entity holds fields of its own, so its `ToolEntityData` holds them in place of ranges. Checkpoint iteration includes each shared allocation once. A visual geometry pose is refreshed by `update_vgeoms()` before reading its state directly.

Every entity is created from its `EntityDescription`: the solver resolves the description in `add_entity` when the caller passes none, and `Scene.export` writes it. A description declares its material as the base material of its solver, such as `FEM.Base`, so a material name read from a scene file resolves among the materials of that solver.

## Key Files Reference

| File | Purpose |
|------|---------|
| `genesis/__init__.py` | Package entry, `gs.init()`, global state |
| `genesis/engine/scene.py` | `Scene` class - main user interface |
| `genesis/engine/simulator.py` | `Simulator` - manages all solvers |
| `genesis/options/morphs.py` | Shape/geometry definitions |
| `genesis/options/solvers.py` | Solver option classes |
