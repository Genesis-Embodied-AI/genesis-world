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
│   │   │   └── pbd_solver.py  # Position Based Dynamics
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

Materials define physical properties, each material belonging to one solver. Each entity options class admits the materials of its solver.

```python
gs.materials.Rigid(rho=1000)
gs.materials.MPM.Elastic(E=1e5, nu=0.3)
gs.materials.SPH.Liquid(sampler="pbs")
gs.materials.PBD.Cloth(stretch_compliance=0.0)
```

Location: `genesis/engine/materials/`

## Entity options

An entity is constructed from typed options, whose class decides the kind of entity and the solver simulating it: the simulator maps each class to the solver simulating its entities, and each class admits the materials of that solver. `gs.entities.Hybrid` maps to no solver, since a hybrid entity adds a rigid entity and an MPM entity of their own. The individual arguments of `Scene.add_entity` are a shorthand for the entity options its material is given to, `gs.entities.Rigid` when it is None.

```python
# Rigid, through the individual arguments
scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
scene.add_entity(
    options=gs.entities.Kinematic(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
    ),
)
scene.add_entity(
    options=gs.entities.MPM(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
        material=gs.materials.MPM.Elastic(),
    ),
)
scene.add_entity(
    options=gs.entities.SPH(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
        material=gs.materials.SPH.Liquid(),
    ),
)
```

Entity options are frozen: their fields are fixed at construction, and `select` and its setters alone record into `selection_blocks` what is set on parts of the entity. Each kind declares the blocks it admits, each with the selectors of the parts it is set on from the coarsest level to the finest (`_blocks_selector_types`), so that a link selection narrows to collision geoms or to visual geoms. `select` returns a view (`LinkSelection`, `GeomSelection`, `VGeomSelection`, `VertexSelection`) whose setters record a `gs.options.SelectionBlock`, the selectors and the block, in call order, and a scene file stores those blocks. The entity holds a frozen copy of its options with its name recorded into it, and its description holds that copy next to what the resolution derives.

The rigid description resolves the blocks against the parsed geoms and visual geoms of each morph before post-processing splits or fuses them, so an ordinal counts geoms as the asset states them and geoms of different values never fuse. The FEM entity resolves the blocks of vertices, and its solver applies the fixed ones through `set_vertex_constraints` once the scene is built. `gs.entities.Rigid.select` states the precedence of blocks and how nested selections resolve.

```python
arm_options = gs.entities.Rigid(morph=gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf"))
arm_options.select(gs.indices.Link.named("arm2")).set_friction(1.2)
scene.add_entity(options=arm_options)
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

Location: `genesis/engine/solvers/`

## Key Files Reference

| File | Purpose |
|------|---------|
| `genesis/__init__.py` | Package entry, `gs.init()`, global state |
| `genesis/engine/scene.py` | `Scene` class - main user interface |
| `genesis/engine/simulator.py` | `Simulator` - manages all solvers |
| `genesis/options/morphs.py` | Shape/geometry definitions |
| `genesis/options/solvers.py` | Solver option classes |
