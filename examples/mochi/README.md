# MochiSolver examples

The `MochiSolver` is a port of the implicit multi-physics solver of Meta's mochi physics engine
(https://github.com/facebookresearch/project_superdex, Apache-2.0). Every substep solves one nonlinear system in which
the inertia of all bodies and a smooth signed-distance penalty contact model are assembled together, so large time
steps (10-25 ms) remain stable, and contact needs no coupler.

Entities are added with `gs.materials.Mochi.Rigid(...)`; solver options live in `gs.options.MochiOptions`. Double
precision (`gs.init(precision="64")`) is recommended: the default contact stiffness of 1e9 Pa/m makes the Newton system
ill-conditioned in single precision.

Contact parameters (per material, combined per pair by geometric mean):

| parameter | default | meaning |
|---|---|---|
| `penalty_coefficient` | 1e9 Pa/m | contact pressure per unit penetration once the ramp is fully active |
| `penalty_threshold` | 1 mm | signed distance at which pressure starts to build |
| `penalty_smoothing_half_distance` | 5 mm | half-width of the smooth pressure ramp (full ramp 10 mm) |
| `friction` | 0.5 | Coulomb friction coefficient |
| `friction_falloff_vel` | 0.01 m/s | sliding speed below which friction is regularized |
| `normal_viscous_damping` | 0 s/m | impact damping; the coefficient of restitution e at impact speed v follows from `c = (1-e)(1+4.5e)/(e(1+8e/3))/v` |

Supported: free and fixed rigid bodies, articulated bodies (URDF/MJCF kinematic trees with fixed, revolute, prismatic,
spherical and free joints; joint damping, armature, stiffness, soft range limits, force/velocity/position drives through
the usual `control_dofs_*` / `set_dofs_kp` / `set_dofs_kv` API), deformable bodies (`gs.materials.Mochi.Elastic`:
linear tetrahedra with a stable neo-Hookean, Saint Venant-Kirchhoff or linear material, mass and stiffness damping,
fixed vertices through `set_vertices_fixed` (or driven along a prescribed trajectory with `set_vertices_target`, a
moving Dirichlet condition reached exactly at the end of every step), tetrahedralized by tetgen from Box/Sphere/Cylinder/Mesh morphs or read from
tetgen `.node`/`.ele` files), plane/sphere/box analytic colliders and grid colliders for meshes, backward Euler and
BDF2 time integration. Contact between the links of one entity is disabled, equality constraints are ignored and drive
forces are not clamped to the force range. Deformable bodies collide through the quadrature samples of their boundary
triangles against the rigid colliders, and act as colliders themselves (`collider_type="sdf"`, the default): a signed
distance field of the rest shape is queried through the deformed tetrahedra, so rigid samples and the samples of other
soft bodies are pushed out once they are inside the body. That path has no contact skin (contact only registers for
points inside the body, so a few millimeters of interpenetration remain) and its tangent drops the derivatives of the
mapping, as in mochi; a soft body never collides with itself. Mesh soft bodies reasonably finely
(`maxvolume=..., nobisect=False`) where sharp rigid features touch them.

Vertices of a deformable body can be rigidly attached to a link of a rigid or articulated entity
(`entity.attach_to_link(link, verts_idx, stiffness=1e6, damping=0.0)`, mochi's deformable-node-to-rigid constraint):
each attached vertex is tied by a stiff penalty `1/2 k |c|^2` (plus an optional damping on the violation rate) to the
point of the link it coincides with at build time, inside the same implicit solve - forces act on both sides and the
coupled bodies form one simulation island. This is how a robot gets soft parts, e.g. compliant fingertips mounted on
a gripper (`finray_gripper.py`). A solid can carry a detailed visual mesh skinned by its simulation tetrahedra
(`entity.set_visual_mesh(file, pos=..., euler=..., scale=...)`, placed in the frame of the simulation mesh file):
every render vertex is embedded barycentrically in the tetrahedron it falls in (vertices outside the coarse mesh
extrapolate linearly from the nearest one) and follows the deformation - the usual pairing of a fine render mesh
with a coarse collision mesh. Tetrahedral simulation meshes can also be read from legacy ASCII VTK unstructured
grids (`gs.morphs.Mesh(file="....vtk")`), the format Drake ships its deformable meshes in.

Thin shells (`gs.materials.Mochi.Shell`) use the surface triangles of a `Mesh` (or primitive) morph as elements: a
Saint Venant-Kirchhoff membrane on the metric of the mid-surface and a Koiter-type bending term on its discrete
curvature, both thickness-integrated from `E`, `nu`, `rho` and `thickness` (each stiffness can be overridden). Shell
samples carry no orientation (the contact normal comes from the collider, so both sides collide), and shells act as
colliders through spheres of radius `collider_radius` placed at their vertices (`collider_type="point_cloud"`). As in
mochi, the spheres only collide with the samples of other point-cloud entities (cloths and rods) or of the same body
under self-contact — rigid bodies and solids interact with a shell through the shell's own samples instead; a shell
never collides with itself unless `self_contact=True`. Consistently wound meshes are required. Light cloth has little inertia: use a lower
`penalty_coefficient` (1e7 Pa/m) or a larger `explosion_rel_tol` so that the sudden contact residual of an impact is
not mistaken for a divergence.

Contact is sampled: a collider is only felt at the boundary quadrature points of the cloth (three per triangle by
default), so a sharp edge visibly pokes through the flat triangles between samples when the cloth mesh is coarse
relative to the collider — a 12-cell 0.8 m sheet draped over a 0.3 m box shows the box edge 5-8 mm proud of the cloth
surface, and the original mochi produces the same picture at the same magnitude (verified side by side at both 1e7 and
1e9 Pa/m). It is a discretization artifact, not a solver defect; what reduces it is a finer cloth mesh (shorter chords
across the edge), more samples per triangle (`boundary_element_type="P1Q6"` on `MochiOptions`), or a larger
`penalty_threshold` on the rigid material so contact engages above the surface and the cloth floats clear of the edge.

Rods (`gs.morphs.Rod` polyline + `gs.materials.Mochi.Rod`) are discrete elastic rods: stretching of the segments,
bending and twisting at the interior nodes, with one twist angle per segment as an extra unknown and the material
frames carried along as parallel-transported state (a Kirchhoff rod, no shear). The stiffnesses derive from `E`, `nu`,
`rho` and the radius (`E A`, `E I`, `G J`, `rho A`, `rho J`) and can be overridden; clamp a rod by fixing its first two
nodes. Rods collide through samples on their centerline (no radius offset on the colliding side) and act as
colliders through spheres of the rod radius carried by their nodes (`collider_type="auto"`); as for shells, the
spheres only collide with other point-cloud entities or the rod itself under self-contact, so a rod lying on another
rod sits one rod radius above its centerline while a rigid body touches the centerline samples directly.

Velocities are recovered by finite differences over the step as in mochi: `get_dofs_velocity()` returns
`sin(dq)/dt` for revolute joints and the sine-based angular velocity `vee(((R - R_prev)/dt) R^T)` for spherical and
free joints, which differ from `dq/dt` by a factor `1 - (dq)^2/6`.

Examples in this directory (every one takes `-v` for the viewer, `-g` for the GPU, `-n` for the step count and `-r`
to record a video):

- `rigid_bodies.py`: a sphere and a cube dropped onto a table standing on the ground.
- `articulated_arm.py`: a Franka Panda arm grasping and lifting a soft box under PD joint control.
- `loop_closure.py`: a four-bar linkage closed by a connect equality constraint, next to two welded boxes.
- `bolt_nut_self_screw.py`: a nut screwing itself down a fixed bolt through non-convex thread contact (the
  mochi counterpart of `examples/rigid/bolt_nut_self_screw.py`), at plain 60 Hz steps with no substeps;
  `--torque 1.0` unscrews it back up, and the self-locking thread holds the nut once the torque is released.
- `newton_cradle.py`: Newton's five-ball cradle, an elastic impact chain (each pendulum its own entity, since
  the links of one entity never collide). The penalty contact is elastic by default; with BDF2 and 0.33 ms
  substeps ~97% of the swing reaches the last ball, while backward Euler at plain 60 Hz steps damps it to ~20%
  (`--substeps`, `--integrator` compare the schemes).
- `cloth.py`: a sheet draped over a rigid box, and a ball caught by a sheet whose held corners then sway.
- `cloth_p1q6.py`: the drape of `cloth.py` with six contact samples per triangle (`-q P1Q3` for the default
  three), printing how deep the box edge shows through the cloth surface — an A/B of the sampled-contact edge
  artifact described above.
- `rods.py`: a rope clamped at both ends, a stiff cantilever, and a loose rope dropped onto the ground.
- `soft_bodies.py`: a soft sphere and cube dropped onto the ground, and a rigid box landing on a soft slab.
- `finray_gripper.py`: a Schunk WSG-50 gripper with deformable FinRay fingers attached to its finger links and
  skinned visual meshes, closing on a box (requires the LBM eval models, see the example).
- `articulated_finray_grasp.py`: the same FinRay gripper mounted on a Franka Panda in place of its hand
  (`RigidEntity.attach`), picking a box off the ground and lifting it 24 cm (requires the LBM eval models).
- `soft_duck.py`: the mochi duck (1899 nodes, 8608 tetrahedra) dropped onto the ground (`soft_duck` benchmark scene).
- `cloth_tshirt.py`: the mochi t-shirt (3593 nodes, self-contact) falling onto the ground (`cloth_tshirt` benchmark
  scene).
- `rod_helix.py`: a helical spring hanging from its first node (`rod_helix` benchmark scene).
- `cloth_arm.py`: a Franka arm pressing a 31x31 self-colliding cloth on the ground (`cloth_arm` benchmark scene).
- `rope_arm.py`: a Franka arm pressing the middle of a 64-node rope on the ground (`rope_arm` benchmark scene).
- `cloth_teleop.py`: keyboard teleoperation of a Franka arm grabbing two stacked cloths draped over a grid of
  cubes (inverse kinematics on the target frame; the mochi counterpart of the IPC robot-cloth teleoperation
  example).

The benchmark-scene examples reproduce the acceptance scenes of `tests/mochi/benchmark` (same meshes, materials and
Newton budgets; the duck, t-shirt and helix assets are converted from the original mochi engine's examples) and read
those meshes from `tests/mochi/benchmark/assets/`.

Contact points are read back with `get_contacts()` on rigid and deformable entities alike (`with_entity` filters a
pair). Every record carries the scene entities, links and geoms of both sides (-1 where a side is deformable), the
entity-local vertices and barycentric weights the force is spread over on a deformable side (`verts_a`/`bary_a` for
the sample side, `verts_b`/`bary_b` for the collider side; a point-cloud collider is a single vertex), the position,
unit normal pointing away from side B, signed distance, force on A and quadrature weight of the sample.
`get_vertices_contact_force()` gives the per-vertex net contact force of a deformable body.

Shells and rods can collide with themselves through their point-cloud collider (`self_contact=True` on the material):
every sample collides with the spheres of the vertices of its own body except those lying within
`collider_radius * self_contact_exclusion_ratio` plus the penalty threshold of it in the rest configuration (its own
and the neighboring elements). Choose the ratio so that the excluded range covers the nearest vertices while the next
ones stay beyond the contact band (radius + threshold + two smoothing half distances), e.g. 3 for a rod whose segments
are twice its radius.

Bodies coupled by the contact candidates of a step (rigid entities, deformable bodies) form simulation islands. The
direct linear solver (`linear_solver="auto"`, the default) factorizes the Newton matrix island by island whenever the
largest island of an environment has at most `dense_solver_max_dofs` degrees of freedom, so a scene of many separate
bodies keeps the exact solver; environments with a larger island fall back to the conjugate gradient. The dense matrix
is only allocated up to `dense_matrix_max_dofs` degrees of freedom in total. On GPU, systems that fit in shared memory
are factorized by a fused tiled Cholesky kernel. `solver.island_state.n_islands` exposes the island count per
environment.

Equality constraints of articulations (MJCF `connect`, `weld` and `joint` couplings, i.e. loop closures such as
four-bar linkages) are enforced as stiff penalties inside the same Newton solve, `0.5 * k |c|^2` on the violation `c`
with `MochiOptions.equality_stiffness` (1e6 by default, mochi's soft constraint default) and an optional damping
`equality_damping` on its rate; the coupled bodies form one simulation island.
