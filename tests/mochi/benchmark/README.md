# MochiSolver benchmark

Per-step cost of Genesis' `MochiSolver` against the original mochi engine on six scenes that both engines build from the
same description (`scenes.py`: same meshes, coordinates, materials, time step 1/60 s, Newton caps).

| scene | contents | Newton cap |
|---|---|---|
| `rigid` | sphere (r 0.2) and cube (0.4) dropped onto a static table over the ground | 20 |
| `articulated` | double pendulum on a limited rail (prismatic + revolute + spherical) striking a ball | 20 |
| `equalities` | two links on a compliant world pivot and a compliant spherical joint | 20 |
| `soft_duck` | mochi's tetrahedral duck (1899 nodes, 8608 tets, neo-Hookean) dropped 1 m onto a plane | 20 |
| `cloth_tshirt` | mochi's t-shirt shell (3593 nodes, 7076 triangles) with self-contact falling onto a plane | 2 |
| `rod_helix` | mochi's helical spring (129 nodes) hanging from its first node | 20 |
| `franka` | Franka arm on a plane next to a box (Genesis only, 16k contact samples) | 20 |

The deformable assets are mochi's own, converted once to Z-up by `convert_mochi_assets.py` (`assets/`); the rigid
primitives are meshed by Genesis and exported to `assets/generated/` so that the mochi build carries the same triangles.
Both engines use the residual-norm line search, the same contact parameters (mochi defaults) and stop on the same
residual tolerances (`abs 1e-3`, `rel 1e-6`); the cap only matters when Newton does not converge.

## Protocol

5 warm-up steps, then 3 windows of 30 steps; the best window is the headline number (ms per step), the mean is kept in
the JSON. Both engines therefore time the fall, the impact and the settling of the same trajectory. One environment on
the CPU is the acceptance comparison (Genesis fp64 vs mochi built with `SUPERDEX_PRECISION=double`, mochi single-threaded);
Genesis GPU numbers are tracked in fp32 (mochi's default precision) at several batch sizes.

```bash
# Genesis (worktree venv); --profile adds launches/step, kernel time and the top kernels
python tests/mochi/benchmark/bench_genesis.py rigid --profile
python tests/mochi/benchmark/bench_genesis.py soft_duck --backend gpu --precision 32 --n-envs 8

# original engine (superdex venv); the rigid/articulated scenes need the meshes exported by bench_genesis.py
SUPERDEX_PRECISION=double SUPERDEX_ASSETS_PATH=... python tests/mochi/benchmark/bench_mochi.py rigid --threads 0

# table from results/*.json
python tests/mochi/benchmark/report.py
```

## Results (Ryzen 9 9900X, RTX 3080; Genesis worktree `feat/mochi-solver`)

Filled in by `report.py` at every optimization phase; see the git history of this file for the progression.

### Baseline, 2026-08-28, `2e467e90` (before the optimization phases)

| scene | mochi fp64 (T=0) ms/step | Genesis CPU fp64 ms/step | ratio | launches/step | Genesis GPU fp32 ms/step (B) | notes |
|---|---|---|---|---|---|---|
| rigid | 0.017 | 1.454 | 87.8x | 62 | - | 12 dofs, newton 1; mochi newton 0 |
| articulated | 0.036 | 3.190 | 88.0x | 94 | - | 11 dofs, newton 2; mochi newton 2 |
| equalities | 0.022 | 1.313 | 59.0x | 138 | - | 12 dofs, newton 4; mochi newton 4 |
| soft_duck | 12.371 | 258.935 | 20.9x | 1303 | - | 5697 dofs, newton 3; mochi newton 3 |
| cloth_tshirt | 32.201 | 648.020 | 20.1x | 962 | - | 10779 dofs, newton 2; mochi newton 2 |
| rod_helix | 2.277 | 364.261 | 159.9x | 12037 | - | 515 dofs, newton 20; mochi newton 20 |
| franka | - | 10.509 | - | 268 | - | 15 dofs, newton 8 |

Newton iteration counts agree between the two engines on every scene, so the ratios compare equal work.

### After phases B1-B2, B5-B6, B8 (rods) and C1-C2 (one-kernel step), 2026-08-29

| scene | mochi fp64 (T=0) ms/step | Genesis CPU fp64 ms/step | ratio | launches/step | Genesis GPU fp32 ms/step (B) | notes |
|---|---|---|---|---|---|---|
| rigid | 0.017 | 0.069 | 4.1x | 1 | 2.27 (B=1024) | 12 dofs, newton 1; mochi newton 0 |
| articulated | 0.036 | 0.157 | 4.3x | 1 | 9.33 (B=1024) | 11 dofs, newton 2; mochi newton 2 |
| equalities | 0.022 | 0.049 | 2.2x | 1 | - | 12 dofs, newton 4; mochi newton 4 |
| soft_duck | 12.371 | 100.945 | 8.2x | 1 | - | 5697 dofs, newton 3; mochi newton 3 |
| cloth_tshirt | 32.201 | 266.994 | 8.3x | 744 | - | 10779 dofs, newton 2; mochi newton 2 |
| rod_helix | 2.277 | 5.977 | 2.6x | 1 | - | 515 dofs, newton 20; mochi newton 20 |
| franka | - | 0.075 | - | 1 | - | 15 dofs, newton 8 |

Every scene is within the 10x target of single-threaded mochi on the CPU. The t-shirt (self-contact) still runs the
multi-kernel pipeline because its point-cloud collider is built by host-driven bounding-volume kernels; all other
scenes run as one kernel launch per step. GPU (RTX 3080, fp32, monolith, 1024 environments): rigid 2.2 us and
articulated 9.1 us per environment step.

### After B8 (neo-Hookean PSD oracle) and C3 (one kernel per step on every scene without self-contact), 2026-08-29, `5f628355`

| scene | mochi fp64 (T=0) ms/step | Genesis CPU fp64 ms/step | ratio | launches/step | Genesis GPU fp32 ms/step (B) | notes |
|---|---|---|---|---|---|---|
| rigid | 0.017 | 0.066 | 4.0x | 1 | 1.18 (B=1), 1.61 (B=64), 2.20 (B=1024) | 12 dofs, newton 1; mochi newton 0 |
| articulated | 0.036 | 0.164 | 4.5x | 1 | 5.12 (B=1), 5.70 (B=64), 9.89 (B=1024) | 11 dofs, newton 2; mochi newton 2 |
| equalities | 0.022 | 0.049 | 2.2x | 1 | 0.93 (B=1), 2.89 (B=1024) | 12 dofs, newton 4; mochi newton 4 |
| soft_duck | 12.371 | 88.701 | 7.2x | 1 | 21.48 (B=1), 29.53 (B=8) | 5697 dofs, newton 3; mochi newton 3 |
| cloth_tshirt | 32.201 | 239.807 | 7.4x | 744 | 32.85 (B=1), 47.65 (B=4) | 10779 dofs, newton 2; mochi newton 2 |
| rod_helix | 2.277 | 5.988 | 2.6x | 1 | 154.84 (B=1), 219.68 (B=64) | 515 dofs, newton 20; mochi newton 20 |
| franka | - | 0.075 | - | 1 | 1.59 (B=1), 3.83 (B=1024) | 15 dofs, newton 8 |

CPU fp32 (`--precision 32`) reproduces the fp64 trajectories on every scene (same probes) at the same speed: the CPU
kernels are scalar, so the precision only matters on the GPU.

#### GPU fp32 (RTX 3080), `5f628355`, automatic kernel choice

| scene | B=1 ms/step | B=4-8 ms/step | B=64 ms/step | B=1024 ms/step | us per env-step at the largest B | step kernel | GPU fp64 before (`2e467e90`) |
|---|---|---|---|---|---|---|---|
| rigid | 1.18 | - | 1.61 | 2.20 | 2.1 | monolith | 1.40 (B=1), 2.07 (B=64), 12.4 (B=1024) |
| articulated | 5.12 | - | 5.70 | 9.89 | 9.7 | monolith | 8.5 (B=1024) |
| equalities | 0.93 | - | - | 2.89 | 2.8 | monolith | - |
| franka | 1.59 | - | - | 3.83 | 3.7 | monolith | - |
| soft_duck | 21.5 | 29.5 (B=8) | - | - | 3690 | pipeline | 68.7 (B=1), 173 (B=8) |
| cloth_tshirt | 32.9 | 47.7 (B=4) | - | - | 11900 | pipeline | 82 (B=1) |
| rod_helix | 155 | - | 220 | - | 3430 | pipeline | 1311 (B=1), 2145 (B=64) |

The small rigid/articulated scenes run as one kernel per step on the GPU too: 1024 environments cost 2-10 us per
environment step (a single mochi core needs 17-36 us per step on these scenes), and one environment costs about the
launch latency of the monolith (1-5 ms, mostly the kernel's serial per-environment work on one GPU thread). The
deformables run the multi-kernel pipeline on the GPU (300-750 launches and the host-driven bounding-volume
hierarchies per step): the duck at 8 environments and the t-shirt at 4 already amortize to 3.7 ms and 11.9 ms per
environment step, but at one environment they are launch-bound (21-33 ms, on par with one mochi core on the t-shirt,
1.7x slower on the duck); the helix (515 dofs, 20 Newton iterations, ~120 PCG iterations per step) is the worst case
of the launch-bound pipeline at 155 ms - 26x its CPU time - and the one-kernel step is no better there (310 ms at B=1:
one GPU thread per environment). A graph-launched step kernel with parallel element loops (Phase D of the plan) is the
remaining route to a fast GPU path for deformables at small batch sizes.

Where the remaining time goes (CPU fp64, one environment):

- duck: 107 PCG iterations x 0.365 ms (scalar CSR matvec over 5697 dofs) + 3.7 full assemblies x 9.0 ms (1.05 us per
  tetrahedron incl. the PSD-projected tangent); mochi's element kernels are fp32 8-wide SIMD, which is the constant
  factor that remains.
- t-shirt: PCG 60% (86 iterations x 2.3 ms: shell CSR plus the self-contact hits in the matvec), point-cloud broadphase
  15% (one bounding-volume-hierarchy query per assembly), point-cloud collider evaluation 13%, shell assembly 7%.
  Reusing the broadphase candidates across the line-search trials or an in-kernel spatial hash for the self-contact
  broadphase (which would also let the t-shirt run as one kernel) are the next steps if it needs to be faster.
- rigid/articulated/equalities: one kernel launch (15-25 us) plus 30-140 us of serial work (forward kinematics, contact
  sampling over the culled sample tree, dense Cholesky, line search); mochi's 9-36 us is the same work in C++.

## Plan 2: batched deformables on the GPU, memory per environment (2026-08-29)

Three Genesis-only scenes model batched reinforcement learning with deformables (a Franka arm pressing a 31x31 cloth
with self-contact, a Franka gripper closing on a 4263-tetrahedron cube, the arm pressing a 64-node rope); `sweep.py`
runs a scene over batch sizes and every run records the solver's memory per environment (`memory_report()`), the
process' device memory, the conjugate-gradient iteration counts of both engines and the usage of the bounded contact
lists.

### Baseline before Plan 2 (`10b72f6f`, GPU fp32, host-driven pipeline with bounding-volume hierarchies)

| scene | B=1 ms/step | B=64 ms/step (us/env-step) | B=256 ms/step (us/env-step) | MiB/env | fits on 10 GB |
|---|---|---|---|---|---|
| cloth_arm | 30.0 | 257 (4010) | out of memory | 84.5 | 64 |
| soft_gripper | 44.4 | 294 (4590) | 1347 (5260) | 17.8 | 256 |
| rope_arm | 5.6 | 9.8 (154) | 18.2 (71) | 20.4 | 256 |

### After the spatial hash (Phase H): deformable colliders located by an in-kernel hash

Both deformable collider kinds (collider spheres of shells and rods, deformed tetrahedra of solids) are found through
a per-environment spatial hash rebuilt at every assembly, as in mochi (items inserted once in the bin of their cell, a
query walks the 27 cells around its own; with a cell no smaller than the contact range the candidate set is a superset
of the exact one and `tests/mochi/test_spatial_hash.py` checks that the hits equal a brute-force evaluation). No host
synchronization remains in an assembly, so every scene now runs as one kernel per step on the CPU, and the contact
range follows mochi's (`radius + penalty threshold`; the former `threshold + 2 x smoothing` band only added zero-force
hits: 30k of the 33k hits of the flat cloth+arm scene).

| scene | CPU fp64 ms/step (before -> after) | GPU B=1 ms/step | GPU B=64 ms/step (us/env-step) | GPU B=256 ms/step (us/env-step) | MiB/env |
|---|---|---|---|---|---|
| cloth_arm | 105 -> 8.9 (at rest) | 30.0 -> 14.6 | 257 -> 40.3 (630) | out of memory | 83.8 |
| soft_gripper | 130 -> 145 (see below) | 44.4 -> 32.1 | 294 -> 102 (1590) | 1347 -> 358 (1400) | 16.9 |
| rope_arm | 10.0 -> 2.8 | 5.6 -> 3.1 | 9.8 -> 4.8 (75) | 18.2 -> 7.8 (30) | 20.2 |
| soft_duck | 88.7 -> 86.8 | 21.7 -> 20.8 | 60.1 (940) | 177 (690) | 3.98 |
| cloth_tshirt | 240 -> 148 | 32.9 -> 25.1 | 81.5 (1270) | out of memory | 86.7 |

The GPU still runs the multi-kernel pipeline for deformables (the one-kernel graph step is Phase I); the gain is the
removal of the per-assembly hierarchy rebuilds (a host sync per tree layer) and of the flat shared query buffer. The
gripper on the CPU got slower because a coarse hash cell (the largest tetrahedron) made the cube's own samples walk
thousands of its own tetrahedra before the entity filter; the follow-up skips queries that cannot hit anything and
tests the entity filter before the dedupe. Memory per environment is unchanged: the hit buffers dominate (cloth+arm
84 MiB of which 33.7 MiB point-cloud hit records sized `16 x queries` and 45 MiB contact-readback records) - Phase G.

Conjugate-gradient iterations per step, mochi / Genesis (fp64, single thread): duck 92 / 101, t-shirt 78 / 86, helix
17.5 / ~120 - the tetrahedron and shell counts match (same block-Jacobi preconditioner and adaptive tolerance), the
rod's exact banded solve should bring the helix to mochi's count (Phase K).

### Memory diet (Phase G) and the bounds hash (H2, H3)

Contact points are recorded for readback only on demand (`get_contacts` counts, allocates and gathers; the readback
fields of the hit lists live in a struct allocated at the first readback), the hit lists keep the solver-side fields
only and their capacities follow three options (`max_soft_hits_per_sample`, `max_deformable_collider_hits_per_query`,
`max_point_cloud_hits_per_query`). The hash then inserts every item in the (at most eight) cells its bounds overlap and a
sample walks its own cell only; the tetrahedron cell is twice the median rest extent, the few larger tetrahedra live in
an overflow list every sample scans.

| scene | MiB/env before | MiB/env after | GPU fp32 B=256 ms/step (us/env-step) | GPU fp32 B=1024 ms/step (us/env-step) |
|---|---|---|---|---|
| cloth_arm | 84.5 | 6.3 | 120 (468) | 548 -> 382 (373) |
| soft_gripper | 17.8 | 4.8 | 1347 -> 257-282 (1000-1100) | out of memory -> 970 (947) |
| rope_arm | 20.4 | 2.2 | 8.2 (32) | 16.6 (16) |
| soft_duck | 3.98 | 2.30 | 176 (689) | 648 (633) |
| cloth_tshirt | 86.7 | 16.3 | 266 (1040) | out of memory (16.7 GB) |

With these, 1024 environments of every RL scene fit on the 10 GB card, and 4096 of each on 80 GB by projection (the
t-shirt at 4096 needs ~67 GB; the symmetric Hessian storage planned in Phase K/G7 halves its largest remaining array).
The deformables still run the multi-kernel pipeline on the GPU at this point; the graph step kernel is next.

### Step kernels on the GPU, the tetrahedron hierarchy and compile time (Phases I, H4, K4)

The one-launch graph step kernel (`step_kernel="graph"`: Newton, line-search and conjugate-gradient loops as nested
device-side `do_while` levels, every stage parallel over items and environments) agrees with the pipeline and the
monolith on every scene (`tests/mochi/test_step_kernels.py`, CPU and GPU). On the RTX 3080, which has no device-side
graph conditionals, the runtime replays the loop bodies from the host with one flag readback per round, and the kernel
runs at the pipeline's speed at every batch size while compiling as a single module several times slower - so it is
opt-in and "auto" keeps the pipeline for deformable scenes on the GPU (the monolith for small rigid ones).

| scene, GPU fp32 | B=1 graph / pipeline ms/step | B=64 | B=256 | B=1024 | cold compile graph / pipeline |
|---|---|---|---|---|---|
| soft_gripper | 33.4 / 33.0 | 79 / 76 | 214 / 208 | - | 85 s / 45 s |
| cloth_arm | 15.1 / 15.2 | 39.4 / 37.9 | - | 372 / 415 | 300 s / 220 s |

Compile time is measured per kernel (`bench_genesis.py --cold` records the first step's time with the offline cache
off; a per-kernel breakdown wraps quadrants' `Kernel.materialize` and the first launch). The cost sits in the backend
code generation at the first launch of a kernel (single-threaded; `num_compile_threads` changes nothing), it grows
faster than linearly with the kernel's size, and one func dominated it: the shell assembly evaluated the 18x18 tangent
as 216 fully unrolled pair contractions, each two 2x2 matrix products - about 100 s per compiled variant, twice per
scene. The tangent now hoists the per-dof products `A^-1 G A^-1` and traces out of the pair loops (the same arithmetic,
a quarter of the generated code), the pipeline kernels take the assembly flags as runtime values (one compiled
variant instead of two), and the tetrahedron assembly is compiled only for scenes that have tetrahedra
(`has_tets`; the monolith of a cloth scene no longer carries it).

| cold compile (s) | soft_gripper | cloth_arm | cloth_tshirt |
|---|---|---|---|
| CPU monolith, before -> after | 79 -> 79 (no shells) | 275 -> 108 | 254 -> 118 |
| CPU pipeline, before -> after | - | 142 -> 60 | - |
| GPU pipeline, before -> after | 45 -> 31 | 220 -> 70 | 239 -> - |

The tetrahedra of solids are no longer hashed: a uniform hash over a fine mesh (4263 tetrahedra of a 5 cm cube over
~64 cells, every tetrahedron in up to eight cells) walked chains of 200-500 entries for every sample near the cube, 48%
of the gripper step on the GPU and growing superlinearly with the batch. A bounding-box hierarchy built once over the
rest tetrahedra (`tet_tree.py`, depth-first order with escape indices, like the contact-sample trees) is refit to the
deformed vertices at every assembly one level at a time and queried with a stackless descent - a few dozen box tests
per sample; `tests/mochi/test_spatial_hash.py` checks the hits against a brute-force evaluation. The collider spheres
of shells and rods keep the hash. (Chunked per-environment reductions - 32 dofs per thread before one atomic - were
tried for the conjugate-gradient dot products and rejected: the vector passes cost the same, 0.09-0.11 ms per
iteration at B=256, so the per-environment atomics are not what limits them.)

| scene, GPU fp32, pipeline | B=1 ms/step | B=64 (us/env-step) | B=256 (us/env-step) | B=1024 (us/env-step) | MiB/env |
|---|---|---|---|---|---|
| soft_gripper, hash (H3) | 33.0 | 76 (1190) | 208 (813) | ~1000 (~980) | 4.82 |
| soft_gripper, hierarchy (H4) | 33.0 | 73 (1135) | 162 (633) | 678 (662) | 4.38 |
| cloth_arm, hierarchy (H4) | 15.2 | 37.9 (592) | 101 (396) | 353 (345) | 6.28 |
| cloth_tshirt (H4) | - | - | 253 (989) | - | 16.6 |
| soft_duck (H4) | - | - | - | 673 (657) | 2.32 |
| rope_arm (H4) | - | - | - | 10.0 (9.8) | 2.18 |

### GPU fp32 scaling at `1294b66e` (RTX 3080 10 GB, pipeline, cold-compiled once per scene)

| scene | B=1 ms/step | B=64 ms/step (us/env-step) | B=256 | B=1024 | MiB/env | max B on 10 GB (estimate) |
|---|---|---|---|---|---|---|
| cloth_arm | 15.0 | 38.9 (608) | 101.0 (394) | 373.2 (364) | 6.28 | 1537 |
| soft_gripper | 33.5 | 67.1 (1048) | 164.1 (641) | 669.4 (654) | 4.38 | 2204 |
| rope_arm | 3.4 | 4.6 (72) | 6.6 (26) | 10.1 (10) | 2.19 | 4452 |
| soft_duck | 21.5 | 62.6 (978) | 184.8 (722) | 669.1 (653) | 2.32 | 4180 |
| cloth_tshirt | 25.3 | 85.7 (1340) | 268.6 (1049) | out of memory | 16.63 | 589 |
| rod_helix | 154.8 | 224.2 (3503) | 305.8 (1195) | 374.7 (366) | 0.24 | 41430 |

Against the Plan 2 baseline (`10b72f6f`): cloth + arm 4010 -> 364 us per env-step at the largest batch that fits (11x,
and 1024 environments fit where 256 did not), gripper 5260 -> 654 (8x), rope + arm 71 -> 10 (7x); the projection to an
80 GB card is 4096 environments for every RL scene (the t-shirt needs 16.6 MiB/env: ~4800 on 80 GB). The helix is
launch-bound at small batches (20 Newton iterations per step) and only pays off at B >= 256.

The tetrahedron stiffness is assembled per node block (`func_tet_stiffness`): the Smith neo-Hookean tangent in closed
form when mochi's oracle proves it definite (`c3 (F g_f)(F g_g)^T + lam (cof g_f)(cof g_g)^T + c2 (g_f . g_g) I
+ coeff S(F (g_f x g_g))`), the analytic eigenmodes otherwise (nine rank-one block updates), instead of a 9x9 tangent
contracted term by term (1296 multiply-adds per element); `tests/mochi/test_soft_materials.py` checks the blocks
against the contraction to 1e-10 on tension, compression, shear and inversion. Gripper GPU B=256 assembly kernel 1.67
-> 1.51 ms per call (step 165 -> 160 ms), duck B=1024 673 -> 641 ms. On the CPU the duck does not move (86 ms): near
rest under compression the oracle fails - the rest state is exactly marginal by construction of the model's alpha - so
most tetrahedra take the SVD + Jacobi eigenmode path, where the block form saves only a fifth of the arithmetic.

Culling the rigid samples that query the deformable colliders was tried and rejected. Per assembly, each link's
sample hierarchy (the trees the rigid contact evaluation traverses) was descended against the current collider bounds
(tetrahedron-tree root box, sphere bounds plus contact range) and only the overlapping leaves' samples queried - exact,
and on the gripper it cut the queries from 16k to ~3k once the fingers hold the cube (a per-step variant on the
broadphase's conservative bounds kept 11k: 12 cm link pads and a 31 cm box around a 5 cm cube). The step did not move
on the gripper (B=256 160 -> 170 ms, B=1024 652 -> 717) because a far sample already costs one root-box test, and it
slowed the cloth + arm scene by 50% (B=256 101 -> 153 ms): the cloth spans the arm's workspace, so every link
traverses its whole tree at every assembly and the list barely shrinks. What the tetrahedron query costs is the
near-field: thousands of finger-pad samples inside the cube's box, each a divergent descent.

The CSR matvec of the deformable Hessian walks the column sequence of a vertex's three rows once per vertex (the
sparsity is built from whole vertex blocks, so the three rows share it; the layout asserts this and adds every vertex's
own 3x3 block): one thread per vertex reads each column index and source value once for three rows, a third of the
index traffic and of the gathered loads. Rod twist rows keep the scalar walk. Gripper GPU B=256 matvec pass 0.209 ->
0.181 ms per PCG iteration; duck GPU B=1024 641 -> 581 ms/step, gripper B=1024 652 -> 627; CPU fp64 single thread: duck 86 -> 67 ms/step (-22%, 5.4x mochi), gripper 71 -> 64,
t-shirt 148 -> 87 (-41%, 2.7x mochi).

The three per-environment reductions of a conjugate-gradient iteration (p.Ap; the residual update fused with the
Polak-Ribiere cross term; r.z and z.z) run as 256-thread blocks tiled as 32 environment lanes by 8 chunks of a 64-dof
tile: every warp reads 32 consecutive environments of one row, the chunks are summed in shared memory and each
environment receives one atomic per tile instead of one per dof (standalone: 44 -> 15 us per pass at B=256, and a
tenth of the fp32 rounding error of per-dof atomics). The CPU and per-environment paths are unchanged. GPU fp32 pipeline: gripper B=256 159 -> 154 ms/step, B=1024 627 -> 569 (-9%),
cloth + arm B=1024 391 -> 365, t-shirt B=256 255 -> 236, duck B=1024 unchanged (581). (The kernel profiler cannot
see this: it inflates every small task to ~0.1 ms, so only wall-clock step times are used to judge such passes.)

### CPU fp64 single thread after Plan 2 (monolith, ms/step; mochi single-threaded in parentheses)

| scene | before Plan 2 | now | vs mochi |
|---|---|---|---|
| soft_duck | 88.7 | 67 | 5.4x (12.4) |
| cloth_tshirt | 240 | 87 | 2.7x (32.2) |
| soft_gripper | 92-130 | 64 | - |
| rod_helix | 6.0 | 6.0 | 2.6x (2.28) |

The t-shirt reached the 2-3x band through the hash (240 -> 148) and the vertex-row matvec (148 -> 87); the duck's
remaining gap is the eigenmode path of its resting tetrahedra (44% of its step) and the scalar fp64 arithmetic that
mochi does in 8-wide fp32.

### fp32 against mochi's own fp32 (Phase L)

mochi runs fp32 by default; its fp32-vs-fp64 deviation on a scene is the natural floor for judging Genesis fp32. Probe
after 95 steps (the benchmark protocol), mochi fp64 as the reference:

| scene | mochi fp64 | mochi fp32 - fp64 | Genesis fp64 (CPU) - mochi fp64 | Genesis fp32 (GPU) - mochi fp64 |
|---|---|---|---|---|
| soft_duck (height, m) | -0.00227 | 2.6e-9 | 5e-13 | 5.7e-9 |
| cloth_tshirt (height, m) | 0.20986 | 3.6e-3 | 1.2e-3 | 9.9e-3 (2.7x mochi's own fp32 deviation) |
| rigid (height, m) | 0.99925 | 1.3e-8 | 2.7e-4 | 2.7e-4 (both precisions equal) |
| articulated | 0.15000 | 6e-9 | 0 | 6e-9 |

The duck and the rigid scenes are precision-insensitive (the 0.27 mm rigid offset between the engines is the same in
both precisions); the t-shirt - 95 steps of a self-colliding cloth - moves 3.6 mm between mochi's own precisions and
9.9 mm between Genesis fp32 and mochi fp64, i.e. within three times mochi's floor. `tests/mochi/test_fp32_parity.py`
pins these bounds (four times the floor plus 1 mm) for the duck and the t-shirt in fp32 on either backend. The
equalities and helix probes measure different quantities in the two harnesses and are not compared.

The tetrahedron and shell assemblies scatter their stiffness through per-block positions (the block's first column in
the row vertex's shared column sequence; the scalar index is `csr_start[row] + position + column`): 16 + 12 / 36 + 18
index loads per element instead of 144 / 324, and static tables nine times smaller. The preconditioner's scalar Jacobi
pass covers the rigid dofs only (the deformable dofs are written by the block-Jacobi, rod-band and twist passes).
GPU fp32: gripper B=1024 568 -> 549 ms/step, duck B=1024 577 -> 554, cloth + arm B=1024 365 -> 348; CPU fp64 duck
67 -> 64.6 ms/step, t-shirt and gripper unchanged; total memory duck 15.2 -> 10.9 MiB, t-shirt 44.3 -> 36.3 at B=1.

Gripper profile at B=256 after the hierarchy: tetrahedral contact query 23% (was 48%), conjugate gradient 37% (the
CSR matvec at ~75% of the card's bandwidth, the vector passes limited by per-environment atomics), tetrahedron assembly
13% (12x12 element tangents; Phase K1).

## Plan 3: single-environment teleoperation speed, CPU threads, batched memory (2026-09-01)

Round-3 goals: (1) one-environment step time within 1-2x of single-threaded mochi on the best backend per scene, for
responsive teleoperation of deformable scenes; (2) faster and smaller batched simulation. Changes, each measured on an
idle machine right before and after (`perf/mochi-round3`):

- The residual-only zeroing variant compiles without the Hessian loops (the line-search trials no longer dispatch the
  full `csr_values` walk): t-shirt GPU fp32 B=256 228.9 -> 223.4 ms/step.
- Opt-in CPU multithreading: `gs.init(cpu_threads=N)` (or QD_NUM_THREADS) raises the solver's own parallelization
  level, and the auto rule runs large deformable scenes through the pipeline so their element and contact loops
  parallelize. The per-environment scalar reductions stay serial on the CPU (a parallel reduction into one
  accumulator measured 10x slower at one environment), and the GPU-shaped blocked reductions are gated off x64.
- The rod band preconditioner now includes the contact and attachment blocks on banded vertices: it was the exact
  inverse only while the rod was free, and a rod pressed into a collider cost 3-5 conjugate-gradient iterations per
  Newton solve instead of 1 (helix: pcg 95 -> 20 per step, 5.32 -> 4.70 ms). Compiled only for scenes with open rods
  (a first ungated version cost the rod-less gripper ~15-20% at B=1024).
- The conjugate-gradient passes are fused into segmented loops (~21 -> ~11 offloaded tasks per iteration; the
  direction update rides in the matvec initialization and skips converged environments): duck GPU fp32 B=1
  21.1 -> 16.9 ms, t-shirt 21.4 -> 17.3; t-shirt CPU fp64 at 8 threads 71.4 -> 56.7 ms.
- The symmetric contact matrices (`hit_D`, `sc_hit_D`, `pc_hit_D`, `acc_D`, `acc_SDS`) are stored as six floats, and
  the self-contact hit capacity default halves: t-shirt 16.63 -> 9.99 MiB/env, cloth + arm 4.32 -> 2.63, gripper
  4.38 -> 3.88, at unchanged step times.

One environment against single-threaded mochi fp64 (best backend per scene; mochi 12.4 / 32.2 / 2.28 ms):

| scene | best backend | ms/step | x mochi | CPU fp64 1 thread | CPU fp64 8 threads |
|---|---|---|---|---|---|
| soft duck | GPU fp32 | 16.9 | 1.4x | 64.5 (5.2x) | monolith kept (pipeline pays no dividend below ~3k vertices) |
| cloth t-shirt | GPU fp32 | 17.3 | 0.54x | 86.7 (2.7x) | 56.7 (1.8x) |
| rod helix | CPU fp64 | 4.70 | 2.1x | 4.70 (2.1x) | monolith kept |

The teleoperation examples take `--cpu-threads`; `finray_gripper.py` runs at ~5.5 ms/step warm (CPU fp64, one
thread). Batched fp32 speed is within a few percent of Plan 2 (the cloth scenes gain ~4%, the tet scenes are
bandwidth- and contact-bound); memory per environment fell 40-65% (above). The dominant batched cost is now the
tetrahedral contact query, re-detected at all five assemblies of every Newton iteration (46.7% of the gripper step
at B=1024): the next lever is mochi's pass structure (a Hessian-only pass from stored hit records, cutting the
detection count per iteration), followed by the symmetric-CSR storage if the matvec gains outweigh its
scatter-transpose atomics.

Measurement note: the batched GPU numbers of this table were taken while the machine was otherwise idle; a
concurrent GPU job inflates them 20-40% (best-of-3-windows does not filter a sustained co-tenant).
