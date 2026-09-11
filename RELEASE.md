# Genesis Release Note

## 1.4.0

This release introduces the capability to export a scene plus a complete trajectory as a standalone archive that can be loaded on any machine while ensuring bit-exact replay using `gs replay`. The objective is making bug report much easier in the future. Only the rigid solver is supported for now. Beyond that, the static description of entities is now clearly separated from runtime getters to avoid confusion. Finally, a large number of minor bugs have been fixed, most of them related to asset parsing.

### New Features

* [BREAKING] Support applying an external wrench at any local point on a link. (@Milotrince) (#3143)
* [BREAKING] Apply morph pose offset to velocity/acceleration getters. (@Milotrince) (#3250)
* [BREAKING] Share a scene as a file that opens without its assets. (@duburcqa) (#3281, #3288, #3294)
* Save and restore a built scene in memory. (@duburcqa) (#3309, #3303)
* Record and replay a trajectory to a file that opens without its assets. (@duburcqa) (#3309)

### Bug Fixes

* Always give recorded videos a valid default filename. (@duburcqa) (#3228)
* Make successive trajectories bit-identical from the reset onwards. (@duburcqa) (#3230)
* Support FEM vertex constraints and topology queries in multi-entity scenes. (@jeetrex17) (#3239)
* Support envs_idx mask in FEM vertex constraints. (@jeetrex17) (#3241)
* Support broadcasting inverse kinematics targets in batched scenes. (@jeetrex17) (#3247)
* Fix moving debug objects in batched scenes. (@duburcqa) (#3255)
* Fix parsing URDF for partially specified dynamics attributes. (@catplotlib) (#3262)
* Fix multidimensional tensor recording as CSV file. (@CoffeeDrivenCoder) (#3266)
* Fix 'RigidEntity.plan_path' ignoring runtime joint limits. (@duburcqa) (#3278)
* Preserve URDF mimic joint relations when scaling robots. (@ktyang512) (#3271)
* Fix buggy hibernation on GPU and correctly report resting force. (@duburcqa) (#3297)
* Report a zero velocity for a body that fell asleep. (@duburcqa) (#3300)
* Fix COM frames of fixed links on aligned bodies breaking thrust points. (@jeetrex17) (#3283)
* Support offscreen rendering with OSMesa. (@duburcqa) (#3302)
* Apply MeshSet member poses to sampled particles. (@jeetrex17) (#3305)
* Keep the authored mass and inertia of fixed links on aligned free bodies. (@duburcqa) (#3304)
* Recover a usable inertia for URDF links stating a zero mass or inertia. (@Milotrince) (#3261)
* Leave links fixed to the world out of the potential energy. (@duburcqa) (#3311)
* Fix dynamics of attached entities and resolve assets identically in every precision. (@duburcqa) (#3311)
* Fix scaling wrongly applied for in-memory URDF file morph. (@jeetrex17) (#3313)
* Correct a rigid-PBD attachment over the substep interval rather than the whole step. (@duburcqa) (#3237)
* Honor the textures of URDF materials. (@duburcqa) (#3319)
* Fix convex collision detection on geom pairs of very different sizes. (@duburcqa) (#3321)

### Miscellaneous

* [BREAKING] Stop offering a checkpoint that restores a fraction of the state. (@duburcqa) (#3228)
* [BREAKING] Separate the static description of a rigid entity from its runtime state. (@duburcqa) (#3281)
* [BREAKING] Read gravity, time and mass from the solver that owns them. (@duburcqa) (#3237)
* [BREAKING] Register a recorder with 'Scene.add_recorder'. (@duburcqa) (#3309)
* [BREAKING] Drop the rotation of a recorder's file on reset.(@duburcqa) (#3309)
* Average the joint-space inertia over every degree of freedom of the scene. (@duburcqa) (#3237)
* Solve the mass matrix for the active trees of a partly hibernated entity. (@duburcqa) (#3237)
* Support instantiating solver options before Genesis is initialized. (@duburcqa) (#3281)
* Let a kinematic entity be attached onto another kinematic one. (@duburcqa) (#3228)
* Remove dockerfiles. (@duburcqa) (#3227)
* Support negative indices for env and entity index args. (@jeetrex17) (#3165)
* Support Inverse Kinematics and Jacobians for any entity unconditionally. (@duburcqa) (#3233)
* Enable cross-platform window-less offscreen rendering. (@duburcqa) (#3238, #3242)
* More representative rigid benchmarks suite. (@hughperkins) (#3249, #3260)
* Add sync-free padded layout to get_contacts. (@hughperkins) (#3253)
* Speed up raycast BVH rebuild by removing atomic contention in the scene-extent reduction. (@Kashu7100) (#3162)
* Report baking USD materials failures traceback. (@duburcqa) (#3298)

## 1.3.3

This minor release fixes rigid solver convergence regressions and introduces a new torch-like option 'use_deterministic_algorithms' to ensure bit-exact reproducibility on a given machine. Besides, Mujoco compatibility mode has been improved to now cover native contact patch-based multi-contact (as opposed to the old perturbation-based approach). This experimental feature is now exposed to the user via 'RigidOptions.enable_contact_patch'.

### New Features

* Add bit-exact deterministic simulation mode on a given machine. (@duburcqa) (#3222)

### Bug Fixes

* Keep the contact manifold and the constraint solve consistent across orientations and scales. (@duburcqa) (#3158)
* Fix physics discrepancy between generic and GPU-only constraint solver implementations. (@duburcqa) (#3218)
* Fix dropped and backend-dependent contacts under MuJoCo compatibility. (@duburcqa) (#3223)

### Miscellaneous

* Add analytical sphere-sphere collision detection. (@Kashu7100) (#2963)
* Stricter Mujoco parity under compatibility mode. (@duburcqa) (#3154)
* Skip already-converged warm-started constraint solves. (@duburcqa) (#3154)
* Support PyTorch prerelease builds. (@PhysicistJohn) (#3203)
* Add documentation issue template. (@Milotrince) (#3189)
* Remove deprecated 'compile_kernels' build arg. (@Milotrince) (#3220)
* Skip Quadrants memory allocation for inactive solvers. (@Kashu7100) (#3155)
* Upgrade Quadrants to 1.3.0. (@hughperkins) (#3219)

## 1.3.2

This minor release improves compliance of rigid body simulation wrt the orientation- and scale-based physics invariants, while improving numerical robustness to ill-conditioning for the mass matrix. Besides, the simulation is now fully deterministic on a given machine for both CPU and GPU.

### New Features

* Add method for querying terrain height. (@jeetrex17) (#3128)

### Bug Fixes

* Fix delete weld constraint when not last added dynamic equality. (@jimwang418) (#3123)
* Fix viewer picking and raycast sensors missing hits. (@duburcqa) (#3147)
* Fix sensor read delay crash and jitter returning undelayed data. (@Milotrince) (#3188)
* Preserve authored inertia matrix when center of mass is unspecified. (@eason4kim-rocket) (#3184)
* Keep rigid body simulation consistent across scene orientations and scales. (@duburcqa) (#3152, #3156, #3194, #3196)

### Miscellaneous

* Cleanup examples for consistency. (@Milotrince) (#3104)
* Report a clear error when EGL is unavailable for offscreen rendering. (@himanshu748) (#3145)
* Speed up simulation with IMU sensors. (@duburcqa) (#3149)
* Share mesh cache between re-exports of the same asset. (@duburcqa) (#3151)
* Simplify per-step update of temperature sensor. (@duburcqa) (#3153)
* Speedup scene build by skipping the hollow-geom SDF probe for convex geoms. (@YilingQiao) (#3180)
* Make GPU-based rigid body simulation deterministic. (@hughperkins) (#3187)

## 1.3.1

This small release fixes support of the Nyx rendering plugin. Besides, contact normal forces no longer depend on the friction coefficient nor the sliding speed, thanks to the new rigid solver option 'contact_resolution' that bounds friction against the normal force actually developed by the contact.

### Breaking

* [FEATURE] Support decoupling contact normal forces from friction coef and sliding speed. (@duburcqa) (#3126, #3139)

### New Features

* Support raycasting against visual meshes in viewer plugins. (@Milotrince) (#3132)

### Miscellaneous

* Fix support of FEM entities in Nyx plugin. (@duburcqa) (#3131)
* Add CI for Nyx renderer plugin. (@duburcqa) (#3130, #3133, #3135, #3136, #3137, #3138)

## 1.3.0

This release extends differentiable rigid body simulation, which is no longer considered experimental. Moreover, torsional and rolling friction have been added, as part of a broader ongoing effort to bridge the sim2real gap.

### Breaking

* [BUG FIX] Record camera videos of any length at a chosen framerate. (@duburcqa) (#3107, #3120)

### New Features

* Add torsional and rolling friction to the rigid solver. (@duburcqa) (#3069)
* Add explicit mounting transform in RigidEntity.attach. (@Kashu7100) (#3041)
* Add link filtering to contact sensors. (@jsw7460) (#2772)
* Add torque visualization for tactile sensor examples. (@Milotrince) (#3085)
* Extend differentiable rigid body simulation. (@SonSang, @duburcqa) (#2842, #3087, #3089)

### Bug Fixes

* Fix missing buoyancy on rigid bodies submerged in SPH fluid. (@yeezhouyi) (#2857)
* Fix rendering assets pairing base color with emissive map. (@duburcqa) (#3088)
* Fix camera sensor not being refreshed after scene reset. (@thanyu-hub) (#3100)
* Fix transparency not deterministic in Rasterizer. (@duburcqa) (#3108)
* Fix per-environment queries of link inertial properties and energies. (@duburcqa) (#3114)
* Fix wrong mechanical energy and speed up computations. (@duburcqa) (#3121)
* Fix handling of negative index for rigid accessor filter 'envs_idx'. (@jeetrex17) (#3117)

### Miscellaneous

* Clarify IPC vertex constraint error (@coyaSONG) (#3056)
* Decouple visual from physics geometry for FEM entities. (@ACMLCZH) (#2904)
* Safer kernel caching. (@hughperkins) (#3071)
* Speed up raycast sensors by splitting the rigid collision BVH into static and dynamic subsets. (@duburcqa) (#3078)
* Add full support of AMD GPU to unit test and benchmark infra. (@v01dXYZ) (#2680)
* Reduce memory footprint of ray casting-based sensors via cross-env memory sharing. (@Kashu7100) (#2914)
* Move to gs-madrona 0.0.10. (@duburcqa) (#3093, #3097)
* Fix reverse-mode autodiff on Apple Metal. (@duburcqa) (#3119)

## 1.2.3

This small release introduces elliptic friction cone with high-impedance option to accurately model static function. In addition, the convergence of the constraint solver under fp32 accuracy has been improved.

### New Features

* Add high-fidelity static friction model via elliptic friction cone with high-impedance option. (@duburcqa) (#3028, #3035, #3042)
* Add support of heterogeneous entities in batch renderer. (@Kashu7100) (#2960)
* Add distances-only mode to raycasting-based sensors. (@Kashu7100) (#2908)
* Add option to ignore MJCF ground plane. (@coyaSONG) (#3050)

### Bug Fixes

* Fix default morph offset_quat clashing with offset_euler. (@duburcqa) (#3048)
* Fix IPC coupler auto-detecting wrong coup_type for Plane entities. (@liminchen) (#2877)
* Fix mass matrix for attached entities. (@duburcqa) (#3055)
* Fix fastcache support. (@hughperkins) (#3034, #3061)
* FiX USD parsing of geometry properties. (@Milotrince) (#3017)
* Fix USD parsing of textures packed in .usdz archives. (@connorsoohoo) (#3064)
* Fix MJCF parsing of 2D textures. (@thanyu-hub) (#3036)
* Fix MJCF parsing of included files with mesh defaults. (@Fstarnb) (#3068)
* Fix MJCF parsing of joint armatures. (@duburcqa) (#3072)
* Honour system scroll direction in viewer plugins. (@coyaSONG) (#3030)
* Release the offscreen EGL context before destroying it during teardown. (@duburcqa) (#3076)

### Miscellaneous

* Speed up non-convex collision detection with vert spatial grid and coarse SDF lower bound. (@duburcqa) (#3026)
* Speed up non-batched CPU-based simulation (up to 30%). (@hughperkins) (#3033)
* Speed up forward kinematics by removing unnecessary atomics. (@gayathiri-venkataraman) (#3066)
* Improve constraint solver float32 convergence and equality constraint stability. (@duburcqa) (#3073)
* Preserve the energy of freely tumbling bodies under implicit integration. (@duburcqa) (#3074)
* Stop requiring 64bit precision in tactile sensors. (@Milotrince, @duburcqa) (#3057, #3058)
* Pack rigid solver data on a per-component basis. (@duburcqa) (#3060)
* Fix broken Apple Metal CI. (@hughperkins) (#3054)
* Add contribution guidelines. (@duburcqa) (#3032, #3049)

## 1.2.2

This release introduces a rich family of realistic yet high-throughput tactile sensors that are suitable for training dexterous policy via RL. Besides, non-convex collision detection is now more robust than ever, with comparable performance to convex decomposition in terms of speed and stability. Notably, spurious deep contacts and thin-shell tunneling has been fixed. Finally, enabling 'noslip' post-processing step should now incur less than 20% slowdown for all backends.

### New Features

* Add noise options for tactile sensors: hysteresis, dead taxels, probe gain (@Milotrince) (#2813)

### Bug Fixes

* Fix contact-pruning bucket merging. (@duburcqa) (#3010)
* Fix performance regression on some backends when sparse option is not specified. (@duburcqa) (#3010)
* Fix data accessors for very large batch sizes (>16k). (@duburcqa) (#3010)
* Fix go2_backflip RL training example. (@kshitijgoel007) (#2986)

### Miscellaneous

* Speed up per-island noslip solver. (@duburcqa) (#3009)
* Speed up contact-island constraint solve for large batches. (@Milotrince) (#3021)
* Speed up tactile sensors. (@Milotrince) (#2922)
* More robust nonconvex collision detection. (@duburcqa) (#3014, #3020)

## 1.2.1

This release dramatically improves scaling to large scenes on CPU by better leveraging sparsity and incremental Hessian update. Scenes with 100 entities / 1000 geometries now run in real-time at 120FPS.

### Breaking changes

* [BUG FIX] Fix center of mass not properly aligned for composite links. (@duburcqa) (#2993)

### New Features

* Add JointTorqueSensor. (@matthieuvigne) (#2989)

### Bug Fixes

* Fix textures when loading GLB meshes with multiple materials. (@duburcqa) (#2984)
* More robust watertighten algorithm use for non-convex meshes and inertia estimation. (@duburcqa) (#2994)
* Fix stale deformable meshes on first rendered frame and wake-up crash during build. (@duburcqa) (#3006)

### Miscellaneous

* More robust convex collision detection. (@duburcqa) (#2999, #3007)
* Speed up CPU rigid constraint solver. (@duburcqa) (#2992, #2993, #2996, #3001, #3005)

## 1.2.0

This release dramatically improves scaling to large scenes using islands partitioning, which a now first-class and enabled by default. Besides, the simulation speed is also scaling much better with the number of degrees of freedom per entity.

### Breaking changes

* [MISC] Parse kinematic trees depth-first instead of breadth-first. (@duburcqa) (#2972)

### New Features

* Add full support of islands and hibernation. (@duburcqa) (#2972, #2974, #2975)

### Bug Fixes

* Fix race condition causing ImGui plugin to crash during scene rebuild. (@Kashu7100) (#2936)
* Fix joint-equality when attaching a floating-base entity. (@duburcqa) (#2948)
* Fix offscreen rendering using wrong resolution when interactive viewer is enabled. (@duburcqa) (#2954)
* Fix Windows offscreen rendering. (@duburcqa) (#2951, #2953, #2961)
* Deduplicate BatchRenderer textures without image_path. (@NoahLinckeScout) (#2940)
* Render each heterogeneous variant in its own environment. (@duburcqa) (#2958)

### Miscellaneous

* Add 'loop_config' kernel names for profiling. (@hughperkins) (#2950)
* Fix typos and redundant placeholder/value in bug report template. (@yeezhouyi) (#2840)
* Improve readability of entity and morph repr. (@duburcqa) (#2956)
* Raise on ambiguous morph access for heterogeneous entities. (@kunni918) (#2798)
* Speedup rigid body mass matrix cholesky decomposition for large dof count. (@Kashu7100) (#2915)
* Partition mass matrix per kinematic tree to allow factoring them separately. (@duburcqa) (#2976)
* Cache mesh processing to speed up adding many identical entities. (@duburcqa) (#2978)
* Scale hibernation ang vel threshold by DOF swept radius. (@duburcqa) (#2979)

## 1.1.2

This minor release introduces morph pose offset to finally provide a viable solution to the long-lasting inconstency of spatial accessors related to non-standard mesh axes conventions (0.3.13) and inertia alignment (0.4.4). Besides, fixed-size buffers are now sized more tightly to significantly reduce the default memory footprint.

### Breaking changes

* Add morph pose offset and relative-frame pose accessors for rigid bodies. (@duburcqa) (#2934)

### New Features

* Add pruning-aware sizing of contact constraint buffers using 'max_contacts' option. (@duburcqa) (#2928)

### Bug Fixes

* Fix island support in rigid body solver. (@duburcqa) (#2930)
* Fix serialized batched simulation on CPU scaling sub-linearly. (@duburcqa) (#2929)

### Miscellaneous

* Allocate rigid entity Jacobian and IK fields lazily on first use. (@duburcqa) (#2933)

## 1.1.1

This minor release mainly improves the robustness of rigid collision detection for both convex and non-convex geometries. It is no longer considered experimental to disable convex decomposition when higher fidelity is necessary.

### Bug Fixes

* Deduplicate textures across GLB submeshes sharing a material. (@duburcqa) (#2896)
* Skip grayscale conversion for colorized segmentation maps. (@ACMLCZH) (#2901)
* Fix non-deterministic simulation on GPU. (@duburcqa) (#2898, #2907, #2909)
* Fix non-convex collision detection for concave geometries. (@duburcqa) (#2903)
* More robust perturbation-based multi-contact convex-convex collision detection. (@duburcqa) (#2917, #2921)

### Miscellaneous

* Support passing sliced env mask to RigidSolver.set_base_links_(pos|quat). (@duburcqa) (#2897)
* Improve interactive scene mode. (@duburcqa) (#2899)
* Use qd.ndrange(axes=) to collapse layout-flip duplications. (@hughperkins) (#2861)
* Switch dedupe contact sort to use Quadrants bitonic sort (@hughperkins) (#2853)
* Move register-tile Cholesky into quadrants. (@hughperkins) (#2860)

## 1.1.0

This release focuses on performance improvement and numerical stability of rigid solver. Genesis speed should drop less aggressively as the complexity of the scene increases, notably the number of decomposed convex geoms, floating-base entities, and dofs.

### Bug Fixes

* Fix contact tunnelling when enabling grad. (@duburcqa) (#2873)
* Fix raycast sensor for heterogeneous entities. (@duburcqa) (#2876, #2891)
* Fix loading of MJCF mesh normals. (@duburcqa) (#2883)
* Fix biased convex-convex multi-contact causing drift and inefficiency. (@duburcqa) (#2889)

### Miscellaneous

* Scale contact arrow radius with object size. (@duburcqa) (#2852)
* Improve ImGui overlay plugin. (@duburcqa) (#2850, #2851, #2859, #2862, #2863, #2866)
* Add interactive scene launcher 'gs launch' to replace now deprecated 'gs view'. (@duburcqa) (#2864)
* Speedup rigid solver for large dofs count (> ~128). (@Kashu7100) (#2869)
* Speedup raycaster sensor by skipping static BVH rebuilds and share BVH across env-identical geometries. (@Kashu7100, @duburcqa) (#2867, #2875)
* Speedup rigid solver using CPU skyline-envelope sparse Cholesky. (@duburcqa) (#2879)
* Speedup raycaster-related sensors using cross-sensor shared context. (@duburcqa) (#2882, #2885)

## 1.0.0

This release adds full support of non-convex multi-contact collision detection. Besides, many components of the simulation have been sped up, incl the rigid solver that should now be up to 35% for contact-reach scenes with about 64 dofs.

### New Features

* Add robust non-convex mesh repair pipeline. (@duburcqa) (#2816)
* Add robust non-convex multi-contact collision detection. (@duburcqa) (#2812, #2821, #2823, #2825, #2828, #2844)
* Add 'ViewerOptions.enable_gui' option to easily enable ImGui overlay. (@duburcqa) (#2846, #2849)

### Bug Fixes

* Always register scene for destroy even in case of build failure. (@duburcqa) (#2807)
* Fix static-contact drift on smooth-vs-polytope contacts. (@duburcqa) (#2801, #2841)
* Fix spurious rotation on flat terrain. (@duburcqa) (#2810)
* Fix elastomer taxel broken on Apple Metal. (@duburcqa) (#2811)
* Fix inconsistency between broadphase and narrowphase. (@duburcqa) (#2838)
* Fix tilted plane rendering. (@duburcqa) (#2824)

### Miscellaneous

* Speed up point cloud tactile sensors. (@Milotrince) (#2800)
* Speed up MPM solver. (@Kashu7100) (#2720)
* Speed up rigid constraint solver. (@hughperkins) (#2809, #2817, #2820, #2827, #2837)
* Add contact pruning at link pair level. (@duburcqa, @hughperkins) (#2829, #2831, #2832, #2834, #2835)

## 0.4.7

This release introduces a new type of tactile sensors while improving the existing raycasting-based and tactile sensors. Besides, a new opt-in experimental viewer plugin to interact with the simulation is now available.

### Breaking changes

* [MISC] Default constraint solver tolerance based on precision. (@duburcqa) (#2713)

### New Features

* Add ImGui overlay for interactive joint control and visualization. (@YilingQiao, @duburcqa) (#2541, #2780)
* Add public API to set / get vertex positions of visual rigid geometries. (@duburcqa) (#2776)
* Add support of visual-mesh to camera and raycasting-based sensors. (@Kashu7100, @duburcqa) (#2769, #2783)
* Add `history_length` to Sensors (@Milotrince) (#2655)
* Add new point-cloud-based tactile sensors. (@Milotrince) (#2735)

### Bug Fixes

* Fix 'qd_zero_grad' not supporting qd.Tensor. (@duburcqa) (#2753)
* Fix zero-copy race condition on Apple Metal. (@duburcqa) (#2758)
* Fix rigid body Jacobian getter for compound joints. (@vlordier) (#2706)
* Fix ImGui interactive viewer plugin. (@Kashu7100) (#2789)
* Fix safe GJK fallback. (@duburcqa) (#2716, #2717)
* Workaround buggy float modulo on AMDGPU. (@duburcqa) (#2714)
* Reject 'CoacdOptions.pca=True' due to upstream CoACD bug (@voidborne-d) (#2757)
* Make Surface shortcut resolution idempotent. (@Kashu7100) (#2761)
* Robust device handling in slerp utils. (@Kashu7100) (#2754)

### Miscellaneous

* Speedup noslip post-processing step used to suppress slip/drift. (@erizmr) (#2671, #2672, #2703)
* Unify linesearch refinement between decomposed and monolith solver paths. (@duburcqa) (#2710)
* Improve stability of CI performance benchmarks. (@hughperkins) (#2722, #2723, #2724, #2731)
* Speedup rigid constraint solver. (@hughperkins) (#2659, #2762)
* Serialize USD bake operations to prevent deadlocks. (@hughperkins) (#2739)
* Migrate to unified Quadrant's tensor abstraction. (@hughperkins) (#2751)
* Add support of autodiff with dynamic loops. (@duburcqa) (#2742, #2743)
* Viewer plugin updates: orthographic cam mode, multi-env mouse interaction, auto dark mode. (@Milotrince) (#2728)
* Refactor sensor pipeline. (@duburcqa, @Milotrince) (#2770, #2786, #2792)
* Avoid manually sync between torch and quadrants on Metal. (@hughperkins) (#2760)
* Update Dockerfile for Ubuntu 24.04 (@alnI3S) (#2744)

## 0.4.6

This release brings recent performance benefits that where CUDA only to all GPU backends while relaxing CUDA Toolkit requirement. Besides, all reported CUDA crashes have been fixed.

### New Features

* Add public API to Scene for drawing cameras frustum and trajectories. (@Mehak261124) (#2593)

### Bug Fixes

* Fix EGL context not properly destroy during scene destruction. (@duburcqa) (#2673)
* Fix loading of GLB meshes missing normals or tex_coord. (@iory) (#2668)
* Fix sparse solve bug when building hessian. (@erizmr) (#2670)
* Fix contact overflow causing unbounded memory access. (@duburcqa) (#2688)

### Miscellaneous

* Enable parallel linesearch on all GPU backends. (@duburcqa, @hughperkins) (#2678, #2689, #2692)
* Add public API to update debug markers. (@duburcqa) (#2665)
* Make markers always foreground with XRAY effect. (@duburcqa) (#2666, #2685)
* Speed up rigid constraint solver init. (@erizmr) (#2521)
* More robust GPU detection in test infrastructure. (@Lidang-Jiang) (#2653)

## 0.4.5

This release continues on the ongoing trend of rigid body simulation speed improvements. A few camera-related bugs and all known regression on Metal backend are now fixed.

### New Features

* Add support of xacro URDF. (@duburcqa) (#2642)
* Add public API to RigidEntity for kinematic and potential energy. (@Lidang-Jiang) (#2613)
* Add support of Mujoco general actuator model. (@duburcqa) (#2641)
* Add support of zerocopy to set_pos/set_quat. (@duburcqa) (#2657)

### Bug Fixes

* Guard gradient computation not supported on Metal backend with dynamic array mode. (@duburcqa) (#2628)
* Fix plotter video export race condition. (@duburcqa) (#2647)
* Fix sensor camera 'lookat' being ignored when 'entity_idx' is set. (@Lidang-Jiang) (#2614)
* Fix various regressions on Metal backend. (@duburcqa) (#2651, #2624, #2657)
* Fix camera sensor per-env rendering with Rasterizer. (@duburcqa) (#2657)

### Miscellaneous

* Add parallel linesearch for constraint solver to speedup simulation on GPU backend. (@erizmr) (#2523)
* Speedup tiled hessian kernel by using direct lower-triangle indexing. (@hughperkins) (#2618)
* Add broadphase all-vs-all to speedup simulation on GPU backend. (@hughperkins) (#2607)
* Add GPU graph to decomposed solver to reduce kernel launch latency. (@hughperkins, @duburcqa) (#2621, #2635, #2636)
* Add support of opt-in shared memory for tiled hessian to improve performance. (@duburcqa) (#2629)
* Better parallelization of add collision constraints. (@hughperkins) (#2639)
* Enable GPU-optimised decomposed constraints solver implementation on all GPU backends. (@duburcqa) (#2623)
* Unify narrowphase codepath on all GPU backends. (@duburcqa) (#2637)
* Automatically select optimal H264 codec for Video recorder. (@duburcqa) (#2657)
* Update all RL examples. (@duburcqa) (#2644, #2657)
* Update docker container. (@duburcqa) (#2643)

## 0.4.4

The numerical stability of the simulation for simple rigid objects has been greatly improved. Apart from that, rigid body simulation is now much faster for complex scenes with many entities. Finally, a significant number of bugs have been fixed.

### Breaking changes

* [FEATURE] Improve numerical stability of simulation by aligning inertial axes of free joints. (@duburcqa) (#2569, #2571, #2573, #2575)
* [BUG FIX] Fix default armature incorrectly applied on free joints for MJCF. (@duburcqa) (#2584)
* [MISC] Set 'RigidEntity.set_quat(..., relative=True)' by default. (@duburcqa) (#2592)
* [MISC] More realistic default material density for Rigid. (@duburcqa) (#2378)

### New Features

* Support batched camera for Rasterizer. (@duburcqa) (#2564)
* Add MacOS support for separated envs rendering with Rasterizer. (@duburcqa) (#2560)

### Bug Fixes

* Fix flickering issue for temperature sensor debug draw. (@duburcqa) (#2557)
* Fix batched env separated rendering with Rasterizer.
* Fix interactive viewer race condition when running background thread. (@duburcqa) (#2585)
* Fix color overwrite for MJCF without visuals. (@duburcqa) (#2586)
* Fix batched sensor read when combining multiple sensor types. (@Milotrince) (#2581)
* Fix USD parsing. (@duburcqa) (#2594)
* Fix GPU synchronization issue on Apple Metal. (@duburcqa) (#2600)
* Fix degenerated invweight computation. (@duburcqa) (#2598)
* Fix viewer plugin registration after build. (@duburcqa) (#2601)
* Fix motion planning crashing for short path < 3 nodes. (@Lidang-Jiang) (#2610)
* Fix mouse interaction void ray casting. (@duburcqa) (#2611)
* Fix IK solver using wrong entity's DOFs for multi-robot scenes. (@Lidang-Jiang) (#2612)

### Miscellaneous

* Improve mouse interaction visualization. (@duburcqa) (#2574)
* Add support of batching to 'RigidLink.set_mass'. (@duburcqa) (#2578)
* Refactor rigid benchmarks into reusable scene factories. (@hughperkins) (#2577)
* Re-raise viewer exception running in background thread. (@duburcqa) (#2583)
* Support non-blocking scene reset for rigid solver. (@duburcqa) (#2580)
* Support more recent GPU devices by migrating Quadrants to LLVM 22. (@hughperkins) (#2595)
* Migrate materials to new pydantic based options with strict validation. (@duburcqa) (#2597)
* Disable perf dispatch re-benchmarking to reduce performance penalty. (@erizmr) (#2599)
* Tune kernel dispatch heuristics to reduce wrong selection. (@erizmr) (#2605)
* Rename 'Drone.set_propellels_rpm' in 'Drone.set_propellers_rpm'. (@Lidang-Jiang) (#2609)
* Speed up simulation by tuning tiled Hessian block size. (@hughperkins) (#2617)
* Optimize performance of collision detection using hardware-derived thread count. (@hughperkins) (#2616)

## 0.4.3

This release introduces more sensors while significantly speeding up collision detection on GPU (up to 30%). As usual, a few bugs have been fixed.

### New Features

* Add ProximitySensor. (@Milotrince) (#2550)
* Add TemperatureGridSensor. (@Milotrince) (#2457)

### Bug Fixes

* Fix rendering FEM vertex normals in interactive viewer. (@duburcqa) (#2555)
* Fix support of ellipsoid geometry. (@duburcqa) (#2554)
* Fix mujoco compatible URDF parsing. (@duburcqa) (#2547)
* Fix typing, validation, and compatibility issues with structured options. (@duburcqa) (#2548)

### Miscellaneous

* Speed up rigid collision detection on GPU for generic convex meshes. (@hughperkins) (#2527)
* Fix spurious Quadrants compilation warnings. (@duburcqa) (#2546)
* Improve typing of 'add_entity', 'add_sensor'. (@duburcqa) (#2552)
* Add init/destroy registration mechanism for external modules. (@duburcqa) (#2551)
* More flexible and reliable sensor plugin registration mechanism. (@duburcqa) (#2553)

## 0.4.2

This release introduces a new type of tactile sensors based on [FOTS](https://arxiv.org/pdf/2404.19217), and add support of parallel simulation of heterogeneous articulated robots. Beyond that, the performance of the simulation has been significantly improved, especially when rigid option 'noslip' is enabled.

### New Features

* Add ElastomerDisplacementSensor. (@Milotrince) (#2447)
* Support articulated assets in heterogeneous parallel simulation. (@Kashu7100, @ACMLCZH) (#2472, #2535)

### Bug Fixes

* Fix GJK multi-contact in mujoco-compatibility mode. (@SonSang, @duburcqa) (#2514, #2516)
* Add support of morph option "fixed" for USD without explicit root and filter out invisible geometries from collision. (@ACMLCZH) (#2528)
* More robust collision detection on 32bits precision. (@SonSang) (#2525)
* Fix parsing of URDF with undefined inertial properties. (@duburcqa) (#2544)

### Miscellaneous

* Propagate interactive viewer exceptions when running in thread. (@duburcqa) (#2510)
* More robust and versatile interactive viewer keybinding. (@duburcqa) (#2512)
* Migrate 'gs view' to kinematic entity to reduce compile time. (@duburcqa) (#2522)
* Fix Python-scope field read to avoid GPU-CPU sync. (@erizmr) (#2518)
* Faster initialisation of collision pairs validity mask. (@ACMLCZH) (#2534)
* Introduce structured options with validation. (@duburcqa) (#2536)
* Accelerate noslip simulation on GPU backend. (@erizmr) (#2532)
* Speedup rigid constraint solver. (@erizmr) (#2524)
* Add dexterous hand benchmark. (@hughperkins) (#2500)

## 0.4.1

This release mainly improves experimental IPC coupler integration. Avatar entity has been re-introduced after being broken for months. Finally, Quadrants' dynamic arrays are now support on Apple Metal to avoid systematic scene compilation.

### Breaking changes

* More consistent IPC coupler API. (@duburcqa) (#2446)

### New Features

* Implement Local Offset in Inverse Kinematics (@alexis779) (#2333)
* Add support of batching to IPC articulated bodies. (@Roushelfy, @duburcqa) (#2448, #2454)
* Add support per-entity contact friction and resistance to IPC coupler. (@Roushelfy, @duburcqa) (#2473, #2479)
* Enable Quadrants dynamic array on Apple Metal. (@hughperkins) (#2498)
* Add avatar (kinematic) entity and refactor rigid entity. (@duburcqa) (#2504, #2508)

### Bug Fixes

* Fix support of Numpy 2.4. (@duburcqa) (#2432, #2433)
* Fix geometries misclassified as non-convex. (@duburcqa) (#2442)
* Fix merge fixed links. (@duburcqa, @YilingQiao) (#2441, #2475)
* Fix wrong type for gs.materials.Rigid.gravity_compensation. (@nimrod-gileadi) (#2469)
* Fix UV size mismatch causing USD parsing hard failure. (@alanray-tech) (#2480)
* Fix support of convex decomposition baked in glTF. (@duburcqa) (#2493)
* Fix mass getter for PBD entities. (@YilingQiao) (#2503)
* Fix Apple Metal bug causing nan. (@hughperkins) (#2481)
* Raise exception in case of invalid IPC coupler options. (@duburcqa) (#2450, #2451, #2486, #2492)

### Miscellaneous

* Refactor internal implementation of IPC coupler. (@duburcqa, @ACMLCZH, @Roushelfy) (#2453, #2455, #2462, #2463, #2465, #2478, #2484, #2485, #2495, #2496)
* More comprehensive IPC coupler unit tests and tune examples. (@duburcqa) (#2444, #2458, #2474, #2488)
* Add support of unspecified inertia origin in URDF. (@duburcqa) (#2499)
* Add torch profile integration to performance benchmarks. (@hughperkins) (#2421, #2466)
* Significantly speedup contact-rich and dynamic simulations. (@hughperkins) (#2406, #2467)
* More comprehensive performance benchmarks. (@hughperkins) (#2470)
* Aggregate speed benchmark wandb uploads into a single run. (@hughperkins) (#2482)

## 0.4.0

This release polishes our newly introduced USD parser and external interactive viewer plugin mechanism. Beyond that, the performance of the simulation has been significantly improved for collision-heavy scenes (up to 30%) and robots using capsule/sphere collision geometries (up to 20%).

### Breaking changes

* Migrate from GsTaichi to Quadrants. (@hughperkins, @duburcqa) (#2399, #2409)

### New Features

* Replace partial broken Vulkan backend by (experimental) AMD ROCm backend. (@duburcqa) (#2393, #2402)
* Support ArticulationRoot-free robots and instanced prims in USD. (@alanray-tech) (#2411)
* Add KinematicContactProbe sensor. (@Milotrince) (#2389)
* Add support of USD in 'gs view'. (@duburcqa) (#2423)
* Add IPC Robot-Cloth Coupling. (@Roushelfy, @alanray-tech) (#2427, #2352)

### Bug Fixes

* Fix weld and connect equality constraints. (@duburcqa) (#2390)
* Fix color overwrite for primitive geometries in URDF. (@duburcqa) (#2403)
* Fix merging chained fixed links. (@duburcqa) (#2404, #2429)
* Fix handling of unspecified spatial inertia properties in USD. (@alanray-tech) (#2401)
* Tune GJK to avoid false negative collision detection. (@SonSang) (#2419)
* Fix object slowly falling when holding object using mouse interaction plugin. (@Milotrince) (#2418)
* Fix interactive viewer deadlock on Wayland. (@alexis779) (#2422)
* Fix default file extension for interactive viewer save. (@duburcqa) (#2424)
* Fix viewer raycast kernel compilation race condition. (@duburcqa) (#2430)

### Miscellaneous

* Speed up linesearch via reducing global reads by recomputing quad. (@erizmr) (#2384)
* Speed up capsule-capsule and capsule-sphere collision detection. (@hughperkins) (#2363, #2413, #2414)
* Lazy-initialize SDF pre-processing in RigidGeom. (@duburcqa) (#2392)
* Add uv installation instructions to translated READMEs. (@antoinedandi) (#2416)
* Fix support of fastcache for raycast kernels. (@duburcqa) (#2417)

## 0.3.14

This release mainly focuses on usability, by extending support of USD and introducing a new external plugin mechanism for the interactive viewer. Besides, the performance of the simulation has been significantly improved for collision-heavy scenes (up to 30%).

### New Features

* Introduce interactive viewer plugins. (@Milotrince) (#2004, #2357)
* Add naming logics to entities. (@YilingQiao) (#2303)
* Support rendering textures for USD scenes. (@ACMLCZH) (#2286)
* Add invalid spatial inertia diagnosis. (@duburcqa) (#2297, #2321, #2367)

### Bug Fixes

* Fix wrong default sampler for SPH solver causing numerical stability issues. (@erizmr) (#2280)
* Fix render destroy. (@nimrod-gileadi, @duburcqa) (#2282, #2358)
* Fix IPC Coupler. (@duburcqa) (#2299)
* Fix batched numpy 'euler_to_R' geom util. (@Kashu7100) (#2306)
* Fix glTF mesh loading. (@duburcqa) (#2296, #2311, #2316, #2329)
* Fix 'Mesh.convert_to_zup' by applying scaling out-of-place. (@duburcqa)
* Fix rigid body entity hibernation mechanism. (@YilingQiao) (#2294)
* Fix DFSPH solver. (@erizmr) (#2302)
* Fix parsing material of primitive geometries in MJCF files. (@ACMLCZH) (#2328)
* Fix 'draw_debug_frames' after PR#1869. (@duburcqa) (#2330)
* Fix compatibility with 'trimesh<4.6.0'. (@duburcqa) (#2334)
* Fix linesearch edge-case fallback. (@erizmr) (#2339)
* Fix mujoco-compatible GJK multi-contact for box primitive. (@hughperkins) (#2341)
* FIX FEM entity rendering with Raytracer backend. (@duburcqa) (#2356)
* Fix combining Rasterizer-based camera sensors and interactive viewer. (@YilingQiao) (#2351)

### Miscellaneous

* Track mesh UVs in FEM/PBD solvers for rendering of deformable entities. (@alelievr) (#2323)
* Support scrolling menu in 'gs view'. (@Kashu7100) (#2335)
* Add memory to CI performance monitoring report. (@hughperkins) (#2281, #2291, #2293, #2295, #2298, #2300, #2312, #2315, #2320)
* Improve support of Linux ARM. (@duburcqa) (#2317)
* Clearer error message of 'RigidEntity.(get_joint|get_link)'. (@Kashu7100) (#2313)
* Add Markdown Files to Facilitate AI Tools. (@YilingQiao) (#2305)
* Support --record in the RL stage in Manipulation example (@SnakeOnex) (#2344)
* More robust OpenGL context initialisation for Rasterizer. (@duburcqa) (#2354)
* Add benchmark for Unitree G1. (@hughperkins) (#2310)
* Recomputing inertia for primitive geometries using analytical formula. (@duburcqa) (#2337)
* Speed up linesearch via batched alpha evals and reduced global memory access. (@erizmr) (#2350)
* Disable shadow and plane reflection when using software rendering. (@duburcqa) (#2365)
* Workaround for 'pyglet' bug. (@duburcqa) (#2385)

## 0.3.13

This small release adds user-friendly diagnosis of invalid Rigid physics properties and improves support of GLTF meshes.

### Breaking changes

* Apply 'FileMorph.file_meshes_are_zup' to all meshes including GLTF. (@duburcqa) (#2275)
* Do not officially support importing GLTF morph Mesh as Z-UP. (@duburcqa) (#2279)

### New Features

* Add Magnetometer measurement to IMU Sensor. (@sunkmechie) (#2265)
* Check validity of links spatial inertia and support forcing computation from geoms. (@duburcqa) (#2273, #2276)

### Bug Fixes

* Improve support of attached RigidEntity. (@duburcqa) (#2256, #2259)
* Fix attaching RayTracer camera sensor. (@duburcqa) (#2266)
* Fix empty data when adding more than 3 Raycast Sensors. (@JackLowry) (#2268)
* Fix Raycast Sensor for batched environments. (@d-corsi) (#2269)
* More robust filtering of self-collision in neutral configuration.  (@duburcqa) (#2278)

### Miscellaneous

* Improve performance and add torch support to 'utils.geom.slerp'. (@Kashu7100) (#2260)

## 0.3.12

This PR focuses on performance improvements (x4 faster for complex scenes with 64 < n_dofs < 96 and n_envs=4096 compared to 0.3.10). Besides, initial support of heterogenous object and USD stage import for rigid body simulation has been introduced.

### New Features

* Add method to compute axis-aligned bounding boxes of visual geometries. (@duburcqa) (#2185)
* Add partial support of batched camera sensor with Rasterizer. (@Narsil) (#2207, #2212)
* Add support for attaching MPM particles to rigid links. (@YilingQiao) (#2205)
* Add support of USD import for Rigid Body. (@alanray-tech) (#2067)
* Add support of batched textures to BatchRenderer. (@ACMLCZH) (#2077)
* Add support of fisheye camera mode to BatchRenderer. (@ACMLCZH) (#2138)
* Add batched simulation of heterogeneous objects. (@YilingQiao) (#2202)
* Filter out self-collision pairs active in neutral configuration. (@duburcqa) (#2251)

### Bug Fixes

* Fix zero-copy for fields on Apple Metal. (@duburcqa) (#2188, #2223)
* Fix compatibility with 'numpy<2.0'. (@duburcqa) (#2197)
* Fix invalid default particle sampler on Linux ARM. (@duburcqa) (#2211)
* Fix 'RigidGeom.get_(pos|quat)' invalid shape. (@duburcqa) (#2218)
* Fix various sensor bugs and add zero-copy to contact force sensors. (@duburcqa) (#2232, #2235)
* Clear dynamic weld at scene reset. (@YilingQiao) (#2233)
* Fix viewer not closed at scene destroy if running in background thread. (@duburcqa) (#2236)
* More robust handling of corrupted cache. (@duburcqa) (#2241)

### Miscellaneous

* More intuitive visualisation of camera frustum in interactive viewer. (@duburcqa) (#2180)
* Remove broken and unmaintained Avatar Solver. (@duburcqa) (#2181)
* Speedup non-tiled hessian cholesky factor and solve. (@duburcqa) (#2182, #2183)
* Force public getters to return by-value to avoid mistake. (@duburcqa) (#2184)
* Improve CI infrastructure. (@hughperkins, @duburcqa) (#1981, #2166, #2190, #2194, #2195, #2242, #2245, #2250)
* Add Ruff format. (@Narsil) (#2213, #2214, #2215)
* Store texture path for primitive morphs as metadata. (@Rush2k) (#2227)
* Improve import logics of y-up vs z-up file meshes. (@AnisB) (#2237)
* Simplify boolean contact sensors update logics. (@duburcqa) (#2238)
* Rigid methods set_qpos,set_dofs_position now clear error code. (@duburcqa) (#2253)

## 0.3.11

The main focus of this release is to improve scaling of the simulation wrt the complexity of the scene, and better leverage GPUn compute for small to moderate batch sizes (0<=n_envs<=8192). As usual, a bunch of minor bugs have been fixed.

### New Features

* Support specifying offset transform for camera sensor. (@YilingQiao) (#2126)
* Enable zero-copy for fields on Metal if supported. (@duburcqa) (#2174)

### Bug Fixes

* Avoid discontinuities in smooth animations caused by singularities. (@Rush2k) (#2116)
* Fix forward update logics. (@duburcqa) (#2122)
* Fix kernel caching mechanism hindering performance. (@duburcqa) (#2123)
* Fix support of old torch for 'set_dofs_velocity' when velocity=None. (@YilingQiao) (#2160)
* Force rendering systematically when updating camera sensor. (@YilingQiao) (#2162)
* Fix incorrect lighting when offscreen cameras based on rasterizer. (@duburcqa) (#2163)
* Fix rasterizer race conditions when running in background thread. (@duburcqa) (#2169)
* Fix broken exception handling when loading obj files with unsupported face type. (@Kashu7100) (#2170)
* Fix 'pysplashsurf' memory leak causing OOM error. (@duburcqa) (#2173, #2176)
* Diagnose out-of-bound SDF gradient index. (@duburcqa) (#2177)

### Miscellaneous

* Stop assessing warmstart vs smooth acc at constraint solver init. (@duburcqa) (#2117)
* Speedup collision detection broad phase on GPU. (@duburcqa) (#2128)
* More comprehensive benchmarks. (@duburcqa) (#2137)
* Accelerate constraint solver first pass using shared memory. (@duburcqa) (#2136, #2140)
* Further optimize cholesky solve using warp reduction and memory padding. (@duburcqa) (#2145, #2146)
* Improve runtime speed by optimize memory layout of constraint solver. (@duburcqa) (#2147)
* Fast mass matrix factorisation on GPU using shared memory. (@duburcqa) (#2154)
* Optimize rigid body dynamics to scale better wrt dofs and entities. (@duburcqa) (#2161)
* Fix spurious deprecated property warnings during introspection. (@duburcqa) (#2168)
* Various solver refactoring to support GsTaichi Main. (@hughperkins, @duburcqa) (#2131, #2135, #2143, #2151)
* Improve single-threaded cpu-based simulation runtime speed by upgrading gstaichi. (@hughperkins) (#2129, #2153)

## 0.3.10

Small release mainly fixing bugs.

### Bug Fixes

* Fix parsing for special material properties in glTF meshes (@duburcqa) (#2110)

### Miscellaneous

* More robust detection of invalid simulation state. (@duburcqa) (#2112)

## 0.3.9

Small release mainly polishing features that were introduced in previous release.

### Breaking changes

* Replace SDF fallback by GJK. (@duburcqa) (#2081)
* Improve inertial estimation if undefined. (@YilingQiao) (#2100)

### New Features

* Add support of boolean masking as index. (@duburcqa) (#2087)
* Fix and improve merging of rigid entities. (@duburcqa) (#2098)

### Bug Fixes

* Fix increased memory usage due to differentiable simulation. (@duburcqa) (#2074)
* Fix 'envs_idx' in motion planning. (@duburcqa) (#2093)
* Fix 'DroneEntity.set_propellels_rpm'. (@duburcqa) (#2095)
* Fix extended broadcasting. (@duburcqa) (#2096)
* Fix 'RigidEntity.set_dofs_velocity'. (@robin271828) (#2102)
* Fix joint stiffness not taking into account neutral position. (@YilingQiao) (#2105)
* Fix explicit URDF material color being ignored. (@duburcqa) (#2107)

### Miscellaneous

* Speed up torch-based geom utils via 'torch.jit.script'. (@duburcqa) (#2075)
* Improve scalability wrt number of contacts. (@duburcqa) (#2085, #2103)
* Make Go2 RL env GPU-sync free. (@duburcqa) (#2092)

## 0.3.8

The performance of data accessors have been dramatically improved by leveraging zero-copy memory sharing between GsTaichi and Torch. Beyond that, the robustness of the default contact algorithm has been improved, and differentiable forward dynamics for Rigid Body simulation is not partially available. Last, but not least, GsTaichi dynamic array mode is finally enabled back by default!

### Breaking changes

* More robust MPR+SDF collision detection algorithm. (@duburcqa) (#1983, #1985)
* Disable box-box by default. (@duburcqa) (#1982)

### New Features

* Enable back GsTaichi dynamic array mode by default except for MacOS. (@duburcqa) (#1977)
* Add error code to rigid solver. (@duburcqa) (#1979)
* Add option to force batching of fixed vertices. (@duburcqa) (#1998)
* Leverage GsTaichi zero-copy in data accessors. (@duburcqa) (#2011, #2019, #2021, #2023, #2025, #2030, #2037, #2048, #2054)
* Add an option to disable keyboard shortcuts (@YilingQiao) (#2026)
* Add support of 'capsule' primitive in URDF file. (@duburcqa) (#2045)
* Add full support of tensor broadcasting in getters. (@duburcqa) (#2051)
* Add rasterizer, batch renderer, and raytracer as sensor (@YilingQiao) (#2010)
* Differentiable forward dynamics for rigid body sim. (@SonSang) (#1808, #2063, #2068)

### Bug Fixes

* Fix sensor IMU accelerometer signal. (@Milotrince) (#1962)
* Fix 'RigidJoint.(get_anchor_pos | get_anchor_axis)' getters. (@alexis779) (#2012)
* Prevent nan to propagate in position and raise exception. (@duburcqa) (#2033)
* Fix camera following entity for 'fix_orientation=True'. (@duburcqa) (#2038)
* Fix support of Hybrid entity with non-fixed base link. (@duburcqa) (#2040)
* Raise exception if trying to load PointCloud as Mesh. (@duburcqa) (#2042)
* Fix boolean mask inversion for PyTorch 2.x (@yoneken) (#2056)
* Fix URDF color overwrite. (@duburcqa) (#2065)

### Miscellaneous

* Reduce memory footprint. (@duburcqa) (#2000, #2031)
* Only enable GJK by default if gradient computation is required. (@duburcqa) (#1984)
* Bump GsTaichi Support Nvidia GPU Blackwell. (@johnnynunez) (#2002)
* Add dependency version upper-bound 'tetgen< 0.7.0'. (@YilingQiao) (#2029)
* Bump up min version requirement for Torch after introducing zero-copy. (@duburcqa) (#2034)
* Add 'parse_glb_with_zup' option to all file-based Morph. (@ACMLCZH) (#1938)
* Enable more example scripts in CI. (@duburcqa) (#2057)
* Fix fast cache and zero-copy bugs. (@hughperkins) (#2050)

## 0.3.7

The performance of GsTaichi dynamic array mode has been greatly improved. Now it should be on par with fixed-size array mode (aka performance mode) for very large batch sizes, and up to 30% slower for non-batched simulations. This mode is still considered experimental and must be enabled manually by setting the env var 'GS_ENABLE_NDARRAY=1'. Just try it if you are tired of endlessly waiting for the simulation to compile!

### New Features

* Implement position-velocity controller. (@matthieuvigne) (#1948)

### Bug Fixes

* Fix missing option `diffuse_texture` to `Glass` surface. (@Kashu7100) (#1934)
* Fix interactive viewer. (@YilingQiao) (#1931)
* Fix external coupling forces from other solvers not affecting rigid bodies. (@SonSang) (#1941)
* Fix silent process killing issue in MPM simulation by raising an exception. (@SonSang) (#1949)
* Fix 'discrete_obstacles_terrain' being completely flat. (@jgillick) (#1972)

### Miscellaneous

* Added warning message about stable timestep for SPH solver. (@SonSang) (#1925)
* Reduce memory usage due to diff constraint solver. (@YilingQiao) (#1930)
* Faster non-batched simulation. (@duburcqa) (#1935)
* Fix or silent dev warnings. (@duburcqa) (#1944)
* Add caching to Rigid Link state getters to improve performance. (@duburcqa) (#1940, #1955)
* Add support of Linux ARM. (@duburcqa) (#1961)
* Add 'GS_PARA_LEVEL' env var to force kernel parallelization level. (@duburcqa) (#1968)

## 0.3.6

A new experimental interface with the Incremental Potential Contact coupling solver [libuipc](https://github.com/spiriMirror/libuipc) has been introduced, mainly targeting cloth simulation.

### New Features

* Add Rigid ‘get_dofs_frictionloss’ public API method. (@ax-anoop) (#1904)
* Add IPC (Incremental Potential Contact) coupling system. (@Roushelfy) (#1859)

### Bug Fixes

* Fix missing dependencies in Docker (@schlagercollin) (#1896)
* Fix sensor recorder Matplotlib plotter on Linux and Windows. (@duburcqa) (#1894)
* Fix particle emitter failure if batch size = num emit particles. (@duburcqa) (#1901)
* Fix MJCF handling of visual groups. (@duburcqa) (#1902)

### Miscellaneous

* Disable GsTaichi dynamic arrays by default (again!) (@duburcqa) (#1915)
* Reset control during reset. (@YilingQiao) (#1920)

## 0.3.5

Minor release mainly aiming at polishing existing features and addressing some major performance regression that was introduced end of august. GsTaichi dynamic array type and fast cache are now enabled by default on Linux and Windows (opt-in on MacOS via env variable 'GS_ENABLE_NDARRAY'), which should help avoiding recompilation in most cases.

### New Features

* Make convex decomposition cache scale-invariant. (@SonSang) (#1810)
* Add parsing of joint friction and damping for URDF files. (@duburcqa) (#1833)
* Differentiable constraint solver (@SonSang) (#1733)
* Add support of GsTaichi dynamic array type and fast cache mode. (@YilingQiao, @duburcqa) (#1868, #1873, #1875, #1880)

### Bug Fixes

* Fix hybrid entity with rigid. (@duburcqa) (#1819)
* Reduce general memory usage. (@hughperkins) (#1828)
* Improve reliability and performance of plotters. (@duburcqa) (#1836)
* Fix loading MJCF file with includes. (@YilingQiao, @duburcqa) (#1838, #1840)
* Fix DepthCamera sensor. (@Milotrince) (#1842)
* Fix and refactor named terrain. (@duburcqa) (#1845)
* Raise exception in case of invalid set entity pos/quat. (@duburcqa) (#1847, #1869)
* Fix gltf Loading for URDF. (@ACMLCZH) (#1857)
* Fix friction loss causing nan on Apple GPU. (@duburcqa) (#1860)
* Fixed minor bug in differentiable contact detection. (@SonSang) (#1864)
* Fix race condition during concurrent viewer refresh and draw_debug_*. (@duburcqa) (#1883, #1884)
* Enable GPU and fix dt in some example scripts. (@schlagercollin) (#1890)

### Miscellaneous

* Reduce memory usage of Raycaster sensor. (@Milotrince) (#1850)
* Add 'get_particles_pos' helper method to SPHEntity. (@YilingQiao) (#1871)
* Enabling gstaichi fast cache by default. (@hughperkins) (#1885)
* Refactor Rigid simulation data management. (@duburcqa) (#1888)
* Do not silent errors during build. (@duburcqa) (#1889)

## 0.3.4

This minor release mainly introduces first-class sensor support (IMU, Contact Sensor, LiDAR, Depth camera and more), incl. recording and plotting facilities. The rigid-rigid hydroelastic contact model has also been added. As usual, a fair share of bugs have been fixed, with unit test coverage gradually improving.

### Behavior Changing

* Support rendering deformable body for batched env. (@YilingQiao) (#1697)
* More sensible defaults for camera far, near. (@duburcqa) (#1678)
* Fix invweight and meaninertia not always considering scale and dofs armature. (@duburcqa) (#1696)

### New Features

* Refactor 'FrameImageExporter' to improve performance and support normal & segmentation. (@duburcqa) (#1671)
* Add support of normal & segmentation for Madrona Batch Rendering. (@ACMLCZH) (#1563)
* Add 'noslip' optional post-processing step to suppress slip/drift. (@YilingQiao) (#1669)
* Add first-class data recorders and plotters. (@Milotrince) (#1646, #1718)
* Add rigid-rigid hydroelastic contact model. (@Libero0809) (#1572)
* Add option to display sensor information in the interactive viewer. (@Milotrince) (#1770)
* Add support of differentiable contact detection (Work In Progress). (@SonSang) (#1701)
* Add Raycaster sensor (Lidar and DepthCamera). (@Milotrince, @duburcqa, @jgillick) (#1726, #1772, #1809, #1815)
* Add full support gstaichi ndarray to Rigid Body solver. (@YilingQiao, @SonSang, @duburcqa) (#1674, #1682, #1683, #1690, #1693, #1695)
* Add full support gstaichi fast caching mechanism. (@hughperkins, @YilingQiao) (#1709, #1720, #1730, #1812)

### Bug Fixes

* Fix data races in getting contact and equality constraints. (@YilingQiao) (#1676)
* Fix MPM muscle activation. (@YilingQiao) (#1692)
* Fix non-flat terrain support. (@Kashu7100, @YilingQiao, @duburcqa) (#1691, #1777, #1779)
* Disable mesh processing when loading URDF for consistency. (@duburcqa) (#1708)
* Fix segfault at exit when running viewer in background thread with offscreen cameras. (@duburcqa) (#1703)
* Fix all the example scripts. (@YilingQiao, @duburcqa) (#1743, #1773, #1724, #1785, #1787, #1801, #1804)
* Fix logics for duplicating collision geometries as visual in MJCF. (@hokindeng, @duburcqa) (#1732, #1750)
* Randomize uniform terrain along both axes. (@jgillick) (#1747)
* Fix contact sensors always returning zeros. (@Milotrince) (#1761)
* Fix LBVH stuck in infinite loop for small number of AABBs. (@duburcqa) (#1766)
* Fix broken interactive viewer backend fallback mechanism. (@duburcqa) (#1797)
* Fix camera follow entity. (@duburcqa) (#1805)
* Fix compound joints for 'set_dofs_position'. (@duburcqa) (#1678)
* Fix some mesh-related issues (@ACMLCZH) (#1800)

### Miscellaneous

* Fix support of 'pyglet<2.0'. (@Kashu7100) (#1670)
* Add vision-based manipulation example (@yun-long) (#1493)
* Reduce max_collision_pairs to save memory (@YilingQiao) (#1672)
* Remove 'gs clean' utility. (@duburcqa) (#1723)
* Rename 'get_aabb' in 'get_AABB' and add deprecation warning. (@duburcqa) (#1778)
* Improve interactive viewer performance. (@duburcqa) (#1784)
* Raise exception in cause of particle sampling failure. (@duburcqa) (#1792)
* Remove 'is_free' that was confusing and partially redundant with 'is_fixed'. (@duburcqa) (#1795)

## 0.3.3

This minor release fixes a few non-blocking rendering issues for the Rasterizer backend.

### Bug Fixes

* Fix shadow map not properly rendered for objects far away from floor plane. (@duburcqa) (#1664)
* Fix genesis import failure if tkinter is failing at init on MacOS. (@duburcqa) (#1666)
* Fix default visualization mode for emitter surface. (@duburcqa) (#1665)

### Miscellaneous

* Expose parameters for ground plane tiling. (@yuhongyi) (#1657)
* Add support of 'ti.ndarray' to 'ti_field_to_torch' and rename in 'ti_to_torch'. (@duburcqa) (#1661)

## 0.3.2

This minor release fixes a few additional regressions and initiates migration to our own open-source fork of Taichi, [GsTaichi](https://github.com/Genesis-Embodied-AI/gstaichi) (contributions are welcome!).

### Behavior Changing

* Disable decimation if deemed unnecessary and unreliable. Reduce default aggressiveness. (@duburcqa) (#1644)
* Fix primitive and mesh's COM. (@YilingQiao) (#1638)

### New Features

* Add 'set_dofs_frictionloss' method for dynamic joint friction control. (@LeonLiu4) (#1614)
* Add initial experimental support of gstaichi fast cache feature. (@hughperkins) (#1631)
* Add 'ref' optional argument to 'get_links_pos'. (@YilingQiao) (#1638)

### Bug Fixes

* Fix Inverse Kinematics algorithm. (@Kashu7100, @duburcqa) (#1582, #1586)
* Fix save frame as png image in interactive viewer. (@Kashu7100) (#1606)
* Filter out collision pairs involved in weld equality constraints. (@duburcqa) (#1621)
* Fix viewer backend fallback when running in main thread. (@duburcqa) (#1630)
* Fix 'quat_to_xyz' singularity edge-case. (@duburcqa) (#1628)
* Fix CUDA runtime being initialized by 'get_device'. (@duburcqa) (#1634)

### Miscellaneous

* Re-enable world-frame in the 'gs view' standalone viewer. (@Kashu7100) (#1584)
* Migrate from 'taichi' to 'gstaichi'. (@duburcqa, @hughperkins) (#1550, #1618, #1645)
* Update documentation so the doc will be compiled based on the latest main. (@YilingQiao) (#1616)

## 0.3.1

This small release addresses the most pressing regressions that has been pointed out by the community since 0.3.0. Support of coupling between Rigid Body and FEM has been improved and should be more reliable, though it is still considered experimental for now. Apart from that, no behavior changes are to be expected.

### New Features

* Support 2-channel (LA) textures in Rasterizer. (@LeonLiu4) (#1519)
* Add 'get_weld_constraints' API. (@LeonLiu4) (#1370)
* Add USD Materials Baking. (@ACMLCZH) (#1300)
* Add dedicated sensor manager. (@Milotrince) (#1518)
* Enhance SAP coupler for coupling between Rigid body and Fem object. (@Libero0809) (#1458)
* Add IMU sensor. (@Milotrince) (#1551)
* Add Fem fixed constraint for implicit solver. (@Libero0809) (#1562)
* Add Joint Equality Constraints for the SAP Coupler. (@Libero0809) (#1565)

### Bug Fixes

* Fix point-cloud rendering from Camera depth map. (@@ceasor-mao, @duburcqa) (#1512, #1515)
* Fix various rendering bugs. (@duburcqa) (#1537)
* Fix Z-up orientation and vertex color. (@ceasor-mao) (#1540)
* Fix hibernation. (@gasnica) (#1542)
* Fix backend fallback mechanism causing deadlock in Rasterizer. (@duburcqa) (#1546)
* Fix video recording dialog when running viewer in thread. (@duburcqa) (#1547)
* Fix joint friction loss. (@duburcqa) (#1555)
* Fix taichi debug mode errors and warnings. (@Libero0809) (#1560)

### Miscellaneous

* Migrate to native Python API for 'splashsurf'. (@duburcqa) (#1531)
* Refactor weld constraint API. (@duburcqa) (#1536)
* Allow following entity or mounting camera building scene. (@duburcqa) (#1548)
* Faster Genesis import. (@duburcqa) (#1549)
* Avoid rendering cameras at reset. (@Kashu7100) (#1552)
* Enable taichi debug mode in tests if possible. (@duburcqa) (#1541)
* Re-enable markers by default on RGB offscreen cameras. (@duburcqa) (#1570)

## 0.3.0

This release focuses primarily on stability, covering everything from MJCF/URDF parsing to rendering and physics, backend by a new CI infrastructure running more than 200 unit tests on all supported platforms. The most requested Mujoco features that were previously missing have been implemented. Native support of batching has been extended to all solvers except Stable Fluid, motion planning, and rendering via [gs-madrona](https://github.com/Genesis-Embodied-AI/gs-madrona). Finally, support of soft body dynamics has been enhanced, with the introduction of constraints and coupling between soft and rigid body dynamics.

### New Features

* Add link-wise mask for poss and quats in multilink Inverse Kinematics. (@ziyanx02) (#499)
* Update HoverEnv, update hyperparams, and visualization while training. (@KafuuChikai) (#533)
* Add Mac OS and Windows OS support to the viewer. (@kohya-ss, @duburcqa) (#610, #782)
* Add method to compute classical links acceleration. (@zswang666, @duburcqa) (#451, #1228)
* Support mounted cameras on rigid links. (@wangyian-me, @abhijitmajumdar) (#618, #1323)
* Cameras and main viewer can now track an entity. (@jgleyze) (#611)
* Support fixed entity. (@ziyanx02, @duburcqa) (#673, #1187)
* Support separated rendering for each environment. (@ACMLCZH) (#545, #723)
* Support environment masking to Inverse Kinematics. (@Kashu7100) (#732)
* Support equality constraint and closed-loop robots. (@YilingQiao) (#636)
* Add support on headless rendering on Windows OS. (@duburcqa) (#798)
* Expose public API for Forward Kinematics. (@Kashu7100) (#802)
* Add setter for robot mass. (@Kashu7100, @YilingQiao) (#828, #605)
* Support advanced collision pair filtering. (@duburcqa, @Kashu7100) (#816, #1438, #1499)
* Enable visualizing path. (@bxtbold) (#815)
* Add environment masking to more rigid body methods. (@Kashu7100) (#832)
* Add support of Ellipsoid geometry. (@duburcqa) (#864)
* Add 'show_link_frame' option to the viewer for better debugging. (@Kashu7100) (#871)
* Add environment masking to 'get_links_net_contact_force'. (@Kashu7100) (#880)
* Add support of ball joint type and compound joints. (@YilingQiao, @duburcqa) (#853, #1078, #1080)
* Luisa Render upgrade to support Apple Silicon. (@ACMLCZH) (#886)
* Support equality joint constraint. (@YilingQiao) (#919)
* Support mimic joint in URDF. (@YilingQiao) (#928)
* Support weld constraint. (@YilingQiao) (#948)
* Add method to render pointcloud on cameras. (@wangyian-me) (#897)
* Expose different convex decomposition error thresholds for robots and objects. (@duburcqa) (#1058)
* Force interactive viewer camera Z-axis up. (@duburcqa) (#1060)
* Add helper to convert terrain mesh to height field. (@YilingQiao) (#1033)
* Add maxvolume support for TetGen-based tetrahedralization. (@kosuke1701) (#1088)
* Support parallel simulation for deformable materials. (@wangyian-me) (#1005)
* Add public API to access rigid body mass matrix. (@Kashu7100) (#1132)
* Add public API getter/setter for constraint solver parameters. (@duburcqa) (#1173)
* Add option to exclude self-collision. (@Kashu7100) (#1229)
* Add profiling options. (@hughperkins) (#1247)
* Add implicit FEM with newton and conjugate gradient methods. (@Libero0809) (#1215)
* Implement BVH using Linear BVH. (@Libero0809) (#1241)
* Add USD parsing. (@ACMLCZH) (#1051)
* Expose optional subterrain parameters. (@LeonLiu4) (#1289)
* Add GJK-EPA algorithm for rigid body collision detection. (@SonSang) (#1213, #1357)
* Add linear_corotated elastic material for FEM. (@Libero0809) (#1304)
* Add fast vs high-performance taichi compilation mode. (@hughperkins) (#1330)
* Support environment-wise gravity. (@LeonLiu4, @Milotrince) (#1324, #1498)
* Expose method to compute Jacobian at a specific point. (@LeonLiu4) (#1353)
* Support segmentation map for deformable materials. (@ACMLCZH) (#1363)
* Support parallel path planning. (@Kashu7100) (#1316)
* Support dragging to physical object interactively in viewer. (@gasnica) (#1346, #1378, #1411, #1443)
* Enhance SAP Coupler to support self collision between FEM objects. (@Libero0809) (#1375)
* Support vertex constraints for FEM objects. (@Milotrince) (#1310)
* Add sensor abstraction. (@Milotrince) (#1381)
* Integrate Madrona batch renderer. (@yuhongyi) (#1416)
* Support joint friction. (@YilingQiao) (#1479)

### Bug Fixes

* Improve URDF and MJCF loading. (@zhenjia-xu, @bxtbold, @YilingQiao, @zswang666, @duburcqa) (#517, #675, #735, #744, #765, #777, #792, #872, #913, #936, #940, #988, #1147, #1154, #1159, #1169, #1218, #1235, #1262, #1287, #1501)
* Rework backend & device selection logic. (@lgleim) (#568)
* More robust cross-platform rendering support (viewer and cameras). (@alesof, @eratc, @duburcqa, @YilingQiao, @Kashu7100) (#404, #644, #774, #779, #783, #784, #787, #796, #799, #800, #807, #809, #810, #813, #814, #915, #983, #1069, #1070, #1071, #1073, #1074, #1420, #1421, #1426)
* More robust cachie mechanism for simulation pre-processing. (@duburcqa) (#801)
* Improve debug and high-precision mode. (@duburcqa) (#863)
* Fix constraint solver termination condition. (@duburcqa) (#867)
* Reset collision detection state when setting qpos. (@duburcqa) (#868)
* Use collision geometry as visual for bodies not having any. (@duburcqa) (#870)
* Only add world link/joint if it has at least one geom. (@duburcqa) (#884)
* Avoid segfault because of exceeding number of collisions. (@duburcqa) (#898)
* Fix box-box collision detection. (@duburcqa) (#910)
* Fix fixed joint and body handling. (@duburcqa) (#916, #952)
* Fix link velocity computation. (@duburcqa) (#941)
* Avoid useless convex decomposition. (@duburcqa) (#957)
* Improve numerical stability of MPR collision detection algorithm. (@duburcqa) (#966, #977, #1336)
* Improve multi-point contact stability. (@duburcqa) (#967, #1012, #1117, #1297)
* Fix genesis destroy. (@duburcqa) (#1007)
* Fix issue when adding multiple FEM entities. (@kosuke1701) (#1014)
* Using Vulkan backend for taichi even if no Intel XPU device is available. (@duburcqa) (#1025)
* Try to repair partially "broken" meshes if possible. (@duburcqa) (#1023, #1075, #1077)
* Fix scaling of poly-articulated robots. (@duburcqa) (#1039, #1108)
* Fix mujoco vs genesis discrepancies. (@duburcqa) (#1097)
* More robust robot loading and default options. (@duburcqa) (#1098)
* Improve external force handling. (@duburcqa) (#1292)
* Fix terrain collision detection. (@duburcqa) (#1338)
* Prevent unrealistic angular velocity of Drone entities causing numerical instability. (@duburcqa) (#1405)
* Do not consider markers as physical objects for rendering. (@duburcqa) (#14948)

### Miscellaneous

* Update Drone Entity and Training Performance Enhancements. (@KafuuChikai) (#598)
* Added PID controller util, quadcopter PID controller, flight examples. (@jebbrysacz) (#501)
* Add support of Numpy 2.0. (@johnnynunez, @duburcqa) (#711, #791)
* Setup unit test infrastructure. (@duburcqa) (#876)
* Various minor improvements. (@duburcqa) (#889)
* Get rid of internally maintained trimesh dependency. (@duburcqa) (#918)
* Refactor rigid body data accessors. (@duburcqa) (#924)
* Speed-up convert taichi field to torch. (@duburcqa) (#935)
* Cleanup Genesis init / exit. (@duburcqa) (#989)
* Use torch cuda with vulkan taichi backend if available. (@duburcqa) (#1043)
* Fix naming convention inconsistencies. (@duburcqa) (#1053)
* Add API documentation. (@zswang666) (#1105)
* Use 2*dt as default timeconst for urdf and mesh. (@YilingQiao) (#1115)
* Improve runtime and compile time performance. (@YilingQiao, @duburcqa) (#1164, #1268, #1277)
* More robust decimation processing. Enable decimation by default. (@duburcqa) (#1186)
* Add compile- and run-time performance monitoring. (@duburcqa) (#1209)
* More efficient unit test distribution across workers. (@duburcqa) (#1275)
* Enable GJK collision detection algorithm by default. (@duburcqa) (#1439)
* Enable box-box collision detection by default. (@duburcqa) (#1442)

## 0.2.1

### Bug Fixes
* Fix various visualization and rendering bugs. (@RobRoyce, @VincentCCandela, @Likhithsai2580)
* Resolve some platform-dependent issues. (@abhaybd, @NekoAsakura)
* Fix the issue with loading box textures when parsing MJCF files.
* Correct asset path handling.
* Fix repr output in IPython. (@JohnnyDing)
* Resolve bugs in locomotion examples. (@yang-zj1026)
* Fix several issues with MPR and contact islands during collision detection.

### New Features
* Add a smoke simulator driven by Stable Fluid, along with a demo. (@PingchuanMa)
* Introduce APIs for applying external forces and torques.
* Introduce APIs for setting friction ratios.
* Add a domain randomization example.
* Improve kernel cache loading speed by 20~30%. (@erizmr)
* Provide an interactive drone control and visualization script. (@PieterBecking)
* Introduce an RL environment and examples for the drone environment. (@KafuuChikai)
* Add an option to enable or disable the batch dimension for (links/DOFs) information. Users can choose the trade-off between performance and flexibility.
* Add Docker files. (@Kashu7100)
* Implement the MuJoCo box-box collision detection algorithm for more stable grasping.
* Include the backflip training script and checkpoints. (@ziyanx02)
* Enable support for entity merging.
* Add support for custom inverse kinematics (IK) chains.

### Miscellaneous
* Improve documentation and fix typos. (@sangminkim-99, @sjtuyinjie, @CharlesCNorton, @eltociear, @00make, @marcbone, @00make, @pierridotite, @takeshi8989, @NicholasZiglio, @AmbarishGK)
* Add CONTRIBUTING.md and update CI configurations.
* Introduce multi-language support. (@TitanSage02, @GengYiran, @DocyNoah)

We would also like to acknowledge the ongoing PRs. Some of them have not yet been merged because we have not had enough time to fully test them:
* Unitree G1 walking. (@0nhc)
* Blood, vessels, and heart simulation. (@lhemerly)
* Cross-platform and rendering compatibility. (@VioletBenin, @DearVa, @JonnyDing)
* Docker support. (@skurtyyskirts, @yuxiang-gao, @serg-yalosovetsky)
