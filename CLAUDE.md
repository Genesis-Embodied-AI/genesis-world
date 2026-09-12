# Genesis Development Guidelines

## Miscellaneous

* getattr / hasattr are prohibited; use None initialization and isinstance checks instead.
* Never add attributes on the fly to external / non-owned instances.
* Keep interop data as torch tensors or numpy arrays from the start instead of Quadrants kernels operating on CPU numpy arrays. Numpy implementations must be fully vectorized. Use Numba (single thread, CPU backend) for computation-heavy "kernels".
* Allocate the exact memory size needed. Preallocation based on max size is prohibited.
* Plain dicts for packing attributes are prohibited. Use strongly typed named data structures instead, either dataclasses or simple NamedTuples. The one exception is the parsed info a rigid asset parser produces: what a format states is unstructured and incomplete until the resolution completes it, so it travels as a plain dict, confined to the parser and the resolution that consumes it. No built object holds one.
* Local imports in functions are prohibited, unless strictly necessary to avoid circular dependencies.
* No commented-out prints. Convert them to logging debug traces.
* No code duplication.
* Variable naming follows the Rigid engine conventions. See also the description of PR#1053 (https://github.com/Genesis-Embodied-AI/Genesis/pull/1053).
* Domain objects are only 'entity', 'link', or 'geom'. Never invent a new noun ('body', 'object', 'piece') when one of these fits. This is also a correctness cue: the rigid-body unit is the link, so group geoms per link (iterate entity.links then link.geoms), not per entity.
* Never add a cast - float() / int() / .astype() / np.asarray() / dtype= - unless strictly necessary. numpy scalars index, compare, and do arithmetic fine; cast only where an external interface forces the dtype (igl float64/int64, native Python for kernel args), and never re-cast what a later step already casts.
* Use the helpers defined in genesis/utils/misc.py and genesis/utils/geom.py. 'detach().cpu().numpy()' is prohibited; use 'tensor_to_array'.
* Calling the Quadrants data instance method 'to_numpy()' is forbidden. Use 'qd_to_(torch|numpy)' instead. Take advantage of torch-based zero-copy as much as possible to update the internal state of other solvers. numpy-based zero-copy is only supported on CPU backend, so prefer torch.
* Catch-all exceptions are prohibited, except for exception forwarding in a multi-processing process. Critical errors must never be replaced by warnings: if something breaks the physics, raise an exception, eventually via re-raise to clarify the error message.
* Direct use of numpy / torch / quadrants data types is prohibited, unless strictly necessary due to external interfaces enforcing specific dtypes. Our types are defined in genesis/__init__.py; the quadrants dataclasses dtypes are defined in genesis/utils/array_class.py.
* Defining local-scope functions should be avoided unless very well motivated.
* Attribute-like instance methods (eg has_any_rigid_coupling) should be converted to properties.
* scene.sim.coupler is private API; users cannot be expected to call 'set_link_ipc_collision'. Expose such controls as entity-level material options.
* Halt the simulation on NaN via the existing errno mechanism (define a new error code if necessary); a warning is not acceptable.
* Remove all legacy code. No deprecation / legacy compatibility layers.
* Remove unused helpers (eg has_coupling_type). Do not implement private helpers proactively if they would be easy to add later on.
* Prefer the dev getters/setters API of solvers over directly accessing quadrants fields: fields may be renamed or removed, while getters/setters are stable dev API.
* A quantity derived from several raw solver fields belongs in a single vectorized solver getter, not assembled by the consumer nor extracted via a bespoke per-element `@qd.kernel` that the caller then re-indexes. This keeps field-access coupling and batched / non-batched handling in one tested place.
* Batching based on batched / non-batched should be avoided. Deal with it at argument-level 'envs_idx=env_idx if batched else None'.
* Do not call getters/setters repeatedly per env idx; deal with all envs at once, even if a for-loop fallback is necessary for IPC-related getters/setters. Calling 'qd_to_numpy' in a tight loop is not acceptable: call it once, and when a subset is needed, pass the slice as an input argument of 'qd_to_numpy' instead of slicing afterward.
* Avoid generic 'all_' prefix/suffix for containers if possible but rather be specific, eg 'joints_xanchor', 'links_inertia_i', 'geoms_pos', 'entities_quat', 'verts_idx'.
* Avoid pure variable name assignment, eg 'mass_mat_env = mass_mat_all'.
* Prefer clear over fresh allocation for mutable containers, eg 'self.abd_data_by_link.clear()' instead of 'self.abd_data_by_link = {}'.
* Never use `flatten()`, `reshape((-1,))`, or `ravel()` in place of squashing. Moreover, prefer `[..., 0, :]` over `squash(axis=-2)`.
* Don't use np.asarray unless strictly necessary, ie for public API with no control on the data source. For internal dev or private code path, it should NEVER be the case.

## Priorities

- **Code quality over source consistency.** When refactoring or porting code from PRs, external projects, or prior implementations, Genesis conventions always take priority over the original code's patterns. Use the original logic as reference, not the implementation style.
- **Build/init time: maintainability first** (as long as performance is reasonable). Use public APIs.
- **Runtime hot paths: efficiency first.** Minimize GPU-CPU transfers, use bulk operations.

## Data Access

- **Build/init time:** Iterate via entity/link/geom public API (`entity.links`, `link.geoms`, `geom.init_verts`, `link.get_pos()`, etc.). Only fall back to low-level solver field access when efficiency, conciseness, or maintainability specifically requires it.
- **Runtime:** Use `qd_to_numpy` / `qd_to_torch` (from `genesis.utils.misc`) for bulk GPU-CPU transfer. Never use `.to_numpy()` / `.to_torch()` directly. Never item-access qd fields (`field[i]`).
- **Always pass `transpose=True`** to `qd_to_numpy` / `qd_to_torch` to move the batch dimension to front (`[B, n, dim]`), aligning with public getter conventions.
- **Hoist conversions before loops.** Call `qd_to_numpy` / `qd_to_torch` once, then index the local array.
- **`qd_to_numpy` with `copy=False` fails on CUDA.** GPU data always requires a copy to numpy.
- **A `qd_to_torch` / `qd_to_numpy` view that gets WRITTEN binds to a named local first** (`flags_t = qd_to_torch(..., copy=False)` then `flags_t[mask] = ...`); never mutate through an anonymous chain. Read-only use is the opposite: chain as many operations as needed inline (`qd_to_torch(f).T.reshape(...).gather(...)`) — a named temporary a one-line read does not need is noise. When the chain overflows the line, split it at a natural intermediate into named single-line statements (`knots = qd_to_torch(f).T.reshape(...)` then `knots_best = knots.gather(...)`), never into a parenthesized multi-line chain.
- **A zero-copy accessor passes the selection on untouched.** A setter whose write is a copy (one vector, one flag, one row per environment) goes through the view when `gs.use_zerocopy` holds and returns there, the kernel path being the fallback. On the view path `Scene._sanitize_envs_idx` is prohibited: pass the selection to `indices_to_mask` as it came, without branching on its type, then `broadcast_tensor` against `view[mask].shape` for the value and `assign_indexed_tensor` for the write. The view path still owes the caller two checks: a scalar naming no environment of the scene, and any selection at all where there are no parallel environments. Negative bounds counted from the end are a known gap of `indices_to_mask`, to be closed there and never worked around one accessor at a time.
- **Metal stream sync is the caller's job**: after a batch of torch zero-copy writes, call `torch.mps.synchronize()` once before the next quadrants kernel reads the buffers. Helpers performing such writes (e.g. `qd_zero_grad`) never sync internally.
- **Branch zero-copy paths on `gs.use_zerocopy` alone; no try/except probing.**
- **Do not pass `copy=` at all** to `qd_to_torch` / `qd_to_numpy` when the value returned to the caller is fresh arithmetic (nothing internal aliases out) — let it zero-copy or copy only as needed. Reserve `copy=True` for when a raw field view would otherwise escape and could be mutated.
- **Store Python references at build time** (entity, link objects in dicts) for runtime lookup, rather than re-deriving from solver fields every step.
- **Never pass numpy scalar types** (e.g., `numpy.float64`) to kernel arguments. Always cast to native Python types (`float()`, `int()`). Numpy scalars break quadrants fastcache (weak reference).
- **Never use `np.asarray` on torch tensors.** Use `tensor_to_array` from `genesis.utils.misc` instead, which properly handles GPU→CPU transfer via `tensor_to_cpu` first. Pass its `dtype` argument (`tensor_to_array(x, dtype=...)`) rather than chaining `.astype`, and convert once inside the consuming function, not at each call site.

## Kernels

- **A kernel that runs under `requires_grad` takes no `continue` and no `while`**: quadrants' autodiff rejects both. A gate inside such a kernel (the kinematics and dynamics walks, the constraint solve) is a flag set under the static condition and tested by an `if`, never an early `continue`.
- **An index read from a field is bound to a temporary before it indexes another field** (`i_r = rigid_info.links_root_rank[i_l]` then `rigid_info.roots_link_end[i_r]`). Nested indirect indexing (`a[b[i]]`) is prohibited in kernels and funcs.
- **New code:** Free function `@qd.kernel`, no `@qd.data_oriented`. Use `V_ANNOTATION` from `genesis.utils.array_class` for type-polymorphic parameters.
- **FEM solver:** Follows old `@qd.data_oriented` method pattern. Any kernel added to FEM solver must stay consistent with this.
- **Rigid kernel and func calls are positional for self-named arguments**, relying on the canonical parameter order below; anonymous constants (bare literals such as trailing static flags) are passed by keyword. A func call that passes a struct member alongside its parent struct stays keyword-only, because quadrants' positional func-argument expansion duplicates the member.
- **Canonical parameter order, for every kernel and func:** indices of any kind first (loop indices, index tensors like `envs_idx`, index-valued scalars like `joint_idx` or range starts/ends), then dynamic native Python scalars and fixed-size quadrants vectors (per-call values: positions, quaternions, penetrations), then the kernel/func-specific tensors, then all state structs, then all info structs, then the static configs, then the kernel/func-specific constants-in-practice (shape integers, eps, tolerances - info by nature even when not declared as such), and the kernel/func-specific static compilation flags (`qd.template()` booleans) at the very end - except `errno`, which systematically takes the last position. Within the state/info/config groups the component order is fixed - dyn, rigid, collider (mpr, gjk, support_field, sdf), constraint - with leaf structs at their aggregate's slot in its declared field order. Call sites pass arguments in signature order.
- **Rigid solver** is the reference implementation for kernel and code quality standards.
- **No dtype in quadrants fixed-size vector annotations**: `qd.types.vector(3)`, never `qd.types.vector(3, dtype=gs.qd_float)`.
- **Pure read-write data accessors on the hot path go through zero-copy views, keeping the kernel only as fallback.** A tiny kernel that merely shuffles or rewrites a field (payload remap, flag sweep) pays full kernel dispatch for microseconds of work: when `gs.use_zerocopy` holds, do it in place through a `qd_to_torch(..., copy=False)` / `qd_to_numpy(..., copy=False)` view, and fall back to the kernel otherwise. Reserve dedicated kernels for genuine computation.
- **One inlined instance of a `qd.func` per kernel.** Every `qd.func` a kernel calls is inlined, its static branches pruned, then compiled: calling the same helper from two runtime branches with different arguments compiles it twice, a template argument taking two values compiles the callee twice, and a heavy body under `qd.static(range(R))` compiles R times, each copy adding its full compile time. Pick the arguments in runtime branches and call once, loop items at runtime with a runtime count (recompute a cheap per-item value rather than carry per-item registers), and where an early exit is needed use a single-iteration `for` with `break` so one call follows (the collider's pattern), or an `if` gate around the body of the top-level loop in a differentiable kernel, where `break` and `continue` are rejected. Reserve static unrolls for tiny bodies. Some readability may go, as long as the code stays plain to follow and its complexity does not blow up. What decides is the measured compile time of the CI benchmarks: below 2% it does not matter, at 8% and above it is critical, in between it is a judgment call.

## API Design

- **Don't force users to set options with a single correct value.** If an option has only one valid value in a given context (e.g., `enable_collision=False` is always required with IPCCoupler), set it automatically and raise an error if the user explicitly provides the wrong value. The right pattern is: default to `None`, resolve to the correct value at init time, raise on conflicting explicit value. The resolution logic can be based on other options, scene contents, or even runtime state — as long as it is well documented.
- **No single user helper.** Do not introduce any extra helper if it is used at one single place for one specific context. A new single-use helper is only acceptable if it is standalone and completely generic, like a math function or conversion utility. Avoid private instance methods.
- **A unit-valued return must not be wrapped in a named structure.** Returning a one-field NamedTuple/dataclass where a bare value suffices is dead boilerplate; only introduce a named structure for genuinely multi-field returns.
- **Option fields for index/array inputs must use the project's array-like type aliases** (`IArrayType` / `OptionalIArrayType` / `FArrayType` / `Vec3FType`, etc. from `genesis.typing`), never a bare `tuple[int, ...]` / `list[...]`. `Options` is strict Pydantic (`ConfigDict(strict=True)`), so a bare typed-collection field rejects the lists and numpy arrays users naturally pass (`dofs_idx_local=[0, 1]`); the aliases carry `strict=False` to coerce array-like inputs to a tuple. Match how sibling option fields are typed (e.g. `filter_link_idx: OptionalIArrayType`).

## Style

- **Write plain declaratives: subject, verb, object.** Name the operation with the standard simulation vocabulary (mass matrix, Jacobian, inverse weight, kinematic tree, substep). Never invent a verb for an ordinary operation ("weigh a link", "quote a value at", "hold a rate"), and never write a clause of the form "X is what Y does". A docstring states what the function does, what each argument means and what the caller gets back. A comment states why the code is the way it is.
- **Docstrings describe the current state, not history.** Write what a function does now, not what it used to do or what was added. Do not write "Preserves the original X scenario" or "Added Y for Z". If a behavior is required for a bug fix, explain why in terms of the current invariant (e.g., "Must handle Z case to avoid Y").
- **User-facing option docs state the TRADEOFF, stand alone, and stay behavioral.** An option's doc must let the user decide: give cost AND benefit and when to pick each value, not just what it does (if one setting looks strictly better, the downside is missing - the doc is useless). State the real cost in user terms (if speed is not it, say what is, e.g. numerical robustness). Explain it in Genesis terms only - no external-engine references (e.g. MuJoCo) and no internal/implementation mechanism (constraint "rows", "coupling", "cone projection", kernels, factor paths); those belong in dev code comments. Keep it compact.
- **Comments justify - generically.** A comment's purpose is to explain why the code is the way it is: the invariant it preserves, the failure mode the tempting alternative runs into. Keep the justification generic and present-tense; never anchor it to perishable specifics (measured values, benchmark figures, particular unit tests or examples) - those go stale long before the mechanism does.
- **Each fact lives in exactly one comment; other sites cross-reference it.** Mechanics belong at the data/declaration they describe, motivation and gating rationale at the decision site that owns them. Never restate the same information at several sites - but every dependent site keeps a one-liner with an explicit pointer to the source of truth (e.g. "see nt_H in array_class.py").
- **No semicolon in prose, and no dash as a parenthesis.** Write two sentences, or use parentheses.
- **Name a local after what it is, not after the property under test.** A scene is `scene_from_substeps`, never `shorthand`, `inheriting` or `idle`.
- **A known upstream bug is a `# FIXME:` naming the issue.** Two lines at most, with the tracker reference (`# FIXME: quadrants#887 - ...`) and what the workaround costs.
- **Never document a contract the code does not offer.** No array shape and no memory layout in a docstring unless the API guarantees it.
- **Compare floats with a tolerance.** `==` and `!=` on floating-point values are prohibited, host-side values included: use `np.allclose(a, b, atol=gs.EPS)` or an explicit bound.
- **Never write to a third-party object's private state.** Keep our own record beside the object and expose it through a property.
- **State what IS, never what is NOT.** In comments, docstrings, and reports, drop "not X", "unlike Y", "does not ...", "it is X, not Y" unless the negation was explicitly asked for or is the literal spec. If a property is not mentioned, it does not exist - defending against an unraised concern is noise. Say the positive fact and stop. The one exception is an expectation the reader would naively hold: where the naive reading is the wrong one, saying so is the fact, and the negation earns its place.
- **Function-level description goes in a docstring, never a block of leading `#` comments** at the top of the body. Reserve inline comments for non-obvious implementation details at their point of use (e.g. a constraint-layout invariant a few lines rely on). A getter/method that opens with a paragraph of `#` prose is wrong; that prose is its docstring.
- **Comments go on their own line above the code they annotate, never trailing inline, whenever possible.**
- **No local imports** unless strictly necessary (e.g., circular dependency avoidance). All imports at module top level.
- **Imports are grouped, blank-line separated, alphabetical within each group, in this order:** standard library; ubiquitous third-party (`numpy`, `torch`, `tqdm`, ...); other third-party; our own libraries (`quadrants`, ...); `genesis` itself, absolute then relative. Nothing enforces this, so it is done by hand.
- **Touching one import means fixing the whole module's imports**: any addition, removal or replacement obliges regrouping and re-sorting every import in that file, and deleting the ones no longer used. Otherwise a hand-maintained order decays one edit at a time.
- No import aliasing (`Cloth` not `ClothMaterial`).
- No temporary variables for `isinstance` checks. Use `isinstance(...)` directly in `if` statements.
- **Pass an anonymous literal by keyword** so its meaning shows at the call site (`_adaptive_params(verts, faces, aggressiveness=7)`, not a bare `7`). Positional is fine for self-named variables. This holds in rigid kernel and func calls too: anonymous constants (and everything after them) go by keyword; only self-named arguments are positional. Exception: numeric constants in module-local qd.func calls stay positional, keeping those internal call chains compact.
- **Line wrapping is all-or-nothing, per nesting level.** A call that fits on one line at its indent (<= 120 chars - count before wrapping) MUST be written on one line. When it overflows, all the arguments move together onto a single hugged continuation line; when that overflows too, every argument goes on its own line. Inside an exploded outer call, a nested call that fits stays on ONE line (`shape=maybe_shape((_B, solver.n_dofs_), solver._rigid_config.enable_cone_free_hessian_reuse)` inside an exploded `V(...)`). ruff never collapses an exploded call (magic trailing comma), so write the compact forms by hand. Nested data literals (matrices, coordinate/edge tables) are the exception: keep one row per line even when the whole literal fits on one.
- **One `if` clause holds either static conditions or dynamic ones, never both.** A dynamic operand makes the whole clause dynamic, so a static gate chained with a runtime test compiles both branches: nest the static `if` outside, or carry the runtime test in a flag set under it. Several static conditions of one clause are packed in a single `qd.static(...)`.
- A block's shared-memory array (`qd.simt.block.SharedArray`) takes the `sh_` prefix.
- **Booleans read as predicates** - every boolean variable/field/attribute takes an `is_` or `has_` prefix, present tense preferred (`is_fixed`, `is_convex`, `has_multi_island_structure`). `was_` only for a genuine past-tense flag (prefer `is_cached_loaded` over `was_cached`). A bare name (`hibernated`, `placed`) or a `did_` prefix (`did_fuse`) is not valid. The prefix is asked of state, not of input arguments: a boolean parameter of a function, method or kernel names what the caller asks for (`refresh_position`, `scale_inertia`, `in_place`) and takes no prefix.
- **A variable name starts with a noun** naming the kind of object it holds, so what it refers to reads without the surrounding code (`rejected_kinds`, `links_offset_pos`), never a bare adjective or participle (`rejected`, `left_out`, `named`). The exceptions are booleans, which take the `is_`/`has_` prefix above, and indices and loop variables, which follow the `i_<type>` convention.
- **Never detect state changes by comparing state snapshots for equality.** Genesis has dedicated change-detection facilities (solver StateChange subscriptions and friends); caching runtime data to poll it against a fresh copy is prohibited. A missing facility or missing granularity is no excuse: extend the mechanism, never fall back to snapshot comparison. Sufficient grounds on its own for rejecting the entire PR.
- No `@qd.data_oriented` on non-solver classes (materials, couplers, entities).
- Naming consistency within directories (e.g., `ipc_*.py` for all IPC examples).
- **ASCII only.** No non-ASCII characters anywhere in source (code, comments, docstrings): write `tau`/`qddot` not `τ`/`q̈`, `*` not `·`, `-`/`--` not `—`, plain `# ---` not box-drawing separators.
- **Spell out acronyms on first use** in each docstring/comment, verbatim with the acronym in parentheses, e.g. "positive semi-definite (PSD)", "second-order cone (SOC)". After that first spell-out the acronym alone is fine within the same docstring/comment.
- **No unverifiable / stale external references.** Do not justify a design by citing external projects a reader cannot verify. When porting a known feature (e.g. MuJoCo's elliptic cone), reference it at the right granularity - name the feature, optionally the official doc section, and/or the math formula - but never the external repo's specific file / function / constant names (they are cross-repo and go stale). "matches MuJoCo's elliptic cone" good; "matches MuJoCo's `ellipticCostDif`" bad.
- **No torch dtype casts** (`.long()`, `.to(torch.int64)`, etc.). For advanced indexing use `indices_to_mask` (from `genesis.utils.misc`); for a `torch.gather` index, let `cumsum` over a boolean mask yield int64 instead of casting.
- **No useless or circular comments.** Do not restate what names/types/control-flow already convey, and never write tautologies ("`dt` is the timestep, so it must match the simulated timestep").
- **Audit any comment or docstring paragraph beyond a couple of lines: every sentence must be essential.** Comments keep only what cannot be reasonably deduced from reading the associated code in isolation. Docstrings are higher-level - the reader never opens the body - so they state everything needed to use the function correctly and determine precisely whether it is relevant. Fill lines to the 120-char width so the surviving facts take the fewest lines.
- **A paragraph of more than two lines never ends with a line holding a single word.** Reflow or tighten the last sentence so the closing line carries at least two words.
- **Match the established sibling pattern; do not invent structure.** Before adding any helper, fixture, option, cast, comment, or file layout, confirm a sibling in the same directory already does it that way and copy it (e.g. `imu.py` / `imu_franka.py` / `tests/sensors/test_imu.py` / `tests/rigid/`).

## Testing Guidelines

- **Never filter or truncate test output inline** (no `pytest ... | tail` / `| grep`). Redirect the full output to a log file, then extract from the file: a filter between pytest and disk destroys the only copy of the failure names and masks the exit code.
- **Never remove or weaken an existing assertion or measurement to silence a failure** (local or CI). Report the failure with the data and ask; a threshold that only holds on one machine is a calibration problem.
- **Bug fix PRs** must include a regression test that fails on `main` and passes with the fix, added to the test already covering the capability that broke.
- **Never write a test that pins a known defect.** Tests assert observable physics and public API behavior, never implementation details.
- **Never test a limitation, but do test that an unsupported input is handled.** A capability Genesis does not offer gets no test. Refusing an input it cannot handle is behavior, so a test asserts the refusal wherever proceeding would corrupt what the user gets.
- **No deprecation tests.** Do not add unit tests that verify deprecation warnings are emitted.
- Feature tests exercise the new behavior, not the internal warning machinery.
- **Unit tests must NOT have docstrings.** A good test name spares writing the short docstring. A code comment is acceptable only when strongly motivated, i.e. it explains something the test body cannot convey. Never state regressions or history.
- **Tests are organized per component, then per capability.** Each component has a folder under `tests/` (`rigid/`, `deformable/`, `particles/`, `ipc/`, `coupling/`, `sensors/`, `rendering/`, `parsers/`, `core/`, `integration/`, `benchmarks/`) holding one file per capability (e.g. `tests/rigid/test_collision.py`, `tests/sensors/test_imu.py`). Add new tests to the existing capability file; a new file is only warranted for a genuinely new capability, never a fresh `test_<one_thing>.py` for a single feature. Complements "Pack tests" below.
- **Strengthen an existing test rather than write a new one.** A test validates one capability of the public API, never a code block or an internal logic path. New coverage for a capability goes into the test already covering it, by extending its scene and its assertions or by adding a `@pytest.mark.parametrize` dimension; never a sibling function.
- **Pack tests: one scene build per test.** Avoiding an extra build is mandatory, even at the cost of a test that is harder to write and to read. What this spares is the build itself, which is slow and compute-heavy, so a scene written to be rejected costs nothing and does not count against the budget: several of them in one test are fine. In order of preference: one comprehensive scene with diverse entities and options over many small single-option tests; differently-configured entities in the same scene; setups that must not interact placed far enough apart within that one scene; configurations swept through `n_envs`; a reference simulation folded in beside what it is compared against.
- **Name tests after the feature they validate, never after the scene being simulated.** A test exercising off-axis contact rejection through a stacking-tower scene is `test_reject_offaxis_contact_on_authored_decomp`, not `test_stacking_tower_stability`: the scene is how, the feature is what. Names never repeat their module or folder name as a prefix (`tests/sensors/test_temperature.py::test_grid_sensor_contact_and_reset`, never `test_temperature_grid_...`).
- **Assert physics, not just execution.** "Simulation runs without error" is not a test. Check quantities with physical meaning: free-fall displacement (`z = z0 - 0.5*g*t^2`), no ground penetration (`min_z > -d_hat`), velocity → 0 at rest, contact stops fall.
- **Non-regression fallback.** If no analytical expectation is available, run the simulation once to get reference values, hardcode them, and assert with a loose tolerance. Add a `FIXME` comment asking to replace with physics-informed assertions later.
- **Use `assert_allclose` / `assert_equal` from `tests/utils/assertions.py`.** Prefer `assert_equal` for exact comparisons over framework-specific forms (`torch.equal`, `np.array_equal`).
- **`tol=` sets both `atol` and `rtol`.** Pass `atol=` for quantities whose scale is fixed and far from 1 (pixel channels in `0..255`); reserve `tol=` for quantities of order 1.
- **Break the implementation on purpose and watch the assertion fail** before committing a test written to catch a specific defect.
- **Assert observables, never internals.** Validate rendered arrays (image, segmentation map, frame differences), poses, velocities, forces; never private fields, draw orders, call counts or rebuild counts. Derive the expected value analytically and design the scene so a broken mechanism moves it: two overlapping same-size semi-transparent surfaces along the view axis compose as `near + (1 - alpha) * (far - background)`.
- **When a swept comparison has cases that legitimately differ, group them by the property responsible; never list their positions in the sweep** (`msaa_mask = [0, 1, 2, 4, 5, 6]`). Bound how far the groups may differ instead of skipping the comparison between them.
- **A scene that needs contacts needs a movable body.** Both-fixed geom pairs are dropped at build time, so sinking a fixed entity below a surface yields no contact and permanently buries its geometry in the render.
- **Every assertion is a specification.** Assert only behavior we genuinely want to commit to, and drop any assert already implied by a stricter one nearby.
- **FEM entity positions:** `entity.get_state().pos` has shape `[B, n_verts, 3]`. Use `[..., 2]` to select z across all envs and vertices.
- **Rigid entity positions:** `entity.get_pos()` returns `[B, 3]` or `[3]`. Use `np.atleast_1d(...)[..., 2]` and `.all()` for multi-env checks.
- **Parametrize batched tests over `n_envs=[0, 2]`.** Multi-env is where shape bugs hide; never test single-env only. Do not add a dead parametrize dimension that a conftest fixture already controls (e.g. `backend`).
- **Step budget (hard).** Use the strict minimum steps needed, measured not guessed; if the checked quantity is FP-sensitive, pad by whichever is larger: +10% or round up to 50. Per-test horizon tiers: `<100` fine, `100-300` grey (needs justification), `>300` near-prohibited (only 4-5 such tests in the whole suite). Also bound total env-steps (`steps * n_envs`) - a 500-step x 16-env settle (8000) is unacceptable; delete it or fit the budget. CI cost multiplies every step across the matrix.
- **Validate the math, not scale.** One constraint proves the theory (works for 1 => works for 1k); no large/high-DOF scene is needed for correctness. A real multi-body scene earns its steps only when it validates something a single constraint cannot (end-to-end stability, e.g. the bowl tower).
- **Prefer FP-robust checks.** A single constraint-solve comparison (one step from a fixed state) is robust; a multi-step trajectory comparison across numerically-distinct codepaths (different solver arms/factor paths) is not - fp32 compounds and correct paths diverge. Compare a single solve, or an analytical physical property. Cross-engine (MuJoCo) consistency = the real-world check; analytical closed-form = the math check. Regression tests are reactive (add when a bug surfaces), not speculative.
- **Gradient-vs-FD tolerances are pinned to measured floors**: floor T = max|ana - fd| / (1 + |fd|) at the configured eps, worst across cpu and BOTH GPU archs; tolerance = 1.5x-5x the floor, values only {1, 2, 5}e-X; do-not-chase floors 1e-10 (fp64) / 5e-5 (fp32); eps per precision (large for fp32, small for fp64). A floor of exactly 0 = vacuous check - fix the loss, not the tolerance. An eps-independent residual is an analytical bug, never an FD artifact.
- **Tighten toward physically-exact conditions; slack hides bugs.** A padded safety factor or generous tolerance that makes a scenario pass trivially also masks real behavioral differences (across backends/arms/factor paths). Prefer the tightest thresholds the physics justifies. When tightening a test makes it start failing, investigate the root, don't back off. And verify a fix against the EXACT assertion path (full horizon, all phases), never a cheaper proxy - the part a proxy drops is where behavior diverges.

### Scene construction in tests and examples

- **Keyword arguments follow the declaration order of the callee**, e.g. `add_entity(morph=..., material=..., surface=...)`; keywords never license reordering.
- **One option per line** for every `scene.add_entity` / `scene.add_sensor` / `gs.morphs.*` / `gs.options.*` call, even with a single option. ruff does not enforce this; it must be done by hand. It is asked of the scenes a test simulates: a scene written to be rejected as it is built is read for the one option that makes it invalid, so its options stay on one line, morph included, and no comment is owed for that.
- **Every test calling `scene.step()` takes the `show_viewer` fixture and sets `ViewerOptions` with a camera viewpoint tuned to frame the simulated scene**, so running it with the viewer enabled is immediately usable for debugging.
- **Only strictly-required options, at default values unless a reason forces otherwise.** Never set redundant or default-valued options (`align=False`, `ProfilingOptions(show_FPS=False)`), and never pick a non-default value without a load-bearing reason (e.g. use a smaller `dt` than the default only if precision or stability actually needs it). The one exception: an option whose value the test/example computation reads (`gravity`, `dt`) must be set explicitly even when it equals the default, so the simulated value matches the asserted one.
- **Pass scalars / lists directly.** Setters and IK accept array-like and broadcast a scalar across DOFs; never wrap in `np.full`, `np.array`, or a single-element `[...]` (`control_dofs_force(TAU)`, `set_dofs_kp([...])`, `inverse_kinematics(pos=[...], quat=[...])`). Inline one-shot target vectors instead of naming them.
- **Custom MJCF/URDF models:** build them with `xml.etree.ElementTree` in a fixture that returns `ET.tostring(mjcf, encoding="unicode")`, and pass that straight to the morph `file=` — `FileMorph.file` accepts inline XML content (string) for both MJCF and URDF morphs, so never write a temporary XML file just to load a model. Follow `tests/rigid/conftest.py`. Never assemble XML via `write_text` / `textwrap.dedent` string literals or f-string concatenation.
- **Temporary assets that must be stored on disk use the session-scoped `asset_tmp_path` fixture** (shared across tests); `tmp_path` is per-test.
- **No module-level test constants, helpers, or parametrize scenario-lists.** Put constants inside the test, build the scene inline, pack diverse configs into one scene (follow `tests/sensors/test_imu.py::test_sensor`). Read derived params back from the built model (`get_dofs_armature`, `get_dofs_damping`, ...) rather than duplicating literals between fixture and assertions.
- **Hide recurring boilerplate in fixtures; never add indirection without a payoff.** The test body must show only what is being validated; recurring model construction and asset resolution go into fixtures (kills duplication, eases parametrization). Scene setup stays in the test body - a unit test reads like an example script - even at the cost of duplication. A fixture that improves neither readability nor factorization is strictly harmful: indirection is harmful by default and must be justified by real benefits - otherwise stay explicit and self-contained. No premature factorization.
- **Do not gate per-step asserts** with `float(...)` / `.item()` casts on tensor reductions. Design the scenario (e.g. a warmup loop that spins joints up) so the assertion holds unconditionally, then assert on the full tensor with `.all()`.
- **Exact analytical dynamics checks:** force `gs.integrator.Euler` so the finite-differenced `qacc` equals the solver's, and account for both rigid-body rotational inertia and the implicit-damping first-order correction (`effective_inertia = I + damping*dt`, as in `test_position_control`) rather than loosening tolerances or distorting geometry.

## Running Tests From a Git Worktree

The package is installed in editable mode pointing at the PRIMARY checkout, so a worktree's `genesis/` is NOT what
gets imported by default. This silently tests the wrong engine: the worktree's `tests/` run against the primary
checkout's `genesis/`, so engine edits appear to have no effect and every result is void.

- **Always pass `PYTHONPATH=$PWD` when invoking pytest from a worktree**, e.g.
  `cd <worktree> && PYTHONPATH=$PWD pytest -n 8 tests/rigid`. Same for the cluster: `PYTHONPATH=$WORKTREE pytest ...`.
- **Verify it once per session, do not assume.** `PYTHONPATH=$PWD pytest -n 0 -s -k <any test>` with a
  `print(genesis.__file__)`, or simply check that the path printed starts with the worktree.
- Plain `python script.py` from a worktree DOES pick up the worktree (cwd precedes the editable path), so a script and
  a pytest run in the same directory can exercise DIFFERENT engines. Never compare their results.
- Symptoms that mean this is happening: an engine edit changes nothing; a kernel `print()` added to the worktree emits
  nothing; an A/B over engine variants returns bit-identical numbers for every arm. Treat all three as this trap
  first, before any physics explanation.
- **Give the worktree its own compile cache when running in parallel**, e.g.
  `QD_OFFLINE_CACHE_FILE_PATH=<scratch>/quadrants GS_CACHE_FILE_PATH=<scratch>/genesis`. A worktree engine differs
  from whatever populated the shared `~/.cache`, so every kernel is new and the xdist workers race to write the same
  cache, which corrupts entries: hundreds of `QuadrantsNameError` on names that are plainly in scope, spread over
  tests that pass one at a time, and getting WORSE on a second run as the bad entries persist. Redirect the cache
  rather than deleting the shared one.

## Testing on Cluster

- SSH: `ssh genesis-coreweave`
- Path: `/mnt/home/duburcqa/workspace/src/genesis`
- **Git operations (fetch, checkout, pull) must be done on the login node**, not inside `gs-srun`. The compute node container cannot reach GitHub.
- **Always use `gs-srun`** (Slurm wrapper) to allocate a GPU node. Never run pytest or python directly on the login node.
- **Never source a venv manually.** The container image already has the correct environment. The `bash -lc` login shell sets everything up automatically.
- Pattern for git + run: `ssh genesis-coreweave 'bash -lc "cd /mnt/home/duburcqa/workspace/src/genesis && git pull && gs-srun --partition=rtx-high --nodes=1 --gpus=1 bash -ilc \"cd /mnt/home/duburcqa/workspace/src/genesis && pytest -n 10 tests/ipc -v --no-header 2>&1\""'`
- Example tests need `-m examples` to override the default marker filter in `pyproject.toml`.
- Copy local files to cluster: `scp <local_path> genesis-coreweave:<remote_path>`. Then run with `gs-srun`.
- **The compute container mounts `/mnt/home`, not the login node's `/tmp`.** A script, patch, or output dir placed under `/tmp` on the login node is invisible under `gs-srun`; stage everything under `/mnt/home/duburcqa`.
- **The interactive login shell (`bash -ilc`) sets `noclobber`:** `cmd > file` aborts with "cannot overwrite existing file" when `file` already exists, silently producing no output for that run. `rm -f file` first (or redirect with `>| file`) whenever re-generating a log at a fixed path.
- **Run a staged script with `bash -ilc "source script.sh"`, never `bash -l script.sh`.** The container's Python env only initializes in an *interactive* login shell; a non-interactive `bash -l script.sh` under `gs-srun` runs without the venv (`ModuleNotFoundError: genesis`). Stage the script under `/mnt/home` and source it, e.g. `gs-srun ... bash -ilc "source /mnt/home/duburcqa/run.sh"` — this also sidesteps nested-quoting breakage from long inline commands.

## Reproducing Apple Software Renderer Failures Locally

All GitHub Apple Silicon macOS runners are VMs whose virtualized GPU has no OpenGL, so rendering falls back to the Apple Software Renderer. That renderer is half-broken; macOS-only rendering failures come from tests tripping one of its failure modes, so be careful about which rendering features unit tests rely on. To force it locally on any Mac, hijack the two pixel-format attributes that pyglet unconditionally appends, before any GL context is created:

```python
# force_sw.py - import before any GL context is created, e.g. 'pytest -p force_sw' with PYTHONPATH set
from pyglet.libs.darwin import cocoapy

cocoapy.NSOpenGLPFAAllRenderers = 70  # NSOpenGLPFARendererID
cocoapy.NSOpenGLPFAMaximumPolicy = 0x00020400  # kCGLRendererGenericFloatID
```

- This reaches pyglet-created contexts only, so offscreen rendering must be routed back through the pyglet platform with `PYOPENGL_PLATFORM=pyglet`; its own default on macOS is CGL, which pyglet knows nothing about.
- Run with the same flags as macOS CI: `PYTHONPATH=<dir-with-force_sw.py> PYOPENGL_PLATFORM=pyglet GS_TORCH_FORCE_CPU_DEVICE=1 pytest -p force_sw --dev --logical --backend cpu --forked <tests>` (CI additionally selects `-m 'required and not slow'`).
- Verify it took effect: Genesis logs "Software rendering context detected" and `scene.visualizer.is_software` is True.
- Known failure mode: any geometry with vertices outside the camera frustum is misrasterized, breaking pixel comparisons. In practice this bites through ground planes, whose default `plane_size` is effectively infinite (1 km x 1 km): give them a finite size and a position that puts every vertex inside the view.
- Known failure mode: shadow mapping is forcibly disabled on software rendering backends for performance reasons, so snapshot scenes must disable shadows explicitly (`shadow=False` for the rasterizer); a snapshot generated on hardware GL with shadows enabled can never match.

## Tooling & Contributing

- Lint/format: ruff (check + format, line length 120) via pre-commit; install hooks with `pre-commit install` - they run on every commit.
- PR titles carry a bracket tag: `[BUG FIX]`, `[FEATURE]`, `[MISC]`, `[CHANGING]` (the simulated physics change by default: a different model, or different default parameters; solver-internal restructuring is `[MISC]`), `[BREAKING]` (API break). Commit titles are plain single-line sentences without the tag. Both PR and commit titles end with a period.
- PR titles state the benefit for end users, not the implementation. Implementation details go in the PR description.
- Contributors must follow `CODING_GUIDELINES.md` and the reference docs in `.github/contributing/`: ARCHITECTURE, TESTING, CODING_CONVENTIONS, EXAMPLES, PULL_REQUESTS, USD_PARSER. On conflict, ask.
