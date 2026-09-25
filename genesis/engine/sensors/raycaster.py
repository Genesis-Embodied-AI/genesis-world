import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.engine.bvh import AABB, LBVH
from genesis.engine.solvers.base_solver import StateChange, Subscriber
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver, kernel_update_all_verts
from genesis.options.sensors import Raycaster as RaycasterOptions
from genesis.options.sensors import RaycastPattern
from genesis.utils.geom import normalize, transform_by_quat, transform_by_trans_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_numpy, qd_to_torch
from genesis.utils.raycast_qd import (
    kernel_cast_rays,
    kernel_cast_rays_visual,
    kernel_remap_leaf_faces,
    kernel_update_grouped_aabbs,
    kernel_update_grouped_subset_aabbs,
    kernel_update_grouped_visual_aabbs,
    kernel_update_subset_aabbs,
    kernel_update_verts_and_aabbs,
    kernel_update_verts_and_subset_aabbs,
    kernel_update_visual_aabbs,
)
from genesis.vis.rasterizer_context import RasterizerContext

from .base_sensor import (
    KinematicSensorMetadataMixin,
    KinematicSensorMixin,
    SharedSensorContext,
    SimpleSensorMetadata,
    SimpleSensor,
)

if TYPE_CHECKING:
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


@dataclass
class BVHContext:
    """A solver's raycast BVH and the bookkeeping for rebuilding and casting it."""

    solver: "KinematicSolver"
    # None for a static entry until the first RaycastContext.update sizes them to the detected tree count (one tree
    # per distinct per-env geometry); allocated per env up front for movable entries.
    bvh: LBVH | None = None
    aabb: AABB | None = None
    # None for a collision BVH (faces_info / verts_info, no per-face mask), else an int8 (n_vfaces,) tensor selecting
    # which visual faces contribute.
    raycast_mask: torch.Tensor | None = None

    # True when the physics cannot move any of the geometry this BVH covers (every covered collision face sits on a
    # fixed link; for a visual BVH, all links in the solver are fixed), so that geometry only ever changes through an
    # explicit set_pos/set_quat (collision) or set_vverts (visual) - all GEOMETRY mutations the subscription catches.
    # Such an entry skips the per-step rebuild - the dominant cost for static raycasting - and rebuilds only when
    # flagged.
    maybe_static: bool = False
    # Lazy GEOMETRY subscriber for a static entry, registered on its solver; None for a movable entry (which rebuilds
    # every step regardless). RaycastContext.update polls it: a pending set_pos/set_quat/set_vverts flags for rebuild.
    # Static collision entries filter it on the fixed links owning their faces, so mutations that cannot move those
    # links (movable-link teleports, configuration-space setters) never flag a rebuild.
    rebuild_subscriber: Subscriber | None = None
    # Set whenever this entry must rebuild before the next cast: at init, on reset, and when its rebuild_subscriber
    # reveals a set_pos/set_quat/set_vverts since the last build. Ignored by non-static entries, which rebuild every
    # step regardless.
    needs_rebuild: bool = True
    # True when every build input of this BVH is environment-independent: all covered faces read unbatched fixed
    # vertices (their links are fixed with batch_fixed_verts=False, so even an explicit env-specific pose is
    # rejected) and no per-env geom range applies (heterogeneous entities). The trees are then env-identical by
    # construction, so update() groups every env onto one tree without reading the geometry signature.
    is_env_uniform: bool = False
    # (B,) BVH tree slot each env casts against: identity for per-env trees, all-zero for one shared tree, group ids
    # for a grouped static BVH (see RaycastContext.update).
    env_bvh_idx: torch.Tensor | None = None
    # Static BVH allocations keyed by tree count: quadrants fields live for the whole process and template-typed
    # kernels specialize per field instance, so replacing the allocation on every regroup would leak GPU memory and
    # recompile the build/cast kernels each time the tree count changes (e.g. per-episode randomize + reset).
    bvh_by_n_trees: dict[int, tuple[AABB, LBVH]] = field(default_factory=dict)
    # Global face index at each AABB slot, for a collision BVH over a compacted face subset (see
    # RaycastContext.activate for the static/dynamic face split); update() rewrites the sorted leaf payloads with it
    # after every build, so traversals read global faces directly. None when the BVH covers every face in order and
    # for visual entries.
    faces_idx: torch.Tensor | None = None


class RaycastContext(SharedSensorContext):
    """
    Per-simulator collision/visual raycast BVHs, shared across sensor types that cast rays.

    Holds one ``BVHContext`` per (active solver, mesh type): one or two collision BVHs over a rigid solver's faces
    (see ``activate`` for the static/dynamic face split) and a visual BVH over the vfaces opted into
    ``material.use_visual_raycasting``.
    """

    def __init__(self, sim):
        super().__init__(sim)
        self._bvh_contexts: list[BVHContext] = []

    @property
    def bvh_contexts(self) -> list[BVHContext]:
        """The per-(solver, mesh-type) BVHs.

        Raises if inactive: only a consumer that activated it may read them.
        """
        if not self._active:
            raise gs.GenesisException("RaycastContext queried before activation; no sensor declared a raycast need.")
        return self._bvh_contexts

    @property
    def collision_bvh_contexts(self) -> list[BVHContext]:
        """The rigid solver's collision BVH entries: one or two trees jointly covering every collision face, with
        global-face leaf payloads (see activate). Empty if no rigid solver is active.
        """
        return [entry for entry in self.bvh_contexts if entry.raycast_mask is None]

    @staticmethod
    def _group_envs_by_parts(B: int, parts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Group envs by exact equality of the (B, ...) signature parts.

        Returns ``(env_bvh_idx, batch_repr_env)``: the (B,) tree slot each env casts against and the (n_trees,)
        representative env whose geometry builds each slot (the lowest env index of its group). Grouping on exact
        equality of every build input makes a wrong merge impossible: identical envs collapse to one tree, N
        heterogeneous variants to N, per-env divergence (e.g. set_pos on a fixed link) to one tree per distinct
        geometry.
        """
        # An empty part is trivially env-shared (e.g. a placeholder buffer fully masked out) and torch.unique
        # rejects zero-width rows, so drop such parts up front.
        parts = [part for part in parts if part[:1].numel() > 0]
        # Identical-envs short-circuit: a zero-copy row comparison, skipping the sort-based grouping of the general
        # case. This is the dominant layout (env-shared terrain) and runs on every regroup, e.g. at episode resets.
        if B == 1 or all(bool((part == part[:1]).all()) for part in parts):
            return (
                torch.zeros(B, dtype=gs.tc_int, device=gs.device),
                torch.zeros(1, dtype=gs.tc_int, device=gs.device),
            )
        # Exact per-part grouping, folded pairwise: equivalent to a unique over the concatenated signature while
        # keeping the parts' dtypes apart (a mixed-dtype concatenation would promote integer parts to float, which
        # is lossy for large indices).
        env_groups = counts = None
        for part in parts:
            _, inv, part_counts = torch.unique(part, dim=0, return_inverse=True, return_counts=True)
            if env_groups is None:
                env_groups, counts = inv, part_counts
            else:
                _, env_groups, counts = torch.unique(env_groups * B + inv, return_inverse=True, return_counts=True)
        # Lowest env index of each group as its representative (the stable sort keeps envs ascending within a group).
        order = torch.argsort(env_groups, stable=True)
        batch_repr_env = order[torch.cumsum(counts, dim=0) - counts]
        return env_groups.to(gs.tc_int), batch_repr_env.to(gs.tc_int)

    @classmethod
    def _static_entry_groups(cls, entry: BVHContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Group envs by the entry's static geometry; see _group_envs_by_parts for the returns.

        Two envs share a tree iff every per-env input of the entry's AABB build matches. For a collision entry: the
        active geom ranges (present only when link info is batched) and the free-vert positions, both restricted to
        the entry's face subset - fixed verts are env-shared, and the restriction keeps a static subset shared
        across envs while the solver's movable verts diverge under physics. For a visual entry: the vgeom poses
        (feeding the forward-kinematics vvert positions) and the custom vvert buffer, restricted to the vgeoms whose
        vfaces its raycast_mask opts into raycasting. An env-uniform entry is env-identical by construction, so its
        signature is left unread. Reads the entry's build state, so callers must refresh it first (verts for
        collision, vgeom poses for visual).
        """
        solver = entry.solver
        if entry.is_env_uniform:
            return cls._group_envs_by_parts(solver._B, [])
        parts = []
        if entry.raycast_mask is None:
            links_mask = free_verts_mask = None
            if entry.faces_idx is not None:
                subset_geoms_idx = qd_to_torch(solver.dyn_info.faces.geom_idx, entry.faces_idx)
                links_mask = torch.zeros(solver.n_links, dtype=torch.bool, device=gs.device)
                links_mask[qd_to_torch(solver.dyn_info.geoms.link_idx, subset_geoms_idx)] = True
                subset_verts_idx = torch.unique(qd_to_torch(solver.dyn_info.faces.verts_idx, entry.faces_idx))
                subset_verts_idx = subset_verts_idx[qd_to_torch(solver.dyn_info.verts.is_fixed, subset_verts_idx) == 0]
                free_verts_mask = torch.zeros(solver.n_free_verts, dtype=torch.bool, device=gs.device)
                free_verts_mask[qd_to_torch(solver.dyn_info.verts.verts_state_idx, subset_verts_idx)] = True
            if solver._options.batch_links_info:
                # (B, n_links) active-geom range per env
                geom_start = qd_to_torch(solver.dyn_info.links.geom_start, None, links_mask, transpose=True)
                geom_end = qd_to_torch(solver.dyn_info.links.geom_end, None, links_mask, transpose=True)
                parts += [geom_start, geom_end]
            if solver.n_free_verts > 0:
                # (B, n_free_verts, 3) per-env vertex positions
                parts.append(qd_to_torch(solver.dyn_state.free_verts.pos, None, free_verts_mask, transpose=True))
        else:
            vgeoms_mask = torch.zeros(solver.n_vgeoms, dtype=torch.bool, device=gs.device)
            vgeoms_mask[qd_to_torch(solver.dyn_info.vfaces.vgeom_idx, entry.raycast_mask != 0)] = True
            # An all-opted-in mask would force advanced indexing to reproduce the whole field, so drop the selection.
            vgeoms_col_mask = None if vgeoms_mask.all() else vgeoms_mask
            # (B, n_vgeoms, 3/4) per-env visual geom poses
            vgeoms_pos = qd_to_torch(solver.dyn_state.vgeoms.pos, None, vgeoms_col_mask, transpose=True)
            vgeoms_quat = qd_to_torch(solver.dyn_state.vgeoms.quat, None, vgeoms_col_mask, transpose=True)
            parts += [vgeoms_pos, vgeoms_quat]
            if solver._options.batch_links_info:
                # (B, n_links) active-vgeom range per env: variants of one link hold identical vgeom poses, so the
                # range alone separates them, and the fit keeps only that of the env backing a tree.
                links_mask = torch.zeros(solver.n_links, dtype=torch.bool, device=gs.device)
                links_mask[qd_to_torch(solver.dyn_info.vgeoms.link_idx, vgeoms_mask)] = True
                vgeom_start = qd_to_torch(solver.dyn_info.links.vgeom_start, None, links_mask, transpose=True)
                vgeom_end = qd_to_torch(solver.dyn_info.links.vgeom_end, None, links_mask, transpose=True)
                parts += [vgeom_start, vgeom_end]
            if solver.dyn_state.vverts.pos.shape[0] > 0:
                # (B, n_vvert_slots, 3) per-env custom vvert positions (set_vverts overrides), restricted to the
                # opted-in vgeoms' slots. The slot selection is derived from vverts_state_idx itself, so that one
                # field is read whole.
                vverts_state_idx = qd_to_torch(solver.dyn_info.vverts.vverts_state_idx)
                slots_mask = (vverts_state_idx >= 0) & vgeoms_mask[qd_to_torch(solver.dyn_info.vverts.vgeom_idx)]
                vverts_mask = torch.zeros(solver.dyn_state.vverts.pos.shape[0], dtype=torch.bool, device=gs.device)
                vverts_mask[vverts_state_idx[slots_mask]] = True
                vverts_col_mask = None if vverts_mask.all() else vverts_mask
                parts.append(qd_to_torch(solver.dyn_state.vverts.pos, None, vverts_col_mask, transpose=True))
        return cls._group_envs_by_parts(solver._B, parts)

    @staticmethod
    def _sized_grouped_trees(entry: BVHContext, n_trees: int, n_aabbs: int):
        """Point a grouped static entry at its (n_trees, n_aabbs) AABB + LBVH pair, allocating and pooling it on
        first use (see bvh_by_n_trees for why allocations are pooled)."""
        if entry.aabb is not None and entry.aabb.n_batches == n_trees:
            return
        trees = entry.bvh_by_n_trees.get(n_trees)
        if trees is None:
            aabb = AABB(n_batches=n_trees, n_aabbs=n_aabbs)
            trees = (aabb, LBVH(aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64))
            entry.bvh_by_n_trees[n_trees] = trees
        entry.aabb, entry.bvh = trees

    def activate(self):
        """
        Build the per-(solver, mesh-type) BVHs on first activation; idempotent.

        Rigid solvers get collision BVHs over their collision faces, partitioned by owning-link fixedness: faces on
        fixed links move only on an explicit set_pos/set_quat while faces on movable links move on every step, so
        giving each group its own tree keeps the static-rebuild skip (and the shared read across envs) available even
        when movable links share the solver, and scales the per-step rebuild with the movable face count alone. The
        subsets are cast separately and merged in place, giving the same result as one combined tree; sphere-query
        consumers (tactile probes) fold both trees the same way. A solver whose faces are all static or all movable
        keeps a single tree.

        Any solver with entities opting in via ``material.use_visual_raycasting`` gets a visual BVH masked to those
        vfaces. Collision and visual entries coexist (the cast kernels merge in place).
        """
        if self._active:
            return
        self._active = True
        for solver in (self._sim.rigid_solver, self._sim.kinematic_solver):
            if not solver.is_active:
                continue
            n_envs = solver._B
            # A solver's visual geometry is static when no link can be moved by the physics (all links fixed); it then
            # changes only through an explicit set_pos/set_quat/set_vverts, all GEOMETRY mutations the subscription
            # catches. Collision entries refine this to a per-face criterion below.
            maybe_static = all(link.is_fixed for link in solver.links)
            if isinstance(solver, RigidSolver):
                n_faces = solver.dyn_info.faces.geom_idx.shape[0]
                # Fixedness per face, from its owning link. A movable link without collision geoms leaves the tree
                # unaffected, so a per-face criterion catches static collision meshes the per-link one misses.
                faces_geom_idx = qd_to_numpy(solver.dyn_info.faces.geom_idx)
                geoms_link_idx = qd_to_numpy(solver.dyn_info.geoms.link_idx)
                faces_link_idx = geoms_link_idx[faces_geom_idx]
                links_is_fixed = np.array([link.is_fixed for link in solver.links])
                faces_is_static = links_is_fixed[faces_link_idx]
                # A face whose three vertices live in the unbatched fixed-verts buffer reads env-independent build
                # inputs; see BVHContext.is_env_uniform.
                verts_is_fixed = qd_to_numpy(solver.dyn_info.verts.is_fixed)
                faces_is_env_uniform = verts_is_fixed[qd_to_numpy(solver.dyn_info.faces.verts_idx)].all(axis=1)
                # Each subset is (faces_idx, is_static), with faces_idx None when the subset covers every face in
                # order (leaf slot == face index, no remap needed).
                if faces_is_static.all() or not faces_is_static.any():
                    subsets = [(None, faces_is_static.all())]
                else:
                    subsets = [
                        (np.nonzero(faces_is_static)[0], True),
                        (np.nonzero(~faces_is_static)[0], False),
                    ]
                # Identity tree routing shared by this solver's per-env entries: each env casts against its own slot.
                env_tree_identity = torch.arange(n_envs, dtype=gs.tc_int, device=gs.device)
                for subset_faces_idx, is_subset_static in subsets:
                    is_env_uniform = not solver._enable_heterogeneous and bool(
                        faces_is_env_uniform.all()
                        if subset_faces_idx is None
                        else faces_is_env_uniform[subset_faces_idx].all()
                    )
                    faces_idx = None
                    if subset_faces_idx is not None:
                        faces_idx = torch.as_tensor(subset_faces_idx, dtype=gs.tc_int, device=gs.device)
                    # Watch GEOMETRY changes that can reach this entry's faces: filtering the subscription on the
                    # fixed links owning them keeps movable-link mutations (e.g. a per-step robot teleport) from
                    # flagging a rebuild, so the static skip survives them.
                    if is_subset_static:
                        rebuild_subscriber = Subscriber(
                            to=frozenset({StateChange.GEOMETRY}),
                            links_filter=np.unique(
                                faces_link_idx if subset_faces_idx is None else faces_link_idx[subset_faces_idx]
                            ),
                        )
                        solver.subscribe(rebuild_subscriber)
                        # Static collision geometry: the trailing self.update() sizes the (initially None) BVH to one
                        # tree per distinct per-env geometry - a single tree when envs are identical - dropping the
                        # n_envs-fold node/aabb/morton/radix-scratch replication that dominates GPU memory for a
                        # high-poly terrain.
                        entry = BVHContext(
                            solver,
                            raycast_mask=None,
                            maybe_static=True,
                            rebuild_subscriber=rebuild_subscriber,
                            is_env_uniform=is_env_uniform,
                            faces_idx=faces_idx,
                        )
                    else:
                        # A movable subset rebuilds every step and may diverge per env, so keep one tree per env.
                        aabb = AABB(
                            n_batches=n_envs,
                            n_aabbs=n_faces if subset_faces_idx is None else subset_faces_idx.shape[0],
                        )
                        bvh = LBVH(aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64)
                        entry = BVHContext(
                            solver,
                            bvh,
                            aabb,
                            raycast_mask=None,
                            env_bvh_idx=env_tree_identity,
                            faces_idx=faces_idx,
                        )
                    self._bvh_contexts.append(entry)
            mask = solver.vfaces_raycast_mask
            if mask.any():
                if maybe_static:
                    # A static visual entry watches every GEOMETRY change: an explicit set_vverts can reshape any
                    # opted-in vgeom, so its subscription carries no links filter.
                    rebuild_subscriber = Subscriber(to=frozenset({StateChange.GEOMETRY}))
                    solver.subscribe(rebuild_subscriber)
                    entry = BVHContext(
                        solver, raycast_mask=mask, maybe_static=True, rebuild_subscriber=rebuild_subscriber
                    )
                else:
                    # A movable visual mesh rebuilds every step and may diverge per env: one tree per env.
                    aabb = AABB(n_batches=n_envs, n_aabbs=mask.shape[0])
                    bvh = LBVH(aabb, max_n_query_result_per_aabb=0, n_radix_sort_groups=64)
                    env_tree_identity = torch.arange(n_envs, dtype=gs.tc_int, device=gs.device)
                    entry = BVHContext(solver, bvh, aabb, mask, env_bvh_idx=env_tree_identity)
                self._bvh_contexts.append(entry)

        self.update()

    def update(self):
        """Rebuild every BVH whose geometry may have changed since the last cast.

        A static entry (maybe_static: the physics cannot move its geometry) is skipped while it is not flagged for
        rebuild, since its tree would come out unchanged. Its rebuild_subscriber flags it after an explicit
        set_pos/set_quat/set_vverts that can reach its geometry (see BVHContext.rebuild_subscriber), and ``reset``
        flags every entry, so a re-randomized terrain or teleported obstacle still rebuilds. Movable entries are
        never static, so they rebuild on every call.
        """
        if not self._active:
            return
        # When several collision entries of one rigid solver rebuild in the same call (e.g. after a reset), the
        # world-space vertices only need refreshing once: the first entry refreshes them, the rest fit their AABBs
        # over the already-updated vertices.
        verts_updated_solver = None
        for entry in self._bvh_contexts:
            # A pending GEOMETRY change means a set_pos/set_quat/set_vverts hit this otherwise-static geometry since the
            # last build; flag it for rebuild and clear the subscriber so the next idle update skips again.
            if entry.rebuild_subscriber is not None and entry.rebuild_subscriber.pending:
                entry.rebuild_subscriber.clear()
                entry.needs_rebuild = True
            if entry.maybe_static and not entry.needs_rebuild:
                continue
            solver = entry.solver
            if entry.raycast_mask is None:
                if entry.maybe_static:
                    # Static collision geometry: refresh every env's verts, group envs by identical geometry, and
                    # build one tree per distinct geometry from its representative env. A per-env set_pos on a fixed
                    # link splits groups (up to one tree per env) and an identical reset merges them back, so a
                    # shared tree is never stale.
                    if solver is not verts_updated_solver:
                        kernel_update_all_verts(solver.dyn_state, solver.dyn_info, solver.rigid_config)
                    env_bvh_idx, batch_repr_env = self._static_entry_groups(entry)
                    if entry.faces_idx is None:
                        n_slots = solver.dyn_info.faces.geom_idx.shape[0]
                    else:
                        n_slots = entry.faces_idx.shape[0]
                    self._sized_grouped_trees(entry, batch_repr_env.shape[0], n_slots)
                    entry.env_bvh_idx = env_bvh_idx
                    if entry.faces_idx is None:
                        kernel_update_grouped_aabbs(
                            batch_repr_env, solver.dyn_state, entry.aabb, solver.dyn_info, solver.rigid_config
                        )
                        entry.bvh.build()
                    else:
                        kernel_update_grouped_subset_aabbs(
                            batch_repr_env,
                            entry.faces_idx,
                            solver.dyn_state,
                            entry.aabb,
                            solver.dyn_info,
                            solver.rigid_config,
                        )
                        entry.bvh.build()
                        # build() resets the leaf payloads to subset slots; rewrite them to global faces (see
                        # kernel_remap_leaf_faces) so every traversal is subset-agnostic.
                        kernel_remap_leaf_faces(entry.faces_idx, entry.bvh.morton_codes)
                elif entry.faces_idx is None:
                    kernel_update_verts_and_aabbs(solver.dyn_state, entry.aabb, solver.dyn_info, solver.rigid_config)
                    entry.bvh.build()
                else:
                    subset_aabbs_kernel = (
                        kernel_update_subset_aabbs
                        if solver is verts_updated_solver
                        else kernel_update_verts_and_subset_aabbs
                    )
                    subset_aabbs_kernel(
                        entry.faces_idx, solver.dyn_state, entry.aabb, solver.dyn_info, solver.rigid_config
                    )
                    entry.bvh.build()
                    # See the grouped subset branch above for the leaf-payload rewrite.
                    kernel_remap_leaf_faces(entry.faces_idx, entry.bvh.morton_codes)
                verts_updated_solver = solver
            else:
                # Reads vverts_state.pos as the source of vvert positions. The buffer is seeded by FK at scene.build()
                # and refreshed for each user-driven entity via set_vverts; entries set via set_vverts survive across
                # calls until set_vverts(None) re-runs FK over the entity's vgeoms. raycast_mask gates which vfaces
                # contribute to the BVH; masked-out vfaces keep an inverted AABB and are skipped by ray queries.
                solver.update_forward_pos()
                solver.update_vgeoms()
                if entry.maybe_static:
                    # Static visual geometry: group envs by identical visual geometry and build one tree per
                    # distinct geometry from its representative env, mirroring the static collision path above.
                    env_bvh_idx, batch_repr_env = self._static_entry_groups(entry)
                    self._sized_grouped_trees(entry, batch_repr_env.shape[0], solver.dyn_info.vfaces.vgeom_idx.shape[0])
                    entry.env_bvh_idx = env_bvh_idx
                    kernel_update_grouped_visual_aabbs(
                        batch_repr_env,
                        entry.raycast_mask,
                        solver.dyn_state,
                        entry.aabb,
                        solver.dyn_info,
                        solver.rigid_config,
                    )
                else:
                    kernel_update_visual_aabbs(
                        entry.raycast_mask, solver.dyn_state, entry.aabb, solver.dyn_info, solver.rigid_config
                    )
                entry.bvh.build()
            entry.needs_rebuild = False

    def reset(self, envs_idx):
        # A reset may change otherwise-static geometry (re-randomized terrain, teleported obstacles), so force every
        # entry to rebuild once; static entries resume skipping on subsequent steps. The BVHs are geometry-global, not
        # per-env, so ``envs_idx`` is unused. No-op when inactive (``_bvh_contexts`` is empty).
        for entry in self._bvh_contexts:
            entry.needs_rebuild = True
        self.update()

    def destroy(self):
        self._bvh_contexts.clear()


@dataclass
class RaycasterSharedMetadata(KinematicSensorMetadataMixin, SimpleSensorMetadata):
    # The BVHs cast against each frame live on the shared ``RaycastContext`` (one per active solver per mesh type),
    # so a Raycaster and a DepthCamera share one set of trees. The cast entries chain into the output cache; see
    # write_ray_hit in raycast_qd.py for the merge scheme. Per-sensor link poses are gathered via
    # KinematicSensorMetadataMixin.solver_groups, independent of which BVH is being cast.

    # Per-step scratch tensors for sensor link poses, lazily allocated on the first cast (B and n_sensors known).
    links_pos: torch.Tensor | None = None
    links_quat: torch.Tensor | None = None

    sensors_ray_start_idx: list[int] = field(default_factory=list)
    total_n_rays: int = 0
    total_cache_size: int = 0

    min_ranges: torch.Tensor = make_tensor_field((0,))
    max_ranges: torch.Tensor = make_tensor_field((0,))
    no_hit_values: torch.Tensor = make_tensor_field((0,))
    return_world_frame: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)

    patterns: list[RaycastPattern] = field(default_factory=list)
    ray_dirs: torch.Tensor = make_tensor_field((0, 3))
    ray_starts: torch.Tensor = make_tensor_field((0, 3))
    ray_starts_world: torch.Tensor = make_tensor_field((0, 3))
    ray_dirs_world: torch.Tensor = make_tensor_field((0, 3))

    points_to_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_offsets: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_point_offsets: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_point_counts: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_return_points: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)


class RaycasterReturnType(NamedTuple):
    points: torch.Tensor | None
    distances: torch.Tensor


class RaycasterSensor(
    KinematicSensorMixin, SimpleSensor[RaycasterOptions, RaycastContext, RaycasterSharedMetadata, RaycasterReturnType]
):
    def __init__(self, options: RaycasterOptions, idx: int, shared_context, shared_metadata, manager: "SensorManager"):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self.debug_objects: list["Mesh"] = []
        self.ray_starts: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)
        self.ray_dirs: torch.Tensor = torch.empty((0, 3), device=gs.device, dtype=gs.tc_float)

    def build(self):
        super().build()

        # A raycaster always casts, so activate the shared ``RaycastContext`` now: the first consumer's activation
        # builds the BVHs. Every raycaster then validates there is geometry to cast against.
        self._shared_context.activate()
        # The first raycaster seeds the leading boundary (0) of the per-sensor offsets into the shared cache tensor.
        if self._idx == 0:
            self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
                self._shared_metadata.sensor_cache_offsets, 0
            )
        if not self._shared_context.bvh_contexts:
            gs.raise_exception(
                "Raycaster sensor has no geometry to raycast against: rigid_solver is inactive and no entity "
                "has material.use_visual_raycasting=True."
            )

        self._shared_metadata.patterns.append(self._options.pattern)

        ray_starts = self._options.pattern.ray_starts.reshape(-1, 3)
        self.ray_starts = transform_by_trans_quat(
            ray_starts, self._shared_metadata.offsets_pos[0, -1, :], self._shared_metadata.offsets_quat[0, -1, :]
        )
        self._shared_metadata.ray_starts = torch.cat([self._shared_metadata.ray_starts, self.ray_starts])

        ray_dirs = self._options.pattern.ray_dirs.reshape(-1, 3)
        self.ray_dirs = transform_by_quat(ray_dirs, self._shared_metadata.offsets_quat[0, -1, :])
        self._shared_metadata.ray_dirs = torch.cat([self._shared_metadata.ray_dirs, self.ray_dirs])

        num_rays = math.prod(self._options.pattern.return_shape)
        self._shared_metadata.sensors_ray_start_idx.append(self._shared_metadata.total_n_rays)

        # Cache offsets are a running cumulative sum of the per-sensor cache sizes, so sensors with different sizes
        # (e.g. a points lidar next to a distances-only depth camera) pack without gaps or overlap.
        self._shared_metadata.total_cache_size += self._cache_size
        self._shared_metadata.sensor_cache_offsets = concat_with_tensor(
            self._shared_metadata.sensor_cache_offsets, self._shared_metadata.total_cache_size
        )
        self._shared_metadata.sensor_point_offsets = concat_with_tensor(
            self._shared_metadata.sensor_point_offsets, self._shared_metadata.total_n_rays
        )
        self._shared_metadata.sensor_point_counts = concat_with_tensor(
            self._shared_metadata.sensor_point_counts, num_rays
        )
        self._shared_metadata.sensor_return_points = concat_with_tensor(
            self._shared_metadata.sensor_return_points, self._options.return_points
        )
        self._shared_metadata.total_n_rays += num_rays

        self._shared_metadata.points_to_sensor_idx = concat_with_tensor(
            self._shared_metadata.points_to_sensor_idx, [self._idx] * num_rays, flatten=True
        )
        self._shared_metadata.return_world_frame = concat_with_tensor(
            self._shared_metadata.return_world_frame, self._options.return_world_frame
        )
        self._shared_metadata.min_ranges = concat_with_tensor(self._shared_metadata.min_ranges, self._options.min_range)
        self._shared_metadata.max_ranges = concat_with_tensor(self._shared_metadata.max_ranges, self._options.max_range)
        self._shared_metadata.no_hit_values = concat_with_tensor(
            self._shared_metadata.no_hit_values, self._options.no_hit_value
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        shape = self._options.pattern.return_shape
        # Distances-only: drop the (*shape, 3) points field so the cache holds just the distances.
        if not self._options.return_points:
            return (shape,)
        return ((*shape, 3), shape)

    def _get_formatted_data(self, tensor: torch.Tensor, envs_idx=None) -> RaycasterReturnType:
        # With points disabled the base class returns a bare distances tensor; re-wrap it as RaycasterReturnType so
        # the (points, distances) NamedTuple contract holds, with points=None.
        data = super()._get_formatted_data(tensor, envs_idx)
        if self._options.return_points:
            return data
        return RaycasterReturnType(points=None, distances=data)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_raw_data(
        cls, shared_context: RaycastContext, shared_metadata: RaycasterSharedMetadata, raw_data_T: torch.Tensor
    ):
        # The BVHs were already refreshed once this step by SensorManager (``RaycastContext.update``); read them here.
        bvh_contexts = shared_context.bvh_contexts

        # Allocate the link-pose scratch buffers on first cast (B and n_sensors are known here). Identity quat is baked
        # into the initial allocation so static sensors (entity_idx<0) leave their rows at identity, letting the cast
        # kernel apply pos_offset / euler_offset in world frame.
        if shared_metadata.links_pos is None:
            B = bvh_contexts[0].solver._B
            shared_metadata.links_pos = torch.zeros(
                B, shared_metadata.n_sensors, 3, device=gs.device, dtype=gs.tc_float
            )
            shared_metadata.links_quat = torch.zeros(
                B, shared_metadata.n_sensors, 4, device=gs.device, dtype=gs.tc_float
            )
            shared_metadata.links_quat[:, :, 0] = 1.0

        # Gather link poses per sensor. Sensors are pre-bucketed into shared_metadata.solver_groups at build time so
        # this loop issues one bulk get_links_pos / get_links_quat per solver with already-tensor-typed indices.
        links_pos = shared_metadata.links_pos
        links_quat = shared_metadata.links_quat
        for group in shared_metadata.solver_groups:
            pos = group.solver.get_links_pos(links_idx=group.links_idx)
            quat = group.solver.get_links_quat(links_idx=group.links_idx)
            if group.solver.n_envs == 0:
                pos = pos[None]
                quat = quat[None]
            links_pos[:, group.sensor_cols, :] = pos
            links_quat[:, group.sensor_cols, :] = quat

        # The entries chain into one output buffer: the first initializes every slot (is_merge=False), each subsequent
        # one merges in closer hits, and the final one (is_last) settles misses to no_hit_value - see write_ray_hit.
        for i, entry in enumerate(bvh_contexts):
            solver = entry.solver
            args_common = (
                shared_metadata.points_to_sensor_idx,
                entry.env_bvh_idx,
                entry.bvh.nodes,
                entry.bvh.morton_codes,
                links_pos,
                links_quat,
                shared_metadata.ray_starts,
                shared_metadata.ray_dirs,
                shared_metadata.max_ranges,
                shared_metadata.no_hit_values,
                shared_metadata.return_world_frame,
                shared_metadata.sensor_cache_offsets,
                shared_metadata.sensor_point_offsets,
                shared_metadata.sensor_point_counts,
                shared_metadata.sensor_return_points,
                raw_data_T,
            )
            if entry.raycast_mask is None:
                kernel_cast_rays(
                    *args_common,
                    solver.dyn_state,
                    solver.dyn_info,
                    eps=gs.EPS,
                    is_merge=i > 0,
                    is_last=i == len(bvh_contexts) - 1,
                    # One tree serving several envs wants the env-major thread mapping (see kernel_cast_rays).
                    is_env_major=entry.aabb.n_batches < solver._B,
                )
            else:
                kernel_cast_rays_visual(
                    *args_common,
                    solver.dyn_state,
                    solver.dyn_info,
                    eps=gs.EPS,
                    is_merge=i > 0,
                    is_last=i == len(bvh_contexts) - 1,
                    # One tree serving several envs wants the env-major thread mapping (see kernel_cast_rays).
                    is_env_major=entry.aabb.n_batches < solver._B,
                )

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw hit points as spheres in the scene.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        data = self.read(env_idx)

        pos = self._link.get_pos(env_idx, relative=False)
        quat = self._link.get_quat(env_idx, relative=False)
        if pos.ndim == 2:
            pos, quat = pos[0], quat[0]

        ray_starts = transform_by_trans_quat(self.ray_starts, pos, quat)

        if self._options.return_points:
            points = data.points.reshape((-1, 3))
            if not self._options.return_world_frame:
                points = transform_by_trans_quat(points + self.ray_starts, pos, quat)
        else:
            # Reconstruct the local-frame hit points as distance * unit ray_dir. Missed rays carry exactly
            # no_hit_value as distance (which may lie below max_range, so an ordering test cannot discriminate) and
            # collapse onto the ray start, matching the (0, 0, 0) stored for them when points are enabled.
            distances = data.distances.reshape((-1, 1))
            hit_points_local = torch.where(
                distances != self._options.no_hit_value, distances * normalize(self.ray_dirs), 0.0
            )
            points = transform_by_trans_quat(hit_points_local + self.ray_starts, pos, quat)

        for debug_object in self.debug_objects:
            context.clear_debug_object(debug_object)
        self.debug_objects.clear()

        self.debug_objects += [
            context.draw_debug_spheres(
                ray_starts, radius=self._options.debug_sphere_radius, color=self._options.debug_ray_start_color
            ),
            context.draw_debug_spheres(
                points, radius=self._options.debug_sphere_radius, color=self._options.debug_ray_hit_color
            ),
        ]
