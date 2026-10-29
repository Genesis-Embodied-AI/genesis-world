"""Translate USD collision filtering (CollisionGroup / FilteredPairsAPI) into contype/conaffinity."""

from pxr import Usd, UsdPhysics

import genesis as gs
from genesis.utils.collision import solve_contype_conaffinity


def _geoms_under(cg_infos: list[dict], path: str) -> list[int]:
    """Indices of collision g_infos whose source prim is `path` or a descendant of it."""
    prefix = path.rstrip("/") + "/"
    return [i for i, g in enumerate(cg_infos) if g["prim_path"] == path or g["prim_path"].startswith(prefix)]


def apply_collision_filtering(context, cg_infos: list[dict]):
    """
    Set contype/conaffinity on collision g_infos from USD collision filtering.

    Honors ``UsdPhysics.CollisionGroup`` (collider membership + ``filteredGroups`` + ``mergeGroupName``)
    and per-prim ``UsdPhysics.FilteredPairsAPI``, for the geoms in ``cg_infos`` (a single entity). The
    resulting "must not collide" pairs are realized as contype/conaffinity bitmasks by the shared
    solver; if they cannot be expressed as bitmasks, a warning is logged and defaults are kept.

    Filtering that spans entities is not expressible per entity, so it is detected against the stage's
    full collider set and reported once via ``context.note_unsupported_cross_entity_filtering()``.

    ``cg_infos`` must each carry a ``prim_path``.
    """
    if not cg_infos:
        return

    stage: Usd.Stage = context.stage
    n = len(cg_infos)
    invalid_pairs: set[frozenset[int]] = set()

    # Single stage sweep gathering all filtering-relevant prims. The full collider set (this entity's
    # geoms are a subset) makes cross-entity filtering relationships visible: add_stage splits the
    # stage into entities, so those relationships cannot be expressed by this per-entity pass.
    stage_collider_paths: set[str] = set()
    group_prims = []
    filter_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            stage_collider_paths.add(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.CollisionGroup):
            group_prims.append(prim)
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            filter_prims.append(prim)
    entity_paths = {g["prim_path"] for g in cg_infos}
    external_collider_paths = stage_collider_paths - entity_paths
    has_cross_entity_filtering = False

    # 1) CollisionGroup filtering: members of groups that filter each other must not collide. Groups
    # sharing a non-empty mergeGroupName are unified under one canonical id.
    canonical_of = {}
    for prim in group_prims:
        path = str(prim.GetPath())
        merge_name = UsdPhysics.CollisionGroup(prim).GetMergeGroupNameAttr().Get() or ""
        canonical_of[path] = merge_name if merge_name else path

    membership_queries = []
    filtered_of: dict[str, set[str]] = {}
    for prim in group_prims:
        path = str(prim.GetPath())
        group = UsdPhysics.CollisionGroup(prim)
        if group.GetInvertFilteredGroupsAttr().Get():
            # Inverted semantics (collide ONLY with the listed groups) are uncommon and not modeled;
            # skip this group's filtering rather than risk disabling the wrong pairs.
            gs.logger.warning(
                f"USD CollisionGroup '{path}' uses invertFilteredGroups, which is not supported; "
                "its collision filtering is ignored."
            )
            continue
        canonical = canonical_of[path]
        membership_queries.append((canonical, group.GetCollidersCollectionAPI().ComputeMembershipQuery()))
        filtered = filtered_of.setdefault(canonical, set())
        for target in group.GetFilteredGroupsRel().GetTargets():
            filtered.add(canonical_of.get(str(target), str(target)))

    if filtered_of:
        group_members: dict[str, set[str]] = {}
        for canonical, query in membership_queries:
            group_members.setdefault(canonical, set()).update(
                p for p in stage_collider_paths if query.IsPathIncluded(p)
            )
        geoms_canonicals = [
            {canonical for canonical, members in group_members.items() if g["prim_path"] in members} for g in cg_infos
        ]
        for i in range(n):
            for j in range(i + 1, n):
                if any(
                    (gb in filtered_of.get(ga, ())) or (ga in filtered_of.get(gb, ()))
                    for ga in geoms_canonicals[i]
                    for gb in geoms_canonicals[j]
                ):
                    invalid_pairs.add(frozenset((i, j)))
        if external_collider_paths:
            for canonical_a, partners in filtered_of.items():
                for canonical_b in partners:
                    members = group_members.get(canonical_a, set()) | group_members.get(canonical_b, set())
                    if (members & entity_paths) and (members & external_collider_paths):
                        has_cross_entity_filtering = True

    # 2) FilteredPairsAPI: explicit prim-subtree pairs that must not collide.
    for prim in filter_prims:
        geoms_a = _geoms_under(cg_infos, str(prim.GetPath()))
        if not geoms_a:
            continue
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            target_path = str(target)
            target_geoms = _geoms_under(cg_infos, target_path)
            if not target_geoms:
                # The target collider (or one below it) lives in another entity.
                target_prefix = target_path.rstrip("/") + "/"
                if any(p == target_path or p.startswith(target_prefix) for p in external_collider_paths):
                    has_cross_entity_filtering = True
            for i in geoms_a:
                for j in target_geoms:
                    if i != j:
                        invalid_pairs.add(frozenset((i, j)))

    if invalid_pairs:
        masks = solve_contype_conaffinity(n, invalid_pairs)
        if masks is None:
            gs.logger.warning(
                "USD collision filtering cannot be expressed with 'contype'/'conaffinity' bitmasks. "
                "Keeping default (all-colliding) values."
            )
        else:
            for g_info, (contype, conaffinity) in zip(cg_infos, masks):
                g_info["contype"], g_info["conaffinity"] = contype, conaffinity

    if has_cross_entity_filtering:
        context.note_unsupported_cross_entity_filtering()
