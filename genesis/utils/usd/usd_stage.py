from contextlib import contextmanager
from typing import Iterator, List, Set

import genesis as gs
from pxr import Sdf, Usd, UsdPhysics

from .usd_context import (
    UsdContext,
    extract_links_referenced_by_joints,
    find_joints_in_range,
    find_rigid_bodies_in_range,
    resolve_rigid_body_link_path,
    usd_context,
)
from .usd_utils import UnionFind


def _find_common_ancestor_path(paths: List[str]) -> str:
    """
    Find the common ancestor path of a list of prim paths.

    Parameters
    ----------
    paths : List[str]
        List of prim paths.

    Returns
    -------
    str
        The common ancestor path (longest common prefix).
    """
    if not paths:
        return "/"

    # Split paths into components, filtering out empty strings (from leading slash)
    path_components = [[comp for comp in path.split("/") if comp] for path in paths]

    # Find the minimum length
    min_len = min(len(components) for components in path_components)

    # Find common prefix
    common_components = []
    for i in range(min_len):
        if all(components[i] == path_components[0][i] for components in path_components):
            common_components.append(path_components[0][i])
        else:
            break

    if not common_components:
        return "/"

    return "/" + "/".join(common_components)


def _find_connected_components(stage: Usd.Stage, all_joints: List[Usd.Prim]) -> List[tuple[List[Usd.Prim], Set[str]]]:
    """
    Find connected components in the joint graph using union-find (disjoint set) algorithm.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage.
    all_joints : List[Usd.Prim]
        List of all joint prims in the stage.

    Returns
    -------
    List[tuple[List[Usd.Prim], Set[str]]]
        List of (joints, links) tuples for each connected component.
    """
    # Union-Find data structure
    uf = UnionFind()

    # Build joint-to-links mapping and union connected links
    joint_to_links: dict[Usd.Prim, tuple[str | None, str | None]] = {}
    all_link_paths: Set[str] = set()

    for joint_prim in all_joints:
        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0_path = resolve_rigid_body_link_path(stage, str(body0_targets[0])) if body0_targets else None
        body1_path = resolve_rigid_body_link_path(stage, str(body1_targets[0])) if body1_targets else None

        joint_to_links[joint_prim] = (body0_path, body1_path)

        # Union connected links. A target that resolves to no rigid-body link is None; a joint with a
        # single resolved side still contributes that link as a standalone component member.
        if body0_path and body1_path:
            all_link_paths.add(body0_path)
            all_link_paths.add(body1_path)
            uf.union(body0_path, body1_path)
        elif body0_path:
            all_link_paths.add(body0_path)
            uf.find(body0_path)  # Ensure it's in the union-find structure
        elif body1_path:
            all_link_paths.add(body1_path)
            uf.find(body1_path)  # Ensure it's in the union-find structure

    # Group links and joints by connected component
    component_roots: dict[str, tuple[List[Usd.Prim], Set[str]]] = {}

    for link_path in all_link_paths:
        root = uf.find(link_path)
        if root not in component_roots:
            component_roots[root] = ([], set())
        component_roots[root][1].add(link_path)

    # Add joints to their components
    # Only add joints that connect at least one rigid body
    for joint_prim, (body0_path, body1_path) in joint_to_links.items():
        # Add joint to component based on which body is a rigid body
        if body0_path and body0_path in all_link_paths:
            root = uf.find(body0_path)
            if root in component_roots and joint_prim not in component_roots[root][0]:
                component_roots[root][0].append(joint_prim)
        elif body1_path and body1_path in all_link_paths:
            root = uf.find(body1_path)
            if root in component_roots and joint_prim not in component_roots[root][0]:
                component_roots[root][0].append(joint_prim)

    # Return only components that have joints
    return [(joints, links) for joints, links in component_roots.values() if joints]


@contextmanager
def parse_usd_stage(morph: gs.morphs.USD) -> Iterator[List[gs.morphs.USD]]:
    """
    Parse a USD stage and extract all rigid entities (articulations and rigid bodies) as separate USD morphs.

    This function uses a graph-based approach to identify connected components: joints are edges, links (rigid bodies
    referenced by joints) are nodes, each connected component becomes one articulation entity, and pure rigid bodies
    (not referenced by any joint) become separate entities. Joint prims are not stored on morphs; they are deduced
    dynamically when each entity is parsed via parse_usd_rigid_entity.

    Parameters
    ----------
    morph : gs.morphs.USD
        The USD morph naming the stage to parse.

    Yields
    ------
    morphs: List[gs.morphs.USD]
        A list of USD morphs, one for each rigid entity found in the stage. Each morph is a copy of the input stage with
        its `prim_path` set to the topmost common ancestor of all links in the component. The list is guaranteed to be
        non-empty (raises an exception if no entities are found). The stage stays open for the block, so the entities
        added from these morphs inside it are parsed from that one opening (see 'usd_context').
    """
    with usd_context(morph.file) as context:
        usd_stage: Usd.Stage = context.stage

        # Find all joints in the stage
        all_joints = find_joints_in_range(Usd.PrimRange(usd_stage.GetPseudoRoot()))

        # Find all rigid bodies in the stage (prune children when rigid body is found)
        all_rigid_bodies = find_rigid_bodies_in_range(Usd.PrimRange(usd_stage.GetPseudoRoot()))

        # Extract links referenced by joints (only rigid bodies)
        links_referenced_by_joints = extract_links_referenced_by_joints(usd_stage, all_joints, check_rigid_body=True)

        morphs: List[gs.morphs.USD] = []
        components: List[tuple[List[Usd.Prim], Set[str]]] = []
        # Process connected components (articulations)
        if all_joints:
            components = _find_connected_components(usd_stage, all_joints)
            for component_joints, component_links in components:
                # Find topmost common ancestor of all links in this component
                link_paths = list(component_links)
                common_ancestor_path = _find_common_ancestor_path(link_paths)
                common_ancestor_prim = usd_stage.GetPrimAtPath(common_ancestor_path)

                assert common_ancestor_prim.IsValid(), f"Invalid common ancestor path: {common_ancestor_path}"

                # Create morph for this connected component
                entity_morph = morph.model_copy()
                entity_morph.prim_path = common_ancestor_path
                morphs.append(entity_morph)

        # Process pure rigid bodies (not referenced by joints)
        # Links referenced by joints are already part of articulations, so exclude them
        pure_rigid_bodies = all_rigid_bodies - links_referenced_by_joints

        for rigid_body_path in pure_rigid_bodies:
            entity_morph = morph.model_copy()
            entity_morph.prim_path = rigid_body_path
            morphs.append(entity_morph)

        if not morphs:
            gs.raise_exception("No entities found in USD stage.")

        num_articulations = len(components) if all_joints else 0
        gs.logger.debug(
            f"Found {len(morphs)} rigid entity(ies) from USD stage: {num_articulations} articulation(s), "
            f"{len(pure_rigid_bodies)} pure rigid body(ies)."
        )

        yield morphs
