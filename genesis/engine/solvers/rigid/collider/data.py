"""The model, state and scratch of the collision detection of the rigid solver."""

import quadrants as qd

from genesis.engine.solvers.rigid.data import GEOMS, LINKS, VERTS
from genesis.utils.data import ENVS, WORKERS, Config, Data, DataKind, Size, field


# Sizes of the collision detection: the link pairs holding a valid geom pair, the possible geom pairs and those the
# broad phase keeps at most
VALID_LINK_PAIRS = Size("valid_link_pairs")
POSSIBLE_PAIRS = Size("possible_pairs")
BROAD_PAIRS = Size("broad_pairs")
# Contacts the narrow phase detects at most per environment, and per nonconvex pair
CANDIDATE_CONTACTS = Size("candidate_contacts")
NONCONVEX_PAIR_CONTACTS = Size("nonconvex_pair_contacts")
# A pair holds its first contact and its four perturbed ones (see N_PERTURBATIONS in narrowphase.py), or its first
# contact alone under the contact patch
CONTACT_SLOTS = Size("contact_slots")
# Threads of the terrain pass, each testing the prism under one triangle
TERRAIN_THREADS = Size("terrain_threads")

# The scratch of the box-box detection
BOX_BOX = "box_box"
# The contact cache, which the convex-convex detection alone reads
CONTACT_CACHE = "contact_cache"
# The inputs of the differentiable contact detection, which the backward pass reads
DIFF_CONTACTS = "diff_contacts"
# The work queue of the split narrowphase and of the terrain pass, its contact slots, and the status of each slot
WORK_QUEUE = "work_queue"
SPLIT_NARROWPHASE = "split_narrowphase"
PERTURBED_SLOTS = "perturbed_slots"
# The scratch of the link-pair contact pruning, read by the threads of the block running an environment
PRUNING = "pruning"


class ContactData(Data, kind=DataKind.DERIVED, shape=(CANDIDATE_CONTACTS, ENVS)):
    """The contacts found in each environment: geoms, links, frame, friction, solver parameters and force."""

    geom_a: qd.Tensor = field(int)
    geom_b: qd.Tensor = field(int)
    normal: qd.Tensor = field((float, 3), grad=True)
    pos: qd.Tensor = field((float, 3), grad=True)
    penetration: qd.Tensor = field(float, grad=True)
    friction: qd.Tensor = field(float)
    friction_torsional: qd.Tensor = field(float)
    friction_rolling: qd.Tensor = field(float)
    sol_params: qd.Tensor = field((float, 7))
    force: qd.Tensor = field((float, 3))
    link_a: qd.Tensor = field(int)
    link_b: qd.Tensor = field(int)
    pair_idx: qd.Tensor = field(int)


class DiffContactInput(Data, kind=DataKind.SCRATCH):
    """The inputs of the differentiable contact detection, from which the backward pass recomputes each contact."""

    geom_a: qd.Tensor = field(int)
    geom_b: qd.Tensor = field(int)
    # Local positions of the vertices of the two geoms defining the face of the Minkowski difference
    local_pos1_a: qd.Tensor = field((float, 3))
    local_pos1_b: qd.Tensor = field((float, 3))
    local_pos1_c: qd.Tensor = field((float, 3))
    local_pos2_a: qd.Tensor = field((float, 3))
    local_pos2_b: qd.Tensor = field((float, 3))
    local_pos2_c: qd.Tensor = field((float, 3))
    # Local positions of the vertices of the two geoms defining the support point of that face
    w_local_pos1: qd.Tensor = field((float, 3))
    w_local_pos2: qd.Tensor = field((float, 3))
    # Support core of the convex geom of a plane-convex contact (box vertex, sphere center, nearest capsule endpoint),
    # in its local frame
    core_local: qd.Tensor = field((float, 3))
    # Plane-convex contacts only: whether the detection returned a zero penetration (see
    # func_differentiable_plane_contact)
    has_zero_penetration: qd.Tensor = field(bool)
    ref_id: qd.Tensor = field(int)
    # Analytic contacts only: the multi-contact detection that found the contact, 0 for the unperturbed one
    detection_idx: qd.Tensor = field(int)
    # Whether the forward and backward passes compute the contact stably
    valid: qd.Tensor = field(int)
    # Penetration depth weighting the contact, the input the gradients flow to
    ref_penetration: qd.Tensor = field(float, grad=True)


class SortBuffer(Data, kind=DataKind.STATE, shape=(2 * LINKS, ENVS)):
    """The sorted bounds of the link bounding boxes along the broadphase sweep axis, per environment."""

    value: qd.Tensor = field(float)
    i_l: qd.Tensor = field(int)
    is_max: qd.Tensor = field(bool)


class ContactCache(Data, kind=DataKind.STATE, shape=(POSSIBLE_PAIRS, ENVS)):
    """The contact normal and penetration of each geom pair at the previous step, in each environment."""

    normal: qd.Tensor = field((float, 3))
    # Penetration of each pair at the previous step, 0 out of contact, which gates the fallback from MPR to GJK
    penetration: qd.Tensor = field(float)


class NarrowphaseWorkQueue(Data, kind=DataKind.SCRATCH, shape=(BROAD_PAIRS * ENVS,)):
    """The queue of geom pairs of the split narrowphase and terrain pass, with their candidate contacts."""

    # The pair of every entry (see collision_pair_idx in ColliderInfo), queued for the multicontact pass, then for the
    # terrain pass, which runs after it
    i_b: qd.Tensor = field(int, labels=WORK_QUEUE)
    i_ga: qd.Tensor = field(int, labels=WORK_QUEUE)
    i_gb: qd.Tensor = field(int, labels=WORK_QUEUE)
    i_pair: qd.Tensor = field(int, labels=WORK_QUEUE)
    # Candidate contacts of the multicontact pass, one slot per contact a pair can hold. The contact0 kernel stores the
    # first contact of the pair in slot 0, the detections of the pass store theirs in their own slot, and the gather
    # accepts them in slot order according to their status (see MULTICONTACT_SLOT in collider/constants.py). Under the
    # contact patch an entry holds slot 0 alone and no status.
    contact_pos: qd.Tensor = field((float, 3), shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=SPLIT_NARROWPHASE)
    contact_normal: qd.Tensor = field((float, 3), shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=SPLIT_NARROWPHASE)
    contact_penetration: qd.Tensor = field(float, shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=SPLIT_NARROWPHASE)
    contact_status: qd.Tensor = field(int, shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=PERTURBED_SLOTS)
    # Where contact 0 of the entry stands (see MULTICONTACT_ENTRY in collider/constants.py). The contact0 kernel leaves
    # it to Gilbert-Johnson-Keerthi (GJK) when the per-pair gate from Minkowski portal refinement (MPR) to GJK fired,
    # and each perturbed contact otherwise falls back from MPR to GJK.
    contact0_status: qd.Tensor = field(int, labels=SPLIT_NARROWPHASE)
    # The number of pairs queued and claimed so far
    n_pairs: qd.Tensor = field(int, shape=(1,), labels=WORK_QUEUE)
    n_claimed: qd.Tensor = field(int, shape=(1,), labels=WORK_QUEUE)


class ColliderState(Data, kind=DataKind.SCRATCH):
    """The collision detection state of each environment: broadphase buffers, contacts and their scratch."""

    sort_buffer: SortBuffer
    active_buffer: qd.Tensor = field(int, shape=(LINKS, ENVS))
    # Bounding box of every link, the box of its frame moved with it (see func_collision_clear in broadphase.py)
    links_aabb_min: qd.Tensor = field((float, 3), shape=(LINKS, ENVS))
    links_aabb_max: qd.Tensor = field((float, 3), shape=(LINKS, ENVS))
    # Valid link pairs whose boxes overlap, by index into ColliderInfo.valid_link_pairs
    n_broad_link_pairs: qd.Tensor = field(int, shape=(ENVS,))
    broad_link_pairs: qd.Tensor = field(int, shape=(VALID_LINK_PAIRS, ENVS))
    n_broad_pairs: qd.Tensor = field(int, shape=(ENVS,))
    box_depth: qd.Tensor = field(float, shape=(NONCONVEX_PAIR_CONTACTS, ENVS), labels=BOX_BOX)
    box_points: qd.Tensor = field((float, 3), shape=(NONCONVEX_PAIR_CONTACTS, ENVS), labels=BOX_BOX)
    box_pts: qd.Tensor = field((float, 3), shape=(6, ENVS), labels=BOX_BOX)
    box_lines: qd.Tensor = field((float, 6), shape=(4, ENVS), labels=BOX_BOX)
    box_linesu: qd.Tensor = field((float, 6), shape=(4, ENVS), labels=BOX_BOX)
    box_axi: qd.Tensor = field((float, 3), shape=(3, ENVS), labels=BOX_BOX)
    box_ppts2: qd.Tensor = field(float, shape=(4, 2, ENVS), labels=BOX_BOX)
    box_pu: qd.Tensor = field((float, 3), shape=(4, ENVS), labels=BOX_BOX)
    # The prism under the triangle every thread of the terrain pass tests, which reads its thread count from its shape
    prism: qd.Tensor = field((float, 3), shape=(6, TERRAIN_THREADS))
    n_contacts: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    # Kept contacts of the sleepers, at the front of the contact buffer with the identity permutation: a hibernated
    # link's contacts against fixed links and other sleepers are carried from step to step with the force of the last
    # solve they took part in, for the contact getters and the per-link contact force alone. Every per-step pass
    # (prune, sort, island edges, constraint rows, noslip) walks the live contacts after them,
    # [n_contacts_hibernated, n_contacts).
    n_contacts_hibernated: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    first_time: qd.Tensor = field(bool, shape=(ENVS,), kind=DataKind.STATE)
    contact_cache: ContactCache = field(labels=CONTACT_CACHE)
    broad_collision_pairs: qd.Tensor = field((int, 2), shape=(BROAD_PAIRS, ENVS))
    contact_data: ContactData
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: DiffContactInput = field(shape=(CANDIDATE_CONTACTS, ENVS), labels=DIFF_CONTACTS)
    work_queue: NarrowphaseWorkQueue
    contact_sort_key: qd.Tensor = field(float, shape=(CANDIDATE_CONTACTS, ENVS))
    contact_sort_idx: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), kind=DataKind.DERIVED)
    contact_proj_v: qd.Tensor = field(float, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)
    contact_keep: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)
    contact_hull_stack: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)
    # Permutation of the (u, v) lexicographic sort of the cooperative contact deduplication (see
    # func_clamp_prune_contacts_coop)
    contact_lex_idx: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)


class VertsSpatialGrid(Data, kind=DataKind.CONSTANT):
    """The uniform grid of the collision vertices of each geom in its frame, bounding a vertex scan."""

    # Per-geom 8x8x8 grid over collision verts in the local AABB: a permutation of vert indices sorted by grid
    # cell (z fastest), the matching vert positions duplicated in that order for sequential streaming, and per-cell
    # vert ranges (8^3 + 1 entries per geom), so a scan visits only the cells overlapping a query box. The cell
    # mapping is anchored by geoms_origin / geoms_inv_cell_size in the geom frame.
    verts_idx: qd.Tensor = field(int, shape=(VERTS,))
    verts_pos: qd.Tensor = field((float, 3), shape=(VERTS,))
    cells_vert_start: qd.Tensor = field(int, shape=(GEOMS * (8**3 + 1),))
    geoms_origin: qd.Tensor = field((float, 3), shape=(GEOMS,))
    geoms_inv_cell_size: qd.Tensor = field((float, 3), shape=(GEOMS,))


class ColliderConfig(Config):
    """The compile-time configuration of the kernels of the collision detection of the rigid solver."""

    has_terrain: bool
    # Whether a convex-convex pair reaches the general narrowphase past the box-box and plane-box specializations: the
    # split narrowphase on GPU, func_narrow_phase_convex_vs_convex on CPU. It also gates the deterministic contact sort
    # (see func_sort_contacts). See its resolution in collider.py.
    has_non_box_plane_convex_convex: bool
    has_convex_specialization: bool
    has_nonconvex_nonterrain: bool
    # Whether the link-pair contact pruning pass is compiled in, see its resolution in collider.py
    has_prunable_contacts: bool
    # Contacts a convex pair and a nonconvex (vertex-versus-SDF) pair hold at most, the second one at least the first
    n_contacts_per_convex_pair: int
    n_contacts_per_nonconvex_pair: int
    ccd_algorithm: int
    # GPU core count, sizing the split narrow phase launches and scratch states
    gpu_cores: int


# Sizes of the scratch of Gilbert-Johnson-Keerthi (GJK) and of the expanding polytope algorithm (EPA): the vertices,
# faces and horizon edges of the polytope, the contacts of a pair and the vertices of a contact polygon
POLYTOPE_VERTS = Size("polytope_verts")
POLYTOPE_FACES = Size("polytope_faces")
POLYTOPE_HORIZONS = POLYTOPE_VERTS + 1
PAIR_CONTACTS = Size("pair_contacts")
POLYGON_VERTS = Size("polygon_verts")
# The arrays of the EPA, of the multi-contact detection and of their results, which the overlap test never reads
EPA = "epa"


class MDVertex(Data, kind=DataKind.SCRATCH):
    """A vertex of the Minkowski difference."""

    obj1: qd.Tensor = field((float, 3))
    obj2: qd.Tensor = field((float, 3))
    local_obj1: qd.Tensor = field((float, 3))
    local_obj2: qd.Tensor = field((float, 3))
    id1: qd.Tensor = field(int)
    id2: qd.Tensor = field(int)
    mink: qd.Tensor = field((float, 3))


class GJKSimplex(Data, kind=DataKind.SCRATCH, shape=(WORKERS,)):
    """The vertex count of the Gilbert-Johnson-Keerthi (GJK) simplex of each worker."""

    nverts: qd.Tensor = field(int)


class GJKSimplexBuffer(Data, kind=DataKind.SCRATCH, shape=(4, WORKERS)):
    """The normal and signed distance to the origin of the faces of each Gilbert-Johnson-Keerthi (GJK) simplex."""

    normal: qd.Tensor = field((float, 3))
    sdist: qd.Tensor = field(float)


class EPAPolytope(Data, kind=DataKind.SCRATCH, shape=(WORKERS,)):
    """The vertex, face and horizon counts of the expanding polytope algorithm (EPA) polytope of a worker."""

    nverts: qd.Tensor = field(int)
    nfaces: qd.Tensor = field(int)
    nfaces_map: qd.Tensor = field(int)
    horizon_nedges: qd.Tensor = field(int)
    horizon_w: qd.Tensor = field((float, 3))


class EPAPolytopeFace(Data, kind=DataKind.SCRATCH, shape=(POLYTOPE_FACES, WORKERS)):
    """A face of the expanding polytope algorithm (EPA) polytope: vertices, neighbors, normal, distance."""

    verts_idx: qd.Tensor = field((int, 3))
    adj_idx: qd.Tensor = field((int, 3))
    normal: qd.Tensor = field((float, 3))
    dist2: qd.Tensor = field(float)
    map_idx: qd.Tensor = field(int)
    visited: qd.Tensor = field(int)


class EPAPolytopeHorizonData(Data, kind=DataKind.SCRATCH):
    """A horizon edge of the expanding polytope algorithm (EPA) polytope, as a face and an edge."""

    face_idx: qd.Tensor = field(int)
    edge_idx: qd.Tensor = field(int)


class ContactFace(Data, kind=DataKind.SCRATCH, shape=(POLYGON_VERTS, WORKERS)):
    """The vertices and normals of the contact faces of the two geoms of a contact patch, per worker."""

    vert1: qd.Tensor = field((float, 3))
    vert2: qd.Tensor = field((float, 3))
    endverts: qd.Tensor = field((float, 3))
    normal1: qd.Tensor = field((float, 3))
    normal2: qd.Tensor = field((float, 3))
    id1: qd.Tensor = field(int)
    id2: qd.Tensor = field(int)


class ContactNormal(Data, kind=DataKind.SCRATCH, shape=(POLYGON_VERTS, WORKERS)):
    """The candidate face or edge normals of a geom in the multi-contact detection by contact patch, per worker."""

    endverts: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))
    id: qd.Tensor = field(int)


class ContactHalfspace(Data, kind=DataKind.SCRATCH, shape=(POLYGON_VERTS, WORKERS)):
    """The halfspaces clipping the polygon of a contact patch, as a normal and a distance, per worker."""

    normal: qd.Tensor = field((float, 3))
    dist: qd.Tensor = field(float)


class Witness(Data, kind=DataKind.SCRATCH, shape=(PAIR_CONTACTS, WORKERS)):
    """The witness points on the two geoms of each contact of a Gilbert-Johnson-Keerthi (GJK) detection, per worker."""

    point_obj1: qd.Tensor = field((float, 3))
    point_obj2: qd.Tensor = field((float, 3))


class GJKState(Data, kind=DataKind.SCRATCH):
    """
    The scratch of a detection by Gilbert-Johnson-Keerthi (GJK) and the expanding polytope algorithm (EPA), one column
    per thread or environment.

    The state of the overlap test alone leaves the arrays labeled EPA unused.
    """

    # GJK simplex
    support_mesh_prev_vertex_id: qd.Tensor = field(int, shape=(2, WORKERS))
    simplex_vertex: MDVertex = field(shape=(4, WORKERS))
    simplex_buffer: GJKSimplexBuffer
    simplex: GJKSimplex
    last_searched_simplex_vertex_id: qd.Tensor = field(int, shape=(WORKERS,))
    simplex_vertex_intersect: MDVertex = field(shape=(4, WORKERS))
    simplex_buffer_intersect: GJKSimplexBuffer
    nsimplex: qd.Tensor = field(int, shape=(WORKERS,))
    # EPA polytope
    polytope: EPAPolytope = field(labels=EPA)
    polytope_verts: MDVertex = field(shape=(POLYTOPE_VERTS, WORKERS), labels=EPA)
    polytope_faces: EPAPolytopeFace = field(labels=EPA)
    polytope_faces_map: qd.Tensor = field(int, shape=(POLYTOPE_FACES, WORKERS), labels=EPA)
    polytope_horizon_data: EPAPolytopeHorizonData = field(shape=(POLYTOPE_HORIZONS, WORKERS), labels=EPA)
    polytope_horizon_stack: EPAPolytopeHorizonData = field(shape=(3 * POLYTOPE_FACES, WORKERS), labels=EPA)
    # Multi-contact detection (MuJoCo compatibility)
    # FIXME: Leave the arrays specific to MuJoCo compatibility unused when it is disabled.
    contact_faces: ContactFace = field(labels=EPA)
    contact_normals: ContactNormal = field(labels=EPA)
    contact_halfspaces: ContactHalfspace = field(labels=EPA)
    contact_clipped_polygons: qd.Tensor = field((float, 3), shape=(2, POLYGON_VERTS, WORKERS), labels=EPA)
    multi_contact_flag: qd.Tensor = field(bool, shape=(WORKERS,))
    # Results
    witness: Witness
    n_witness: qd.Tensor = field(int, shape=(WORKERS,))
    n_contacts: qd.Tensor = field(int, shape=(WORKERS,), labels=EPA)
    contact_pos: qd.Tensor = field((float, 3), shape=(PAIR_CONTACTS, WORKERS), labels=EPA)
    normal: qd.Tensor = field((float, 3), shape=(PAIR_CONTACTS, WORKERS), labels=EPA)
    is_col: qd.Tensor = field(bool, shape=(WORKERS,), labels=EPA)
    penetration: qd.Tensor = field(float, shape=(WORKERS,), labels=EPA)
    distance: qd.Tensor = field(float, shape=(WORKERS,))
    # The EPA polytope face nearest to the origin, -1 when EPA did not run, from whose support simplex the multi-contact
    # perturbation recovers the exact unperturbed normal
    nearest_face: qd.Tensor = field(int, shape=(WORKERS,))
    # Differentiable contact detection
    diff_contact_input: DiffContactInput = field(shape=(PAIR_CONTACTS, WORKERS), labels=DIFF_CONTACTS)
    n_diff_contact_input: qd.Tensor = field(int, shape=(WORKERS,), labels=EPA)
    diff_penetration: qd.Tensor = field(float, shape=(PAIR_CONTACTS, WORKERS), labels=DIFF_CONTACTS)


class GJKInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """The tolerances and caps of Gilbert-Johnson-Keerthi (GJK) and the expanding polytope algorithm (EPA)."""

    max_contacts_per_pair: qd.Tensor = field(int)
    max_contact_polygon_verts: qd.Tensor = field(int)
    # Iteration caps of GJK and EPA
    gjk_max_iterations: qd.Tensor = field(int)
    epa_max_iterations: qd.Tensor = field(int)
    FLOAT_MIN: qd.Tensor = field(float)
    FLOAT_MIN_SQ: qd.Tensor = field(float)
    FLOAT_MAX: qd.Tensor = field(float)
    FLOAT_MAX_SQ: qd.Tensor = field(float)
    # Convergence tolerance of GJK and EPA on smooth geometries
    tolerance: qd.Tensor = field(float)
    # Distance below which two geoms collide
    collision_eps: qd.Tensor = field(float)
    # Degeneracy measure (colinearity, coplanarity) below which safe GJK discards a simplex, degenerate simplices being
    # the main source of EPA errors
    simplex_max_degeneracy_sq: qd.Tensor = field(float)
    polytope_max_faces: qd.Tensor = field(int)
    # Reprojection errors above which the projection of the origin onto a polytope face is rejected when computing the
    # witness points: the point rebuilt from its barycentric coordinates must match it. The relative error catches
    # degenerate faces, and the absolute one keeps the tiny faces of smooth geometries near convergence.
    polytope_max_rel_reprojection_error: qd.Tensor = field(float)
    polytope_max_abs_reprojection_error: qd.Tensor = field(float)
    # Alignment tolerances of the normals of a face-face and an edge-face parallel contact
    contact_face_tol: qd.Tensor = field(float)
    contact_edge_tol: qd.Tensor = field(float)
    # Minimum norm of a valid normal of the differentiable contact
    diff_contact_min_normal_norm: qd.Tensor = field(float)


# The states GJK runs on: one per env for the single-kernel narrowphase, and those of the passes of the split
# narrowphase, one per thread, which the collider opts in as it runs them
class GJKScratch(Data, kind=DataKind.SCRATCH):
    main: GJKState = field(opt_in=True)
    contact0: GJKState = field(opt_in=True)
    multicontact: GJKState = field(opt_in=True)


class GJKConfig(Config):
    """The compile-time configuration of the kernels of the Gilbert-Johnson-Keerthi (GJK) detection."""

    enable_contact_patch: bool


VERT_NEIGHBORS = Size("vert_neighbors")
# The samples of the height field of the terrain
TERRAIN_ROWS = Size("terrain_rows")
TERRAIN_COLS = Size("terrain_cols")


class ColliderInfo(Data, kind=DataKind.CONSTANT):
    """The model of the collision detection: vertex adjacency, valid pairs, terrain, contact caps, tolerances."""

    vert_neighbors: qd.Tensor = field(int, shape=(VERT_NEIGHBORS,))
    vert_neighbor_start: qd.Tensor = field(int, shape=(VERTS,))
    vert_n_neighbors: qd.Tensor = field(int, shape=(VERTS,))
    verts_spatial_grid: VertsSpatialGrid
    # Index of the pair of geoms (i_ga, i_gb) among the possible pairs, -1 for an invalid pair
    collision_pair_idx: qd.Tensor = field(int, shape=(GEOMS, GEOMS))
    # Index of the pair of links (i_la, i_lb) in valid_link_pairs, -1 for a pair without a valid geom pair
    link_pair_idx: qd.Tensor = field(int, shape=(LINKS, LINKS))
    # Lower and upper corners of the box of every link in its own frame, bounding the boxes of its geoms
    links_init_aabb_min: qd.Tensor = field((float, 3), shape=(LINKS,))
    links_init_aabb_max: qd.Tensor = field((float, 3), shape=(LINKS,))
    max_possible_pairs: qd.Tensor = field(int, shape=())
    max_collision_pairs: qd.Tensor = field(int, shape=())
    max_candidate_contacts: qd.Tensor = field(int, shape=())
    max_collision_pairs_broad: qd.Tensor = field(int, shape=())
    # Contacts per environment after pruning, which sizes the contact rows of the constraint solve. Below
    # max_candidate_contacts under contact pruning or the option 'max_contacts'.
    max_contacts: qd.Tensor = field(int, shape=())
    # Link pairs holding at least one valid geom pair, the link carrying most geoms first (see func_broad_phase_geoms)
    n_valid_link_pairs: qd.Tensor = field(int, shape=())
    valid_link_pairs: qd.Tensor = field((int, 2), shape=(VALID_LINK_PAIRS,))
    terrain_hf: qd.Tensor = field(float, shape=(TERRAIN_ROWS, TERRAIN_COLS))
    terrain_rc: qd.Tensor = field(int, shape=(2,))
    terrain_scale: qd.Tensor = field(float, shape=(2,))
    terrain_xyz_maxmin: qd.Tensor = field(float, shape=(6,))
    # Perturbation and tolerance of the multi-contact detection
    mc_perturbation: qd.Tensor = field(float, shape=())
    mc_tolerance: qd.Tensor = field(float, shape=())
    # Tolerances of the differentiable contact
    diff_pos_tolerance: qd.Tensor = field(float, shape=())
    diff_normal_tolerance: qd.Tensor = field(float, shape=())
    # Tolerances of the differentiable contact: the distance along its normal from a face to the support point, zero on
    # the boundary of the Minkowski difference, the excess of the distance from the origin to the face over the
    # expanding polytope algorithm (EPA) depth, and the excursion of the affine coordinates of the projection of the
    # origin outside [0, 1].
    # FIXME: Calibrate these values on real scenes.
    diff_contact_eps_boundary: qd.Tensor = field(float, shape=())
    diff_contact_eps_distance: qd.Tensor = field(float, shape=())
    diff_contact_eps_affine: qd.Tensor = field(float, shape=())
    # Link-pair contact pruning
    contact_pruning_tolerance: qd.Tensor = field(float, shape=())
    prune_deep_penetration_ratio: qd.Tensor = field(float, shape=())
