import dataclasses
import math
from collections.abc import Iterable, Iterator, Mapping
from enum import IntEnum
from typing import ClassVar, NamedTuple

import quadrants as qd
from typing_extensions import dataclass_transform  # Made it into standard lib from Python 3.12
import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch


def _tensor_backend():
    return qd.Backend.NDARRAY if gs.use_ndarray else qd.Backend.FIELD


def V(*args, **kwargs):
    return qd.tensor(*args, backend=_tensor_backend(), **kwargs)


def V_VEC(*args, **kwargs):
    return qd.Vector.tensor(*args, backend=_tensor_backend(), **kwargs)


def V_MAT(*args, **kwargs):
    return qd.Matrix.tensor(*args, backend=_tensor_backend(), **kwargs)


def maybe_shape(shape, is_on):
    return shape if is_on else ()


@dataclass_transform(eq_default=True, kw_only_default=False, frozen_default=True)
class AutoInitMeta(type):
    """Metaclass that generates __init__ from annotations, like a mutable dataclass."""

    def __new__(cls, name, bases, namespace):
        names = tuple(namespace["__annotations__"].keys())
        defaults = {k: namespace[k] for k in names if k in namespace}

        def __init__(self, *args, **kwargs):
            # Initialize assigned arguments from defaults
            assigned = defaults.copy()

            # Assign positional arguments
            if len(args) > len(names):
                raise TypeError(f"{name}() takes {len(names)} positional arguments but {len(args)} were given")
            for key, value in zip(names, args):
                assigned[key] = value

            # Assign keyword arguments
            for key, value in kwargs.items():
                if key not in names:
                    raise TypeError(f"{name}() got unexpected keyword argument '{key}'")
                if key in names[: len(args)]:
                    raise TypeError(f"{name}() got multiple values for argument '{key}'")
                assigned[key] = value

            # Check for missing arguments
            for key in names:
                if key not in assigned:
                    raise TypeError(f"{name}() missing required argument: '{key}'")

            # Set attributes
            for key, value in assigned.items():
                setattr(self, key, value)

        namespace["__init__"] = __init__

        return super().__new__(cls, name, bases, namespace)


def V_SCALAR_FROM(dtype, value):
    data = V(dtype=dtype, shape=())
    data.fill(value)
    return data


class DataKind(IntEnum):
    """The kind of every array or static config a solver yields (see 'Solver.data'), which checkpoints and trajectories
    select by.

    CONFIG is a static config of plain Python natives that keys the compiled kernels, recorded as provenance only.
    CONSTANT is an array the build computes from the scene description once. INFO is a model parameter the setters
    write, such as mass, friction, gains or limits. STATE is the reduced-space configuration, velocities,
    accelerations, control inputs, contacts and constraint solution. WARMSTART is a cache the next solve reads to
    reproduce the previous result bit-for-bit. Forward kinematics and dynamics recompute DERIVED from STATE. A step or
    a solve writes SCRATCH before it reads it.
    """

    CONFIG = 0
    CONSTANT = 1
    INFO = 2
    STATE = 3
    WARMSTART = 4
    DERIVED = 5
    SCRATCH = 6


# Metadata key under which 'of_kind' stores the kind of a field.
KIND = "kind"
# The kinds a checkpoint carries: INFO, STATE, WARMSTART and DERIVED. The static configs travel in a separate field of
# the checkpoint as provenance.
CHECKPOINT_KINDS = frozenset((DataKind.INFO, DataKind.STATE, DataKind.WARMSTART, DataKind.DERIVED))


def of_kind(kind: DataKind):
    """Declare a struct field whose kind differs from the default its struct states (see DataKind)."""
    return dataclasses.field(metadata={KIND: kind})


class DataItem(NamedTuple):
    """One array or static config a solver holds, under the dotted name of the slot and fields reaching it."""

    name: str
    value: object
    kind: DataKind


def iter_data(struct, name: str, kind: DataKind | None = None) -> Iterator[DataItem]:
    """Yield a DataItem for every array in a struct of arrays, named by 'name' plus field names joined with dots.

    Structs are the frozen dataclasses and named tuples of this module, nested to any depth. The class attribute
    'kind' gives the default kind of a struct's fields, and 'of_kind' on a field overrides it. An override on a
    struct-valued field applies to every array of that struct. An array carrying gradients yields them as one more item
    of its kind, named 'grad'. A static config yields one item of kind CONFIG, and anything else raises and names the
    offending slot.
    """
    if isinstance(struct, (qd.Tensor, qd.Field, qd.Ndarray)):
        if kind is None:
            gs.raise_exception(f"'{name}' holds an array that no struct or annotation gives a kind.")
        yield DataItem(name, struct, kind)
        if struct.has_grad():
            yield DataItem(f"{name}.grad", struct.grad, kind)
    elif isinstance(type(struct), AutoInitMeta):
        yield DataItem(name, struct, DataKind.CONFIG)
    elif dataclasses.is_dataclass(struct) or isinstance(struct, tuple):
        if dataclasses.is_dataclass(struct):
            values = vars(struct)
            members = [
                (field.name, values[field.name], field.metadata.get(KIND)) for field in dataclasses.fields(struct)
            ]
        else:
            members = [(member, value, None) for member, value in zip(struct._fields, struct)]
        for member, value, stated_kind in members:
            if stated_kind is None and kind is None and isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
                stated_kind = type(struct).kind
            yield from iter_data(value, f"{name}.{member}", kind if stated_kind is None else stated_kind)
    else:
        gs.raise_exception(f"'{name}' holds a {type(struct).__name__} where an array or a struct of arrays stands.")


def check_data(items: Iterable[DataItem], values: Mapping[str, np.ndarray | torch.Tensor]) -> list[tuple]:
    """Return the (array, value) pairs 'fill_data' writes, rejecting a record made for another scene.

    A value with the array's shape fills it, cast to its dtype, so a one-byte boolean fills an int32 flag. A floating
    value for an integer array, or the reverse, raises. 'maybe_shape' shapes an unused array as (), so an array with
    that shape on either side stays as built. Any other shape mismatch raises, since the record comes from a different
    build. A name on one side only raises, since the record comes from another build or version.
    """
    items = tuple(items)
    unshared = sorted({name for name, _, _ in items} ^ set(values))
    if unshared:
        gs.raise_exception(f"The record and this scene do not share {unshared}: another build or version wrote it.")
    writes = []
    for name, array, _ in items:
        value = values[name]
        element_shape = array.element_shape if isinstance(array, (qd.VectorTensor, qd.MatrixTensor)) else ()
        if tuple(array.shape) == () or tuple(value.shape) == element_shape:
            continue
        shape = (*array.shape, *element_shape)
        if tuple(value.shape) != shape:
            gs.raise_exception(f"'{name}' is shaped {shape} here and {value.shape} in the record: another build.")
        if isinstance(value, torch.Tensor):
            is_real = value.dtype.is_floating_point
        else:
            is_real = np.issubdtype(value.dtype, np.floating)
        if is_real != (array.dtype in qd.types.primitive_types.real_types):
            gs.raise_exception(f"'{name}' holds {array.dtype} values here where {value.dtype} ones were recorded.")
        writes.append((array, value))
    return writes


def fill_data(items: Iterable[DataItem], values: Mapping[str, np.ndarray | torch.Tensor]) -> None:
    """Write into each array the value recorded under its name, once every value passed 'check_data', so a record
    from another build leaves the scene as it was.

    Where zero-copy views exist the values go through them, so a device-resident record stays on the device. On Metal
    these writes stay on the torch stream, so the caller must call 'torch.mps.synchronize' once before the next kernel.
    """
    for array, value in check_data(items, values):
        if gs.use_zerocopy:
            view = qd_to_torch(array, copy=False)
            view.copy_(torch.as_tensor(value, device=view.device))
        else:
            array.from_numpy(value)


# =========================================== ErrorCode ===========================================


class ErrorCode(IntEnum):
    SUCCESS = 0b000000000000000000000000000000000
    OVERFLOW_COLLISION_PAIRS = 0b00000000000000000000000000000001
    OVERFLOW_CANDIDATE_CONTACTS = 0b00000000000000000000000000000010
    OVERFLOW_CONTACTS = 0b00000000000000000000000000000100
    OVERFLOW_HIBERNATION_ISLANDS = 0b00000000000000000000000000001000
    INVALID_CONTACT_NAN = 0b00000000000000000000000000010000
    INVALID_FORCE_NAN = 0b00000000000000000000000000100000
    INVALID_ACC_NAN = 0b00000000000000000000000001000000


# =========================================== RigidInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RigidInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    # *_bw: Cache for backward pass
    n_awake_dofs: qd.Tensor = of_kind(DataKind.STATE)
    awake_dofs: qd.Tensor = of_kind(DataKind.STATE)
    n_awake_entities: qd.Tensor = of_kind(DataKind.STATE)
    awake_entities: qd.Tensor = of_kind(DataKind.STATE)
    n_awake_links: qd.Tensor = of_kind(DataKind.STATE)
    awake_links: qd.Tensor = of_kind(DataKind.STATE)
    qpos0: qd.Tensor = of_kind(DataKind.INFO)
    qpos: qd.Tensor = of_kind(DataKind.STATE)
    qpos_next: qd.Tensor = of_kind(DataKind.SCRATCH)
    links_T: qd.Tensor = of_kind(DataKind.DERIVED)
    envs_offset: qd.Tensor
    geoms_init_AABB: qd.Tensor
    mass_mat: qd.Tensor = of_kind(DataKind.DERIVED)
    mass_mat_L: qd.Tensor = of_kind(DataKind.DERIVED)
    mass_mat_D_inv: qd.Tensor = of_kind(DataKind.DERIVED)
    mass_mat_tiled_scratch: qd.Tensor = of_kind(DataKind.SCRATCH)
    mass_mat_mask: qd.Tensor = of_kind(DataKind.STATE)
    # The kinematic trees, the links sharing a root link (links.root_idx). Tree i_t is rooted at trees_root_idx[i_t],
    # holds the trees_n_links[i_t] links of the span [trees_root_idx[i_t], trees_link_end[i_t]) whose root it is (the
    # span may interleave links of 0-dof entities created between attached ones, so consumers gate each link on the
    # root) and the contiguous dofs [trees_dof_start[i_t], trees_dof_start[i_t] + trees_n_dofs[i_t]). The trees
    # carrying a dof come first, in ascending dof order, so the island partition (see island.py) lists the dofs of the
    # trees it groups in ascending order; the dof-less trees (fixed bodies) follow in root order and join no island.
    # links_tree_idx maps every link to its tree.
    trees_root_idx: qd.Tensor
    trees_link_end: qd.Tensor
    trees_n_links: qd.Tensor
    trees_dof_start: qd.Tensor
    trees_n_dofs: qd.Tensor
    links_tree_idx: qd.Tensor
    # Per-DOF bounds of the mass block the DOF belongs to: the DOFs of its branch rooted where the fixed structure
    # ends (deeper branches stay mass-coupled to their chain and belong to the enclosing block), merged across
    # entities and kept contiguous by attach(). The assemble/factor/solve restrict to these bounds.
    dofs_mass_block_start: qd.Tensor
    dofs_mass_block_end: qd.Tensor
    # DOF range spanned by the mass blocks rooted in each entity: a leading run merged into an earlier-rooted block is
    # excluded, and the last rooted block may extend into a merged child (empty range for a fully-merged child). Lets
    # the per-entity assemble/factor/solve iterate their blocks as one flat, autodiff-compatible loop over DOFs.
    entities_mass_block_dof_start: qd.Tensor
    entities_mass_block_dof_end: qd.Tensor
    # Mask of the (dof, dof) pairs the mass matrix couples: a dof with its ancestors along the kinematic chain and
    # the dofs of its own link, within its mass block.
    mass_parent_mask: qd.Tensor
    meaninertia: qd.Tensor = of_kind(DataKind.INFO)
    gravity: qd.Tensor = of_kind(DataKind.INFO)
    # Runtime constants
    substep_dt: qd.Tensor
    iterations: qd.Tensor
    tolerance: qd.Tensor
    ls_iterations: qd.Tensor
    ls_tolerance: qd.Tensor
    noslip_iterations: qd.Tensor
    noslip_tolerance: qd.Tensor
    impratio: qd.Tensor
    n_equalities: qd.Tensor
    n_candidate_equalities: qd.Tensor
    hibernation_thresh_vel: qd.Tensor
    EPS: qd.Tensor


def get_rigid_info(solver, kinematic_only):
    _B = solver._B

    mass_mat_shape = (solver.n_dofs_, solver.n_dofs_, _B)
    if math.prod(mass_mat_shape) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Mass matrix shape (n_dofs={solver.n_dofs_}, n_dofs={solver.n_dofs_}, n_envs={_B}) is too large."
        )
    requires_grad = solver._requires_grad

    # Batch-first scratch for the register-tiled mass factor (qd.simt tile ops are batch-first, so the factorization
    # cannot run in place on the batch-last mass_mat_L). Allocated only when that path is enabled, with the constraint
    # Hessian's shape so nt_H can alias it (get_constraint_state) instead of allocating a second buffer; the factor only
    # touches it before the constraint solve repopulates it in the same step. Empty otherwise.
    mass_mat_tiled_scratch_shape = ()
    if not kinematic_only and solver.rigid_config.enable_register_tiled_mass:
        mass_mat_tiled_scratch_shape = (_B, solver.n_dofs_, solver.n_dofs_)

    # Flip mass_mat from canonical (n_dofs(i_d1), n_dofs(i_d2), _B) -> physical (_B, n_dofs(i_d2), n_dofs(i_d1)) via
    # layout=(2, 1, 0): i_d1 becomes innermost / stride-1, which coalesces consumer kernels whose lanes stride i_d1
    # with a serial inner i_d2 loop. The trade-off is regression on writer-side kernels that pair with cooperative
    # rewrites to recover under the same enable_cooperative_constraint_kernels flag.
    #
    # mass_mat_L stays canonical. Its dominant consumer is a serial Cholesky-style back-substitution that is already
    # coalesced under (n_dofs, n_dofs, _B) with lanes varying i_b, so flipping L would regress that path more than
    # the corresponding writer-side win on the tiled factor_mass.
    mass_mat_layout = (
        (2, 1, 0) if not kinematic_only and solver.rigid_config.enable_cooperative_constraint_kernels else None
    )

    # FIXME: Add a better split between kinematic and Genesis
    if kinematic_only:
        return RigidInfo(
            envs_offset=V_VEC(3, dtype=gs.qd_float, shape=(_B,)),
            gravity=V_VEC(3, dtype=gs.qd_float, shape=()),
            meaninertia=V(dtype=gs.qd_float, shape=()),
            n_awake_dofs=V(dtype=gs.qd_int, shape=(_B,)),
            n_awake_entities=V(dtype=gs.qd_int, shape=(_B,)),
            n_awake_links=V(dtype=gs.qd_int, shape=(_B,)),
            awake_dofs=V(dtype=gs.qd_int, shape=(solver.n_dofs_, _B)),
            awake_entities=V(dtype=gs.qd_int, shape=(solver.n_entities_, _B)),
            awake_links=V(dtype=gs.qd_int, shape=(solver.n_links_, _B)),
            qpos0=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
            qpos=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
            qpos_next=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
            links_T=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(solver.n_links_,)),
            geoms_init_AABB=V_VEC(3, dtype=gs.qd_float, shape=()),
            mass_mat=V(dtype=gs.qd_float, shape=()),
            mass_mat_L=V(dtype=gs.qd_float, shape=()),
            mass_mat_D_inv=V(dtype=gs.qd_float, shape=()),
            mass_mat_tiled_scratch=V(dtype=gs.qd_float, shape=()),
            mass_mat_mask=V(dtype=gs.qd_bool, shape=()),
            trees_root_idx=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
            trees_link_end=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
            trees_n_links=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
            trees_dof_start=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
            trees_n_dofs=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
            links_tree_idx=V(dtype=gs.qd_int, shape=(solver.n_links_,)),
            dofs_mass_block_start=V(dtype=gs.qd_int, shape=()),
            dofs_mass_block_end=V(dtype=gs.qd_int, shape=()),
            entities_mass_block_dof_start=V(dtype=gs.qd_int, shape=()),
            entities_mass_block_dof_end=V(dtype=gs.qd_int, shape=()),
            mass_parent_mask=V(dtype=gs.qd_float, shape=()),
            substep_dt=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            ls_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            ls_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            noslip_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            noslip_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            impratio=V_SCALAR_FROM(dtype=gs.qd_float, value=1.0),
            n_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            n_candidate_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            hibernation_thresh_vel=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=gs.EPS),
        )

    return RigidInfo(
        envs_offset=V_VEC(3, dtype=gs.qd_float, shape=(_B,)),
        gravity=V_VEC(3, dtype=gs.qd_float, shape=(_B,)),
        meaninertia=V(dtype=gs.qd_float, shape=(_B,)),
        n_awake_dofs=V(dtype=gs.qd_int, shape=(_B,)),
        n_awake_entities=V(dtype=gs.qd_int, shape=(_B,)),
        n_awake_links=V(dtype=gs.qd_int, shape=(_B,)),
        awake_dofs=V(dtype=gs.qd_int, shape=(solver.n_dofs_, _B)),
        awake_entities=V(dtype=gs.qd_int, shape=(solver.n_entities_, _B)),
        awake_links=V(dtype=gs.qd_int, shape=(solver.n_links_, _B)),
        qpos0=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
        qpos=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B), needs_grad=requires_grad),
        qpos_next=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B), needs_grad=requires_grad),
        links_T=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(solver.n_links_,)),
        geoms_init_AABB=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_geoms_, 8)),
        mass_mat=V(dtype=gs.qd_float, shape=mass_mat_shape, layout=mass_mat_layout, needs_grad=requires_grad),
        mass_mat_L=V(dtype=gs.qd_float, shape=mass_mat_shape, needs_grad=requires_grad),
        mass_mat_D_inv=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), needs_grad=requires_grad),
        mass_mat_tiled_scratch=V(dtype=gs.qd_float, shape=mass_mat_tiled_scratch_shape),
        mass_mat_mask=V(dtype=gs.qd_bool, shape=(solver.n_entities_, _B)),
        trees_root_idx=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        trees_link_end=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        trees_n_links=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        trees_dof_start=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        trees_n_dofs=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        links_tree_idx=V(dtype=gs.qd_int, shape=(solver.n_links_,)),
        dofs_mass_block_start=V(dtype=gs.qd_int, shape=(solver.n_dofs_,)),
        dofs_mass_block_end=V(dtype=gs.qd_int, shape=(solver.n_dofs_,)),
        entities_mass_block_dof_start=V(dtype=gs.qd_int, shape=(solver.n_entities_,)),
        entities_mass_block_dof_end=V(dtype=gs.qd_int, shape=(solver.n_entities_,)),
        mass_parent_mask=V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver.n_dofs_)),
        substep_dt=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._substep_dt),
        iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._options.iterations),
        tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.tolerance),
        ls_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._options.ls_iterations),
        ls_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.ls_tolerance),
        noslip_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._options.noslip_iterations),
        noslip_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.noslip_tolerance),
        impratio=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.impratio),
        n_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._n_equalities),
        n_candidate_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=solver.n_candidate_equalities_),
        hibernation_thresh_vel=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._hibernation_thresh_vel),
        EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=gs.EPS),
    )


# =========================================== Constraint ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class IslandSlices:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Per-(island, env) slices into a packed id array: island i_island's items are id[start[i_island, i_b] : start +
    # n[i_island, i_b]]. curr is the write cursor the partition build advances while filling each slice; once built,
    # curr == start + n. Indexed [n_trees, B] since an island holds at least one tree (see trees_root_idx in RigidInfo).
    curr: qd.Tensor
    n: qd.Tensor
    start: qd.Tensor


def get_slices(solver, layout, is_active=True):
    _B = solver._B
    n_trees = solver.n_trees_

    return IslandSlices(
        curr=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), is_active), layout=layout if is_active else None),
        n=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), is_active), layout=layout if is_active else None),
        start=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), is_active), layout=layout if is_active else None),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class IslandState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Partition of the dof-carrying kinematic trees (see trees_root_idx in RigidInfo) into islands, the connected
    # components of the trees under the contact, equality and hibernation couplings, rebuilt every step by island.py.
    # trees_parent_idx is the union-find forest over the trees, trees_island_idx the island of each tree and
    # links_island_idx that of each link, -1 for a dof-less tree and its links. link_slices maps an island to its slice
    # of link_id, dof_slices to its slice of dof_id (island-local dof -> global dof, ascending unless the CPU skyline
    # path reorders it by contact adjacency).
    trees_parent_idx: qd.Tensor
    trees_island_idx: qd.Tensor
    links_island_idx: qd.Tensor
    n_islands: qd.Tensor
    link_slices: IslandSlices
    link_id: qd.Tensor
    dof_slices: IslandSlices
    dof_id: qd.Tensor
    # Inverse of dof_id: dof_local_pos[d] is the island-local position of global dof d
    dof_local_pos: qd.Tensor
    dofs_island_idx: qd.Tensor
    # First dof of an island whose dof list holds consecutive dofs in ascending order, -1 otherwise (see func_list_item
    # in linesearch.py).
    dof_range_start: qd.Tensor
    # Skyline envelope of the CPU per-island path: dof_env_start_local[dof_slices.start[i] + ld] is the smallest
    # island-local column structurally nonzero in local row ld of island i's Hessian block, 0 (dense) when uncomputed.
    # dof_env_col_end is its transpose, the largest local row whose envelope reaches column ld, which bounds the
    # column-oriented factor and solve sweeps to the band: a tall island of a few hundred dofs otherwise tests every
    # row below the diagonal against its envelope, an O(n^2) scan per sweep. It has no safe uncomputed default (0
    # truncates), so only the CPU per-island path reads it.
    dof_env_start_local: qd.Tensor
    dof_env_col_end: qd.Tensor
    # Per-island contact lists of the CPU per-island skyline reorder (see rcm_tree_pos below, empty otherwise).
    # contact_id also carries the input order of the cooperative contact sort (see func_sort_contacts_coop in
    # island.py).
    contact_slices: IslandSlices
    contact_id: qd.Tensor
    constraint_slices: IslandSlices
    constraint_id: qd.Tensor
    # Island of each constraint, -1 for one touching no island
    constraint_island_idx: qd.Tensor
    # Hibernation (empty unless use_hibernation): is_hibernated marks an island whose every link is asleep, and
    # hibernated_next_link is the per-link daisy chain the partition build re-unions before labeling, which keeps a
    # sleeping component, whose bodies generate no live contacts, one island across steps.
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)
    hibernated_next_link: qd.Tensor = of_kind(DataKind.STATE)
    # (env, island) work-lists of the cooperative per-island factor+solve, one per island size class (see
    # island_tile_caps): class c holds factor_worklist_size[c] items in its own region of the two index lists, of
    # n_trees * _B slots each, in the order of the atomic reservation of the partition pass.
    factor_worklist_i_b: qd.Tensor
    factor_worklist_i_island: qd.Tensor
    factor_worklist_size: qd.Tensor
    # Scratch of the reverse Cuthill-McKee tree reorder of the CPU per-island skyline path: a tree's island-local slot,
    # contact degree and ordered flag, and the resulting visit order. Distinct buffers, since quadrants assumes
    # distinct kernel arguments never alias.
    rcm_tree_pos: qd.Tensor
    rcm_tree_degree: qd.Tensor
    rcm_tree_is_ordered: qd.Tensor
    rcm_tree_order: qd.Tensor
    # Per-island Newton / CG iteration state (see linesearch.py), indexed [i_island, i_b]: inertia is the trace of the
    # mass matrix over the island's dofs, the scale of its convergence tests, and improved says whether the island
    # still iterates, every pass of an iteration being confined to the islands that do. ls_improvement carries
    # -cost(alpha_accepted) in shifted form, cost(alpha) - cost(0), which stays resolvable in float32 near convergence.
    inertia: qd.Tensor
    improved: qd.Tensor
    ls_improvement: qd.Tensor


def get_island_state(solver, collider):
    _B = solver._B
    n_links = max(solver.n_links, 1)
    n_dofs = max(solver.n_dofs, 1)
    n_trees = solver.n_trees_
    # The per-island Hessian is assembled and factored in place in constraint_state.nt_H (block-diagonal), so
    # island_state itself holds only the partition maps and the per-island iteration state.
    rcm_active = solver.rigid_config.sparse_solve
    coop_active = solver.rigid_config.enable_cooperative_constraint_kernels
    # Batch-first under the cooperative kernels, whose block serves one env: the lanes then read consecutive items of
    # their env from consecutive addresses (see the constraint-state layouts in get_constraint_state).
    island_layout = (1, 0) if solver.rigid_config.constraint_layout_batch_first else None
    n_classes = len(island_tile_caps(solver.rigid_config))
    max_candidate_contacts = max(collider.collider_info.max_candidate_contacts[None], 1)
    # Safe upper bound on active constraints, mirroring ConstraintSolver.len_constraints: rows_per_contact per
    # contact + joint-limit/frictionloss (<= n_dofs each) + equality rows (<= 6 each). The equality term must use the
    # candidate count (model equalities plus the dynamic-weld budget), not just the model equalities, otherwise
    # constraint_id is undersized once dynamic welds are added and the per-island grouping writes out of bounds.
    n_constraints_max = max(
        max_candidate_contacts * solver.rigid_config.rows_per_contact
        + 2 * n_dofs
        + max(solver.n_candidate_equalities_, 1) * 6,
        1,
    )
    return IslandState(
        trees_parent_idx=V(dtype=gs.qd_int, shape=(n_trees, _B), layout=island_layout),
        trees_island_idx=V(dtype=gs.qd_int, shape=(n_trees, _B), layout=island_layout),
        links_island_idx=V(dtype=gs.qd_int, shape=(n_links, _B), layout=island_layout),
        n_islands=V(dtype=gs.qd_int, shape=(_B,)),
        link_slices=get_slices(solver, island_layout, solver._use_hibernation),
        link_id=V(
            dtype=gs.qd_int,
            shape=maybe_shape((n_links, _B), solver._use_hibernation),
            layout=island_layout if solver._use_hibernation else None,
        ),
        dof_slices=get_slices(solver, island_layout),
        dof_id=V(dtype=gs.qd_int, shape=(n_dofs, _B), layout=island_layout),
        dof_local_pos=V(dtype=gs.qd_int, shape=(n_dofs, _B), layout=island_layout),
        dofs_island_idx=V(dtype=gs.qd_int, shape=(n_dofs, _B), layout=island_layout),
        dof_range_start=V(dtype=gs.qd_int, shape=(n_trees, _B), layout=island_layout),
        dof_env_start_local=V(dtype=gs.qd_int, shape=(n_dofs, _B), layout=island_layout),
        dof_env_col_end=V(dtype=gs.qd_int, shape=(n_dofs, _B), layout=island_layout),
        contact_slices=get_slices(solver, island_layout, rcm_active),
        contact_id=V(
            dtype=gs.qd_int,
            shape=maybe_shape((max_candidate_contacts, _B), rcm_active or coop_active),
            layout=island_layout if rcm_active or coop_active else None,
        ),
        constraint_slices=get_slices(solver, island_layout),
        constraint_id=V(dtype=gs.qd_int, shape=(n_constraints_max, _B), layout=island_layout),
        constraint_island_idx=V(dtype=gs.qd_int, shape=(n_constraints_max, _B), layout=island_layout),
        is_hibernated=V(dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), solver._use_hibernation)),
        hibernated_next_link=V(dtype=gs.qd_int, shape=maybe_shape((n_links, _B), solver._use_hibernation)),
        factor_worklist_i_b=V(dtype=gs.qd_int, shape=maybe_shape((n_classes * n_trees * _B,), coop_active)),
        factor_worklist_i_island=V(dtype=gs.qd_int, shape=maybe_shape((n_classes * n_trees * _B,), coop_active)),
        factor_worklist_size=V(dtype=gs.qd_int, shape=maybe_shape((n_classes,), coop_active)),
        rcm_tree_pos=V(
            dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), rcm_active), layout=island_layout if rcm_active else None
        ),
        rcm_tree_degree=V(
            dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), rcm_active), layout=island_layout if rcm_active else None
        ),
        rcm_tree_is_ordered=V(
            dtype=gs.qd_bool, shape=maybe_shape((n_trees, _B), rcm_active), layout=island_layout if rcm_active else None
        ),
        rcm_tree_order=V(
            dtype=gs.qd_int, shape=maybe_shape((n_trees, _B), rcm_active), layout=island_layout if rcm_active else None
        ),
        inertia=V(dtype=gs.qd_float, shape=(n_trees, _B), layout=island_layout),
        improved=V(dtype=gs.qd_bool, shape=(n_trees, _B), layout=island_layout),
        ls_improvement=V(dtype=gs.qd_float, shape=(n_trees, _B), layout=island_layout),
    )


# =========================================== IKState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class IKState:
    """Damped-least-squares inverse-kinematics scratch, one column per parallel solve.

    A column is an environment for the entity API or a candidate restart for the planner; error_dim = 6 * n_targets
    stacks the per-target pose residuals.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Single-target spatial Jacobian (6 x n_dofs), rebuilt per target link and copied into the stacked block.
    jacobian: qd.Tensor
    # Stacked multi-target Jacobian and its transpose, feeding the damped normal-equations solve.
    jacobian_stacked: qd.Tensor
    jacobian_stacked_t: qd.Tensor
    # Damped normal matrix J J^T + damping^2 I, its lower / upper LU factors and forward-substitution scratch, and
    # its inverse.
    mat: qd.Tensor
    lu_lower: qd.Tensor
    lu_upper: qd.Tensor
    lu_y: qd.Tensor
    inv: qd.Tensor
    # Stacked pose residual, the best residual seen across restarts, and inv @ residual.
    err_pose: qd.Tensor
    err_pose_best: qd.Tensor
    vec: qd.Tensor
    # Joint-space step and the best configuration found across restarts.
    delta_qpos: qd.Tensor
    qpos_best: qd.Tensor


def get_ik_state(n_qs, n_dofs, error_dim, n_cols):
    return IKState(
        jacobian=V(dtype=gs.qd_float, shape=(6, n_dofs, n_cols)),
        jacobian_stacked=V(dtype=gs.qd_float, shape=(error_dim, n_dofs, n_cols)),
        jacobian_stacked_t=V(dtype=gs.qd_float, shape=(n_dofs, error_dim, n_cols)),
        mat=V(dtype=gs.qd_float, shape=(error_dim, error_dim, n_cols)),
        lu_lower=V(dtype=gs.qd_float, shape=(error_dim, error_dim, n_cols)),
        lu_upper=V(dtype=gs.qd_float, shape=(error_dim, error_dim, n_cols)),
        lu_y=V(dtype=gs.qd_float, shape=(error_dim, error_dim, n_cols)),
        inv=V(dtype=gs.qd_float, shape=(error_dim, error_dim, n_cols)),
        err_pose=V(dtype=gs.qd_float, shape=(error_dim, n_cols)),
        err_pose_best=V(dtype=gs.qd_float, shape=(error_dim, n_cols)),
        vec=V(dtype=gs.qd_float, shape=(error_dim, n_cols)),
        delta_qpos=V(dtype=gs.qd_float, shape=(n_dofs, n_cols)),
        qpos_best=V(dtype=gs.qd_float, shape=(n_qs, n_cols)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class IKScratchFK:
    """Per-column forward-kinematics scratch for the entity inverse-kinematics solve.

    Holds the working configuration and the link / joint frames it produces, so each Gauss-Newton step evaluates
    and integrates poses on caller-owned buffers without mutating live solver state. A column is an environment.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Entity-local working configuration iterated by the solve.
    qpos: qd.Tensor
    # Link poses and joint frames placed by func_forward_kinematics_scratch, feeding the error and the Jacobian.
    links_pos: qd.Tensor
    links_quat: qd.Tensor
    joints_xanchor: qd.Tensor
    joints_xaxis: qd.Tensor


def get_ik_scratch_fk(n_qs, n_links, n_joints, n_cols):
    return IKScratchFK(
        qpos=V(dtype=gs.qd_float, shape=(n_qs, n_cols)),
        links_pos=V(dtype=gs.qd_vec3, shape=(n_links, n_cols)),
        links_quat=V(dtype=gs.qd_vec4, shape=(n_links, n_cols)),
        joints_xanchor=V(dtype=gs.qd_vec3, shape=(n_joints, n_cols)),
        joints_xaxis=V(dtype=gs.qd_vec3, shape=(n_joints, n_cols)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class IKTargets:
    """Inverse-kinematics targets and DOF selection for one solve batch, one column per parallel solve.

    A column is an environment for the entity API or a candidate restart for the planner. Positions and
    orientations are vec-typed so the solver reads a target pose per (link, column) directly; the host populates
    every field before the solve.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Global indices of the target links, the effective DOFs to move, and the columns (solver envs) to solve.
    links_idx: qd.Tensor
    dofs_idx: qd.Tensor
    envs_idx: qd.Tensor
    # Target pose per (link, column) and the link-local point whose pose is driven, in the world frame.
    pos: qd.Tensor
    quat: qd.Tensor
    local_point: qd.Tensor
    # Optional custom initial configuration per (column, q); used only when custom_init_qpos is set.
    init_qpos: qd.Tensor
    # Which position / rotation axes to solve (shared across links), and per-link position / rotation enables.
    pos_mask: qd.Tensor
    rot_mask: qd.Tensor
    link_pos_mask: qd.Tensor
    link_rot_mask: qd.Tensor


def get_ik_targets(n_links, n_dofs, n_qs, n_cols):
    return IKTargets(
        links_idx=V(dtype=gs.qd_int, shape=(n_links,)),
        dofs_idx=V(dtype=gs.qd_int, shape=(n_dofs,)),
        envs_idx=V(dtype=gs.qd_int, shape=(n_cols,)),
        pos=V_VEC(3, dtype=gs.qd_float, shape=(n_links, n_cols)),
        quat=V_VEC(4, dtype=gs.qd_float, shape=(n_links, n_cols)),
        local_point=V_VEC(3, dtype=gs.qd_float, shape=(n_links,)),
        init_qpos=V(dtype=gs.qd_float, shape=(n_cols, n_qs)),
        pos_mask=V(dtype=gs.qd_float, shape=(3,)),
        rot_mask=V(dtype=gs.qd_float, shape=(3,)),
        link_pos_mask=V(dtype=gs.qd_int, shape=(n_links,)),
        link_rot_mask=V(dtype=gs.qd_int, shape=(n_links,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class KinematicsScratch:
    """Buffers backing the kinematics queries of every entity of one solver.

    A column is an environment. The Jacobian spans the widest entity, and the configuration cache spans the solver
    configuration so the global q indices callers already work in index it directly.
    """

    kind: ClassVar[DataKind] = DataKind.SCRATCH

    jacobian: qd.Tensor
    qpos_cache: qd.Tensor


def get_kinematics_scratch(solver):
    n_dofs = max((entity.n_dofs for entity in solver.entities), default=0)
    is_active = n_dofs > 0

    return KinematicsScratch(
        jacobian=V(dtype=gs.qd_float, shape=maybe_shape((6, n_dofs, solver._B), is_active)),
        qpos_cache=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_qs, solver._B), is_active)),
    )


class WeightScratch(NamedTuple):
    """Buffers used to compute the constraint inverse weights of one solver.

    One weight needs one row of a link Jacobian and the mass-matrix solve against that row, which is what these two
    hold.
    """

    kind = DataKind.SCRATCH
    jac_row: qd.Tensor
    solve_out: qd.Tensor


def get_weight_scratch(solver):
    return WeightScratch(
        jac_row=V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver._B)),
        solve_out=V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver._B)),
    )


class IKScratch(NamedTuple):
    """The three solve-time buffers an inverse-kinematics call needs, shared by every entity of one solver.

    Sized by the largest entity the solver holds and by the target cap, so each solve writes the leading rows its own
    entity spans; a column is an environment.
    """

    kind = DataKind.SCRATCH
    state: IKState
    fk: IKScratchFK
    targets: IKTargets


def get_ik_scratch(solver):
    n_qs = max(entity.n_qs for entity in solver.entities)
    n_dofs = max(entity.n_dofs for entity in solver.entities)
    n_links = max(entity.n_links for entity in solver.entities)
    n_joints = max(entity.n_joints for entity in solver.entities)
    n_tgts = solver._options.IK_max_targets

    return IKScratch(
        state=get_ik_state(n_qs, n_dofs, 6 * n_tgts, solver._B),
        fk=get_ik_scratch_fk(n_qs, n_links, n_joints, solver._B),
        targets=get_ik_targets(n_tgts, n_dofs, n_qs, solver._B),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ConstraintState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Union-find partition of links into contact islands, read by the per-island Newton solve and hibernation.
    island: IslandState
    is_warmstart: qd.Tensor = of_kind(DataKind.WARMSTART)
    n_constraints: qd.Tensor = of_kind(DataKind.STATE)
    qd_n_equalities: qd.Tensor = of_kind(DataKind.STATE)
    jac: qd.Tensor
    diag: qd.Tensor
    aref: qd.Tensor
    jac_dofs_idx: qd.Tensor
    jac_n_dofs: qd.Tensor
    n_constraints_equality: qd.Tensor = of_kind(DataKind.STATE)
    n_constraints_frictionloss: qd.Tensor = of_kind(DataKind.STATE)
    # Number of elliptic-cone contact rows (rows_per_contact per contact), laid out contiguously at the start of the
    # collision segment. Zero for the pyramidal cone. The cone rows occupy
    # [ne + n_frictionloss, ne + n_frictionloss + n_cone).
    n_constraints_cone: qd.Tensor = of_kind(DataKind.STATE)
    improved: qd.Tensor
    Jaref: qd.Tensor
    Ma: qd.Tensor
    Ma_ws: qd.Tensor
    grad: qd.Tensor
    Mgrad: qd.Tensor
    search: qd.Tensor
    # Previous-iteration cone-row residuals, kept only for the elliptic cone so the incremental factor can downdate the
    # prior coupled cone block (Jaref is overwritten by the linesearch apply). Empty for the pyramidal cone.
    cone_prev_jaref: qd.Tensor
    efc_D: qd.Tensor
    # Frictionloss rows store their friction loss; elliptic-cone head (normal) rows reuse the field to carry the
    # contact sliding friction coefficient read by the cone solver, and with torsional friction the spin row carries
    # the torsional coefficient the same way (the tangent rows hold 0).
    efc_frictionloss: qd.Tensor
    efc_force: qd.Tensor = of_kind(DataKind.STATE)
    active: qd.Tensor = of_kind(DataKind.STATE)
    prev_active: qd.Tensor
    qfrc_constraint: qd.Tensor = of_kind(DataKind.STATE)
    qacc: qd.Tensor = of_kind(DataKind.STATE)
    qacc_ws: qd.Tensor = of_kind(DataKind.WARMSTART)
    qacc_prev: qd.Tensor
    cost_ws: qd.Tensor
    cost: qd.Tensor
    mv: qd.Tensor
    jv: qd.Tensor
    # Optional CG fields
    cg_prev_grad: qd.Tensor
    cg_prev_Mgrad: qd.Tensor
    # Optional Newton fields
    # Hessian matrix of the optimization problem as a dense 2D tensor.
    # Note that only the lower triangular part is updated for efficiency because this matrix is symmetric by definition.
    # As a result, the values of the strictly upper triangular part is undefined - except under
    # enable_cone_free_hessian_reuse, where each used lower-triangle slot's mirror persists the cone-free assembled
    # Hessian (M + J^T D J of the active rows only), maintained across a step's Newton iterations by signed flip
    # scatters; its diagonal lives in nt_H_cone_free_diag since the lower diagonal belongs to the factor. A rebuild
    # restores the mirror into the lower triangle and bakes the current cone blocks on top, skipping the full
    # J^T D J reassembly.
    # In practice, this variable is re-purposed to store the Cholesky factor L st H = L @ L.T to spare memory resources.
    # TODO: Optimize storage to only allocate memory half of the Hessian matrix to sparse memory resources.
    nt_H: qd.Tensor
    # Per-DOF Jacobi scale s_i = 1/sqrt(diag(M + J^T D J)_i), at natural DOF id, refreshed ahead of every direct
    # rebuild (func_jacobi_scale; 1 on an empty diagonal). Assembly writes H already scaled, nt_H holds L of S H S,
    # update vectors are scaled at construction, and the batch solve wraps grad/Mgrad, so Mgrad = H^-1 grad exactly. A unit diagonal keeps the Hessian's mixed-unit spread within float32's
    # conditioning capability at any geometry scale and makes the bare-EPS pivot floor scale-relative. Only meaningful
    # with enable_jacobi_equilibration.
    nt_jacobi: qd.Tensor
    # Diagonal of the persisted cone-free Hessian packed in nt_H's mirror slots (see nt_H). Only meaningful with
    # enable_cone_free_hessian_reuse.
    nt_H_cone_free_diag: qd.Tensor
    nt_vec: qd.Tensor
    # Compacted list of constraints whose active state changed, used by incremental Cholesky update
    # to reduce GPU thread divergence by iterating only over constraints that need processing.
    incr_changed_idx: qd.Tensor
    incr_n_changed: qd.Tensor
    # Backward gradients
    dL_dqacc: qd.Tensor
    dL_dM: qd.Tensor
    dL_djac: qd.Tensor
    dL_daref: qd.Tensor
    dL_defc_D: qd.Tensor
    dL_dforce: qd.Tensor
    # Backward buffers for linear system solver
    bw_u: qd.Tensor
    bw_r: qd.Tensor
    bw_p: qd.Tensor
    bw_Ap: qd.Tensor
    bw_Ju: qd.Tensor
    bw_y: qd.Tensor
    bw_w: qd.Tensor
    # Timers for profiling
    timers: qd.Tensor
    # Solver loop iteration counter (0-indexed, increments each iteration in the graph loop)
    solver_iter_counter: qd.Tensor
    # Always ndarray (not field): graph.do_while requires the same physical ndarray on every call.
    graph_counter: qd.types.ndarray()
    early_exit_flag: qd.Tensor


def get_constraint_state(constraint_solver, solver, collider):
    _B = solver._B
    len_constraints_ = constraint_solver.len_constraints_

    # The constraint-state layout flips (con / jac / dof_vec) gate on constraint_layout_batch_first; jac additionally
    # picks its batch-first permutation from enable_cooperative_constraint_kernels. See the per-flip docs below.
    cooperative = solver.rigid_config.enable_cooperative_constraint_kernels
    batch_first = solver.rigid_config.constraint_layout_batch_first
    # Serialized execution visits envs in the outermost loop of every constraint kernel, so the hot per-env rows of the
    # constraint tensors must be contiguous in memory: batch-first physical layout. With the canonical batch-last layout
    # every scalar access strides by n_envs, wasting a full cache line per element, which makes the constraint solver
    # DRAM-bound once the combined per-env working sets exceed the CPU caches and batched stepping scales super-linearly
    # with n_envs.
    serialized = solver.rigid_config.para_level < gs.PARA_LEVEL.ALL
    # Layout-flippable constraint-state tensors (Jaref, jv, efc_D, efc_frictionloss, diag, active) keep their
    # canonical (len_constraints_, _B) shape; the static config flag picks the physical layout via ``layout=(1, 0)``.
    # Cooperative kernels read the same flag at compile time to switch between serial and warp-cooperative reductions.
    # The remaining (len_constraints_,) tensors outside the GPU cooperative flip set follow only the serialized flip.
    con_layout = (1, 0) if batch_first else None
    serial_layout = (1, 0) if serialized else None
    # The CPU incremental factor maintains the elliptic cone by a per-iteration rank-3 update reading the previous cone
    # residuals, so the residual cache is allocated for the CPU elliptic case.
    is_cone_incremental = solver.rigid_config.enable_elliptic_friction and solver.rigid_config.backend == gs.cpu
    # The 3D Jacobian and its sparse-column-index sibling extend the flip: canonical (len_constraints_, n_dofs_, _B) ->
    # physical (_B, n_dofs_, len_constraints_) via layout=(2, 1, 0). This makes cooperative-warp-per-env access (lanes
    # stride i_c) coalesced for the hot p0 J@search, hessian_direct_tiled, and patch_hessian_delta kernels.
    # Serialized execution instead wants physical (_B, len_constraints_, n_dofs_) via layout=(2, 0, 1): constraint rows
    # are read and written dof-by-dof for a fixed env.
    jac_layout = (2, 1, 0) if cooperative else (2, 0, 1) if serialized else None
    # DOF-vec family flip: canonical (n_dofs_, _B) -> physical (_B, n_dofs_) via layout=(1, 0). Adjacent-lane reads
    # striding i_d in cooperative kernels become stride-1; the regression on 1T-per-(i_d, i_b) writers is patched on
    # a per-consumer basis under the same enable_cooperative_constraint_kernels flag.
    dof_vec_layout = (1, 0) if batch_first else None
    # Rank-1 working vectors of the incremental Cholesky update, flattened slot-minor as [i_d * n_slots + i_u]: one
    # slot per fused update on the CPU per-island path (func_rank_batch_update_island), a single slot elsewhere
    # (indexing then reduces to [i_d]). Flat 2D so the buffer keeps the DOF-vec rank and layout on every backend.
    nt_vec_n_slots = solver.rigid_config.hessian_rank_update_batch if constraint_solver.sparse_solve else 1

    jac_shape = (len_constraints_, solver.n_dofs_, _B)
    # The sparse-Jacobian representation is always active, so its index buffers are always allocated. The skyline DOF
    # permutation/envelope buffers stay gated on sparse_solve (CPU-only skyline Cholesky).
    jac_dofs_idx_shape = jac_shape
    jac_n_dofs_shape = (len_constraints_, _B)

    if math.prod(jac_shape) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Jacobian shape (n_constraints={len_constraints_}, n_dofs={solver.n_dofs_}, n_envs={_B}) is too large."
        )

    # /!\ Changing allocation order of these tensors may reduce runtime speed by >10%  /!\
    return ConstraintState(
        n_constraints=V(dtype=gs.qd_int, shape=(_B,)),
        qd_n_equalities=V(dtype=gs.qd_int, shape=(_B,)),
        n_constraints_equality=V(dtype=gs.qd_int, shape=(_B,)),
        n_constraints_frictionloss=V(dtype=gs.qd_int, shape=(_B,)),
        n_constraints_cone=V(dtype=gs.qd_int, shape=(_B,)),
        is_warmstart=V(dtype=gs.qd_bool, shape=(_B,)),
        improved=V(dtype=gs.qd_bool, shape=(_B,)),
        cost_ws=V(dtype=gs.qd_float, shape=(_B,)),
        cost=V(dtype=gs.qd_float, shape=(_B,)),
        Ma=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        Ma_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        grad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        Mgrad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        cone_prev_jaref=V(
            dtype=gs.qd_float,
            shape=maybe_shape((constraint_solver.n_cone_constraints_, _B), is_cone_incremental),
            layout=serial_layout if is_cone_incremental else None,
        ),
        search=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        qfrc_constraint=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        qacc=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        qacc_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        qacc_prev=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        mv=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        cg_prev_grad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        cg_prev_Mgrad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        nt_vec=V(dtype=gs.qd_float, shape=(solver.n_dofs_ * nt_vec_n_slots, _B), layout=dof_vec_layout),
        # When the register-tiled mass factor is on, reuse its scratch (rigid_info.mass_mat_tiled_scratch, allocated
        # with this exact shape) as the Hessian buffer rather than allocating a second one: the factor only writes it
        # before the constraint solve repopulates it in the same step.
        nt_H=(
            solver.rigid_info.mass_mat_tiled_scratch
            if solver.rigid_config.enable_register_tiled_mass
            else V(dtype=gs.qd_float, shape=(_B, solver.n_dofs_, solver.n_dofs_))
        ),
        nt_jacobi=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), layout=dof_vec_layout),
        incr_changed_idx=V(dtype=gs.qd_int, shape=(len_constraints_, _B), layout=serial_layout),
        incr_n_changed=V(dtype=gs.qd_int, shape=(_B,)),
        # Layout-flippable constraint-state tensors: allocated as qd.Tensor wrappers, optionally with
        # ``layout=(1, 0)`` to physically store as (_B, len_constraints_). Canonical shape stays (len_constraints_, _B);
        # kernel-body indexing ``Jaref[i_c, i_b]`` is rewritten by the AST when ``layout != None``.
        active=V(dtype=gs.qd_bool, shape=(len_constraints_, _B), layout=con_layout),
        prev_active=V(dtype=gs.qd_bool, shape=(len_constraints_, _B), layout=serial_layout),
        diag=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=con_layout),
        aref=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=serial_layout),
        Jaref=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=con_layout),
        efc_frictionloss=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=con_layout),
        efc_force=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=serial_layout),
        efc_D=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=con_layout),
        jv=V(dtype=gs.qd_float, shape=(len_constraints_, _B), layout=con_layout),
        jac=V(dtype=gs.qd_float, shape=jac_shape, layout=jac_layout),
        jac_dofs_idx=V(
            dtype=gs.qd_int, shape=jac_dofs_idx_shape, layout=jac_layout if constraint_solver.sparse_solve else None
        ),
        jac_n_dofs=V(dtype=gs.qd_int, shape=jac_n_dofs_shape, layout=serial_layout if jac_n_dofs_shape else None),
        # Backward gradients
        dL_dqacc=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        dL_dM=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, solver.n_dofs_, _B), solver._requires_grad)),
        dL_djac=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, solver.n_dofs_, _B), solver._requires_grad)),
        dL_daref=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        dL_defc_D=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        dL_dforce=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_u=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_r=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_p=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_Ap=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_Ju=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        bw_y=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        bw_w=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        # Timers
        timers=V(dtype=qd.i64 if gs.backend != gs.metal else qd.i32, shape=(10, _B)),
        solver_iter_counter=V(dtype=qd.i32, shape=()),
        graph_counter=qd.ndarray(qd.i32, shape=()),
        early_exit_flag=V(dtype=qd.i32, shape=()),
        nt_H_cone_free_diag=V(
            dtype=gs.qd_float,
            shape=maybe_shape((_B, solver.n_dofs_), solver.rigid_config.enable_cone_free_hessian_reuse),
        ),
        # Allocated last to preserve the allocation order of the tensors above (see the warning at the top).
        island=get_island_state(solver, collider),
    )


# =========================================== Collider ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactData:
    kind: ClassVar[DataKind] = DataKind.STATE

    geom_a: qd.Tensor
    geom_b: qd.Tensor
    penetration: qd.Tensor
    normal: qd.Tensor
    pos: qd.Tensor
    friction: qd.Tensor
    friction_torsional: qd.Tensor
    friction_rolling: qd.Tensor
    sol_params: qd.Tensor
    force: qd.Tensor
    link_a: qd.Tensor
    link_b: qd.Tensor
    pair_idx: qd.Tensor


def get_contact_data(solver, max_candidate_contacts, requires_grad):
    _B = solver._B
    max_candidate_contacts_ = max(max_candidate_contacts, 1)

    return ContactData(
        geom_a=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        geom_b=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        normal=V(dtype=gs.qd_vec3, shape=(max_candidate_contacts_, _B), needs_grad=requires_grad),
        pos=V(dtype=gs.qd_vec3, shape=(max_candidate_contacts_, _B), needs_grad=requires_grad),
        penetration=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B), needs_grad=requires_grad),
        friction=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        friction_torsional=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        friction_rolling=V(dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        sol_params=V_VEC(7, dtype=gs.qd_float, shape=(max_candidate_contacts_, _B)),
        force=V(dtype=gs.qd_vec3, shape=(max_candidate_contacts_, _B)),
        link_a=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        link_b=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
        pair_idx=V(dtype=gs.qd_int, shape=(max_candidate_contacts_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DiffContactInput:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    ### Non-differentiable input data
    # Geom id of the two geometries
    geom_a: qd.Tensor
    geom_b: qd.Tensor
    # Local positions of the 3 vertices from the two geometries that define the face on the Minkowski difference
    local_pos1_a: qd.Tensor
    local_pos1_b: qd.Tensor
    local_pos1_c: qd.Tensor
    local_pos2_a: qd.Tensor
    local_pos2_b: qd.Tensor
    local_pos2_c: qd.Tensor
    # Local positions of the 1 vertex from the two geometries that define the support point for the face above
    w_local_pos1: qd.Tensor
    w_local_pos2: qd.Tensor
    # Plane-convex contacts only: the convex support "core" (box vertex / sphere center / capsule nearest endpoint)
    # in the convex geom's local frame.
    core_local: qd.Tensor
    # Reference id of the contact point, which is needed for the backward pass
    ref_id: qd.Tensor
    # Flag whether the contact data can be computed in numerically stable way in both the forward and backward passes
    valid: qd.Tensor
    ### Differentiable input data
    # Reference penetration depth, which is needed for computing the weight of the contact point
    ref_penetration: qd.Tensor


def get_diff_contact_input(_B, max_contacts_per_pair, is_active, requires_grad=False):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active and requires_grad)
    return DiffContactInput(
        geom_a=V(dtype=gs.qd_int, shape=shape),
        geom_b=V(dtype=gs.qd_int, shape=shape),
        local_pos1_a=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos1_b=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos1_c=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_a=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_b=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_c=V_VEC(3, dtype=gs.qd_float, shape=shape),
        w_local_pos1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        w_local_pos2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        core_local=V_VEC(3, dtype=gs.qd_float, shape=shape),
        ref_id=V(dtype=gs.qd_int, shape=shape),
        valid=V(dtype=gs.qd_int, shape=shape),
        ref_penetration=V(dtype=gs.qd_float, shape=shape, needs_grad=True),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SortBuffer:
    kind: ClassVar[DataKind] = DataKind.WARMSTART

    value: qd.Tensor
    i_g: qd.Tensor
    is_max: qd.Tensor


def get_sort_buffer(solver):
    _B = solver._B

    return SortBuffer(
        value=V(dtype=gs.qd_float, shape=(2 * solver.n_geoms_, _B)),
        i_g=V(dtype=gs.qd_int, shape=(2 * solver.n_geoms_, _B)),
        is_max=V(dtype=gs.qd_bool, shape=(2 * solver.n_geoms_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactCache:
    kind: ClassVar[DataKind] = DataKind.WARMSTART

    normal: qd.Tensor
    # Previous-step penetration per pair (reset to 0 when out of contact), the warm-start for the MPR->GJK gate.
    penetration: qd.Tensor


def get_contact_cache(solver, n_possible_pairs):
    _B = solver._B
    return ContactCache(
        normal=V_VEC(3, dtype=gs.qd_float, shape=(n_possible_pairs, _B)),
        penetration=V(dtype=gs.qd_float, shape=(n_possible_pairs, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class NarrowphaseWorkQueues:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    mpr_i_b: qd.Tensor
    mpr_i_ga: qd.Tensor
    mpr_i_gb: qd.Tensor
    mpr_i_pair: qd.Tensor
    mpr_contact_pos_0: qd.Tensor
    mpr_normal_0: qd.Tensor
    mpr_penetration_0: qd.Tensor
    # Whether contact0 preferred GJK (the per-pair MPR->GJK gate fired). The multicontact pass uses GJK for contact0
    # when set, and otherwise tries MPR first and falls back to GJK per perturbed contact.
    mpr_prefer_gjk: qd.Tensor
    mpr_queue_size: qd.Tensor
    mpr_work_counter: qd.Tensor


def get_narrowphase_work_queues(max_entries):
    return NarrowphaseWorkQueues(
        mpr_i_b=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_i_ga=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_i_gb=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_i_pair=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_contact_pos_0=V_VEC(3, dtype=gs.qd_float, shape=(max_entries,)),
        mpr_normal_0=V_VEC(3, dtype=gs.qd_float, shape=(max_entries,)),
        mpr_penetration_0=V(dtype=gs.qd_float, shape=(max_entries,)),
        mpr_prefer_gjk=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_queue_size=V(dtype=gs.qd_int, shape=(1,)),
        mpr_work_counter=V(dtype=gs.qd_int, shape=(1,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ColliderState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    sort_buffer: SortBuffer
    contact_data: ContactData
    active_buffer: qd.Tensor
    n_broad_pairs: qd.Tensor
    broad_collision_pairs: qd.Tensor
    active_buffer_awake: qd.Tensor
    active_buffer_hib: qd.Tensor
    box_depth: qd.Tensor
    box_points: qd.Tensor
    box_pts: qd.Tensor
    box_lines: qd.Tensor
    box_linesu: qd.Tensor
    box_axi: qd.Tensor
    box_ppts2: qd.Tensor
    box_pu: qd.Tensor
    xyz_max_min: qd.Tensor
    prism: qd.Tensor
    n_contacts: qd.Tensor = of_kind(DataKind.STATE)
    n_contacts_hibernated: qd.Tensor = of_kind(DataKind.STATE)
    first_time: qd.Tensor = of_kind(DataKind.WARMSTART)
    contact_cache: ContactCache
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: DiffContactInput
    narrowphase_work_queues: NarrowphaseWorkQueues
    contact_sort_key: qd.Tensor
    contact_sort_idx: qd.Tensor = of_kind(DataKind.STATE)
    contact_proj_v: qd.Tensor
    contact_keep: qd.Tensor
    contact_hull_stack: qd.Tensor
    # Per-bucket lex sort permutation used by the cooperative dedup kernel
    # (func_clamp_prune_contacts_coop) for the phase-3 (u, v) lex sort. Sized to max_candidate_contacts because
    # each env writes its own permutation.
    contact_lex_idx: qd.Tensor


def get_collider_state(
    solver, rigid_config, n_possible_pairs, max_collision_pairs_broad_k, collider_info, collider_static_config
):
    _B = solver._B
    n_geoms = solver.n_geoms_
    max_collision_pairs = min(solver.max_collision_pairs, n_possible_pairs)
    max_collision_pairs_broad = max_collision_pairs * max_collision_pairs_broad_k
    # Already sized per regime (convex vs nonconvex) by Collider._init_max_contacts, which runs before this.
    max_candidate_contacts = max(collider_info.max_candidate_contacts[None], 1)
    requires_grad = rigid_config.requires_grad

    box_depth_shape = maybe_shape(
        (collider_static_config.n_contacts_per_nonconvex_pair, _B), rigid_config.box_box_detection
    )
    box_points_shape = maybe_shape(
        (collider_static_config.n_contacts_per_nonconvex_pair, _B), rigid_config.box_box_detection
    )
    box_pts_shape = maybe_shape((6, _B), rigid_config.box_box_detection)
    box_lines_shape = maybe_shape((4, _B), rigid_config.box_box_detection)
    box_linesu_shape = maybe_shape((4, _B), rigid_config.box_box_detection)
    box_axi_shape = maybe_shape((3, _B), rigid_config.box_box_detection)
    box_ppts2_shape = maybe_shape((4, 2, _B), rigid_config.box_box_detection)
    box_pu_shape = maybe_shape((4, _B), rigid_config.box_box_detection)
    prune_shape = maybe_shape((max(max_candidate_contacts, 1), _B), collider_static_config.has_prunable_contacts)

    return ColliderState(
        sort_buffer=get_sort_buffer(solver),
        active_buffer=V(dtype=gs.qd_int, shape=(n_geoms, _B)),
        n_broad_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        active_buffer_awake=V(dtype=gs.qd_int, shape=(n_geoms, _B)),
        active_buffer_hib=V(dtype=gs.qd_int, shape=(n_geoms, _B)),
        box_depth=V(dtype=gs.qd_float, shape=box_depth_shape),
        box_points=V_VEC(3, dtype=gs.qd_float, shape=box_points_shape),
        box_pts=V_VEC(3, dtype=gs.qd_float, shape=box_pts_shape),
        box_lines=V_VEC(6, dtype=gs.qd_float, shape=box_lines_shape),
        box_linesu=V_VEC(6, dtype=gs.qd_float, shape=box_linesu_shape),
        box_axi=V_VEC(3, dtype=gs.qd_float, shape=box_axi_shape),
        box_ppts2=V(dtype=gs.qd_float, shape=box_ppts2_shape),
        box_pu=V_VEC(3, dtype=gs.qd_float, shape=box_pu_shape),
        xyz_max_min=V(dtype=gs.qd_float, shape=(6, _B)),
        prism=V_VEC(3, dtype=gs.qd_float, shape=(6, _B)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts_hibernated=V(dtype=gs.qd_int, shape=(_B,)),
        first_time=V(dtype=gs.qd_bool, shape=(_B,)),
        contact_cache=get_contact_cache(solver, n_possible_pairs),
        broad_collision_pairs=V_VEC(2, dtype=gs.qd_int, shape=(max(max_collision_pairs_broad, 1), _B)),
        contact_data=get_contact_data(solver, max_candidate_contacts, requires_grad),
        diff_contact_input=get_diff_contact_input(_B, max(max_candidate_contacts, 1), True, requires_grad),
        narrowphase_work_queues=get_narrowphase_work_queues(
            max(max_collision_pairs_broad * _B, 1) if collider_static_config.has_non_box_plane_convex_convex else 1
        ),
        contact_sort_key=V(dtype=gs.qd_float, shape=(max(max_candidate_contacts, 1), _B)),
        contact_sort_idx=V(dtype=gs.qd_int, shape=(max(max_candidate_contacts, 1), _B)),
        contact_proj_v=V(dtype=gs.qd_float, shape=prune_shape),
        contact_keep=V(dtype=gs.qd_int, shape=prune_shape),
        contact_hull_stack=V(dtype=gs.qd_int, shape=prune_shape),
        contact_lex_idx=V(dtype=gs.qd_int, shape=prune_shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsSpatialGrid:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    # Per-geom 8x8x8 grid over collision verts in the local AABB: a permutation of vert indices sorted by grid
    # cell (z fastest), the matching vert positions duplicated in that order for sequential streaming, and per-cell
    # vert ranges (8^3 + 1 entries per geom), so a scan visits only the cells overlapping a query box. The cell
    # mapping is anchored by geoms_origin / geoms_inv_cell_size in the geom frame.
    verts_idx: qd.Tensor
    verts_pos: qd.Tensor
    cells_vert_start: qd.Tensor
    geoms_origin: qd.Tensor
    geoms_inv_cell_size: qd.Tensor


def get_verts_spatial_grid(solver):
    return VertsSpatialGrid(
        verts_idx=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        verts_pos=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_verts_,)),
        cells_vert_start=V(dtype=gs.qd_int, shape=(max(solver.n_geoms * (8**3 + 1), 1),)),
        geoms_origin=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_geoms_,)),
        geoms_inv_cell_size=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_geoms_,)),
    )


@qd.data_oriented
class ColliderStaticConfig(metaclass=AutoInitMeta):
    has_terrain: bool
    # True when the scene has convex-convex collision pairs not handled by
    # func_narrow_phase_convex_specializations (box-box, plane-box). Computed once
    # at scene build time by iterating all geom pairs in collider._init_static_config().
    # On GPU, the split narrowphase path runs (contact0 + multicontact + sort).
    # On CPU, falls back to the monolithic func_narrow_phase_convex_vs_convex.
    has_non_box_plane_convex_convex: bool
    has_convex_specialization: bool
    has_nonconvex_nonterrain: bool
    # True when link-pair contact pruning can ever do useful work. False when every link has at most one convex geom and
    # no terrain is present (each (link_a, link_b) bucket then holds at most one geom-pair's contacts, capped at
    # n_contacts_per_convex_pair, so the 2D hull is at best a marginal reduction). Lets us skip the pruning kernel call
    # and its scratch buffers entirely. Composes with contact islands: pruning writes a logical permutation into
    # contact_sort_idx and the island construction reads contacts through it, so pruning collapses the contacts first.
    has_prunable_contacts: bool
    # True when contacts are ordered deterministically by position in add_inequality_constraints, making the contact
    # order independent of the racy atomic_add narrowphase layout. Only meaningful when has_non_box_plane_convex_convex
    # on GPU; disabled in autodiff (the gradient writeback indexes contacts by physical layout, so a non-identity
    # permutation would misattach gradients).
    spatial_sort_supported: bool
    # maximum number of contact pairs per collision pair
    n_contacts_per_convex_pair: int
    # maximum number of contact pairs per nonconvex (vertex-vs-SDF) collision pair; >= n_contacts_per_convex_pair
    n_contacts_per_nonconvex_pair: int
    # ccd algorithm
    ccd_algorithm: int


# =========================================== MPR ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRSimplexSupport:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    v1: qd.Tensor
    v2: qd.Tensor
    v: qd.Tensor


def get_mpr_simplex_support(B_):
    return MPRSimplexSupport(
        v1=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v2=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    simplex_support: MPRSimplexSupport
    simplex_size: qd.Tensor
    # What the depth of the contact in simplex_support[1..3] is worth, a PORTAL_STATUS value. Zero means no portal.
    # Only VALID portals are reused (perturbation reconstruction, EPA seeding); INVALID forces a GJK refine.
    portal_status: qd.Tensor


def get_mpr_state(B_):
    return MPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=V(dtype=gs.qd_int, shape=(B_,)),
        portal_status=V(dtype=gs.qd_int, shape=(B_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    CCD_EPS: qd.Tensor
    CCD_TOLERANCE: qd.Tensor
    CCD_ITERATIONS: qd.Tensor


def get_mpr_info(**kwargs):
    return MPRInfo(
        CCD_EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_EPS"]),
        CCD_TOLERANCE=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_TOLERANCE"]),
        CCD_ITERATIONS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_ITERATIONS"]),
    )


# =========================================== GJK ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MDVertex:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # Vertex of the Minkowski difference
    obj1: qd.Tensor
    obj2: qd.Tensor
    local_obj1: qd.Tensor
    local_obj2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor
    mink: qd.Tensor


def get_gjk_simplex_vertex(_B, is_active):
    shape = maybe_shape((_B, 4), is_active)
    return MDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


def get_epa_polytope_vertex(_B, gjk_info, is_active):
    max_num_polytope_verts = 5 + gjk_info.epa_max_iterations[None]
    shape = maybe_shape((_B, max_num_polytope_verts), is_active)
    return MDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKSimplex:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    nverts: qd.Tensor


def get_gjk_simplex(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return GJKSimplex(nverts=V(dtype=gs.qd_int, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKSimplexBuffer:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    normal: qd.Tensor
    sdist: qd.Tensor


def get_gjk_simplex_buffer(_B, is_active):
    shape = maybe_shape((_B, 4), is_active)
    return GJKSimplexBuffer(normal=V_VEC(3, dtype=gs.qd_float, shape=shape), sdist=V(dtype=gs.qd_float, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytope:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    nverts: qd.Tensor
    nfaces: qd.Tensor
    nfaces_map: qd.Tensor
    horizon_nedges: qd.Tensor
    horizon_w: qd.Tensor


def get_epa_polytope(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return EPAPolytope(
        nverts=V(dtype=gs.qd_int, shape=shape),
        nfaces=V(dtype=gs.qd_int, shape=shape),
        nfaces_map=V(dtype=gs.qd_int, shape=shape),
        horizon_nedges=V(dtype=gs.qd_int, shape=shape),
        horizon_w=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytopeFace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    verts_idx: qd.Tensor
    adj_idx: qd.Tensor
    normal: qd.Tensor
    dist2: qd.Tensor
    map_idx: qd.Tensor
    visited: qd.Tensor


def get_epa_polytope_face(_B, polytope_max_faces, is_active):
    shape = maybe_shape((_B, polytope_max_faces), is_active)
    return EPAPolytopeFace(
        verts_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        adj_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dist2=V(dtype=gs.qd_float, shape=shape),
        map_idx=V(dtype=gs.qd_int, shape=shape),
        visited=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytopeHorizonData:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    face_idx: qd.Tensor
    edge_idx: qd.Tensor


def get_epa_polytope_horizon_data(_B, polytope_max_horizons, is_active):
    shape = maybe_shape((_B, polytope_max_horizons), is_active)
    return EPAPolytopeHorizonData(face_idx=V(dtype=gs.qd_int, shape=shape), edge_idx=V(dtype=gs.qd_int, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactFace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    vert1: qd.Tensor
    vert2: qd.Tensor
    endverts: qd.Tensor
    normal1: qd.Tensor
    normal2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor


def get_contact_face(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return ContactFace(
        vert1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vert2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactNormal:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    endverts: qd.Tensor
    normal: qd.Tensor
    id: qd.Tensor


def get_contact_normal(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return ContactNormal(
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactHalfspace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    normal: qd.Tensor
    dist: qd.Tensor


def get_contact_halfspace(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return ContactHalfspace(normal=V_VEC(3, dtype=gs.qd_float, shape=shape), dist=V(dtype=gs.qd_float, shape=shape))


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class Witness:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    point_obj1: qd.Tensor
    point_obj2: qd.Tensor


def get_witness(_B, max_contacts_per_pair, is_active):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active)
    return Witness(
        point_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape), point_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape)
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    support_mesh_prev_vertex_id: qd.Tensor
    simplex_vertex: MDVertex
    simplex_buffer: GJKSimplexBuffer
    simplex: GJKSimplex
    simplex_vertex_intersect: MDVertex
    simplex_buffer_intersect: GJKSimplexBuffer
    nsimplex: qd.Tensor
    last_searched_simplex_vertex_id: qd.Tensor
    polytope: EPAPolytope
    polytope_verts: MDVertex
    polytope_faces: EPAPolytopeFace
    polytope_faces_map: qd.Tensor
    polytope_horizon_data: EPAPolytopeHorizonData
    polytope_horizon_stack: EPAPolytopeHorizonData
    contact_faces: ContactFace
    contact_normals: ContactNormal
    contact_halfspaces: ContactHalfspace
    contact_clipped_polygons: qd.Tensor
    multi_contact_flag: qd.Tensor
    witness: Witness
    n_witness: qd.Tensor
    n_contacts: qd.Tensor
    contact_pos: qd.Tensor
    normal: qd.Tensor
    is_col: qd.Tensor
    penetration: qd.Tensor
    distance: qd.Tensor
    # Index of the EPA polytope face nearest to the origin (the penetration face), or -1 when EPA did not run. Lets the
    # multi-contact perturbation reconstruct the exact unperturbed normal from that face's un-rotated support simplex.
    nearest_face: qd.Tensor
    # Differentiable contact detection
    diff_contact_input: DiffContactInput
    n_diff_contact_input: qd.Tensor
    diff_penetration: qd.Tensor


def get_gjk_state(_B, rigid_config, gjk_info, is_active, requires_grad=False):
    enable_mujoco_compatibility = rigid_config.enable_mujoco_compatibility
    polytope_max_faces = gjk_info.polytope_max_faces[None]
    max_contacts_per_pair = gjk_info.max_contacts_per_pair[None]
    max_contact_polygon_verts = gjk_info.max_contact_polygon_verts[None]

    # FIXME: Define GJKState and MujocoCompatGJKState that derives from the former but defines additional attributes
    return GJKState(
        # GJK simplex
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(_B, is_active),
        simplex_buffer=get_gjk_simplex_buffer(_B, is_active),
        simplex=get_gjk_simplex(_B, is_active),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B, is_active),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B, is_active),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA polytope
        polytope=get_epa_polytope(_B, is_active),
        polytope_verts=get_epa_polytope_vertex(_B, gjk_info, is_active),
        polytope_faces=get_epa_polytope_face(_B, polytope_max_faces, is_active),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(_B, polytope_max_faces)),
        polytope_horizon_data=get_epa_polytope_horizon_data(_B, 6 + gjk_info.epa_max_iterations[None], is_active),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_B, polytope_max_faces * 3, is_active),
        # Multi-contact detection (MuJoCo compatibility)
        contact_faces=get_contact_face(_B, max_contact_polygon_verts, is_active),
        contact_normals=get_contact_normal(_B, max_contact_polygon_verts, is_active),
        contact_halfspaces=get_contact_halfspace(_B, max_contact_polygon_verts, is_active),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(_B, 2, max_contact_polygon_verts)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Final results
        witness=get_witness(_B, max_contacts_per_pair, is_active),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        is_col=V(dtype=gs.qd_bool, shape=(_B,)),
        penetration=V(dtype=gs.qd_float, shape=(_B,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        nearest_face=V(dtype=gs.qd_int, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(_B, max(max_contacts_per_pair, 1), is_active, requires_grad),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(_B,)),
        diff_penetration=V(dtype=gs.qd_float, shape=maybe_shape((_B, max_contacts_per_pair), requires_grad)),
    )


def get_gjk_state_contact_only(_B):
    """Minimal GJK state for contact detection only (no EPA, no multi-contact).

    Used by kernel 1 to run func_gjk as a boolean overlap test. All EPA polytope,
    multi-contact, and differentiable fields are allocated at dummy size (1,) since
    func_gjk never accesses them.
    """
    _dummy_B = 1

    return GJKState(
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(_B, is_active=True),
        simplex_buffer=get_gjk_simplex_buffer(_B, is_active=True),
        simplex=get_gjk_simplex(_B, is_active=True),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B, is_active=True),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B, is_active=True),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA - dummy allocations, never accessed by func_gjk
        polytope=get_epa_polytope(_dummy_B, is_active=True),
        polytope_verts=MDVertex(
            obj1=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            obj2=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            local_obj1=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            local_obj2=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            id1=V(dtype=gs.qd_int, shape=(1, 1)),
            id2=V(dtype=gs.qd_int, shape=(1, 1)),
            mink=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        ),
        polytope_faces=get_epa_polytope_face(_dummy_B, 1, is_active=True),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(1, 1)),
        polytope_horizon_data=get_epa_polytope_horizon_data(_dummy_B, 1, is_active=True),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_dummy_B, 1, is_active=True),
        # Multi-contact - dummy
        contact_faces=get_contact_face(_dummy_B, 1, is_active=True),
        contact_normals=get_contact_normal(_dummy_B, 1, is_active=True),
        contact_halfspaces=get_contact_halfspace(_dummy_B, 1, is_active=True),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(1, 2, 1)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Results - full _B for fields func_gjk writes; dummy for EPA-only fields
        witness=get_witness(_B, 1, is_active=True),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(1,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        is_col=V(dtype=gs.qd_bool, shape=(1,)),
        penetration=V(dtype=gs.qd_float, shape=(1,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        nearest_face=V(dtype=gs.qd_int, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(_dummy_B, 1, is_active=False),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(1,)),
        diff_penetration=V(dtype=gs.qd_float, shape=()),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    max_contacts_per_pair: qd.Tensor
    max_contact_polygon_verts: qd.Tensor
    # Maximum number of iterations for GJK and EPA algorithms
    gjk_max_iterations: qd.Tensor
    epa_max_iterations: qd.Tensor
    FLOAT_MIN: qd.Tensor
    FLOAT_MIN_SQ: qd.Tensor
    FLOAT_MAX: qd.Tensor
    FLOAT_MAX_SQ: qd.Tensor
    # Tolerance for stopping GJK and EPA algorithms when they converge (only for non-discrete geometries).
    tolerance: qd.Tensor
    # If the distance between two objects is smaller than this value, we consider them colliding.
    collision_eps: qd.Tensor
    # In safe GJK, we do not allow degenerate simplex to happen, because it becomes the main reason of EPA errors.
    # To prevent degeneracy, we throw away the simplex that has smaller degeneracy measure (e.g. colinearity,
    # coplanarity) than this threshold.
    simplex_max_degeneracy_sq: qd.Tensor
    polytope_max_faces: qd.Tensor
    # Threshold for reprojection error when we compute the witness points from the polytope. In computing the
    # witness points, we project the origin onto the polytope faces and compute the barycentric coordinates of the
    # projected point. To confirm the projection is valid, we compute the projected point using the barycentric
    # coordinates and compare it with the original projected point. If the difference is larger than this threshold,
    # we consider the projection invalid, because it means numerical errors are too large.
    # We check both relative and absolute errors: the relative error catches numerically degenerate faces,
    # while the absolute error prevents false rejections on smooth geometries (e.g. spheres) where
    # polytope faces become extremely small near convergence, amplifying the relative error.
    polytope_max_rel_reprojection_error: qd.Tensor
    polytope_max_abs_reprojection_error: qd.Tensor
    # Tolerance for normal alignment between (face-face) or (edge-face). The normals should align within this
    # tolerance to be considered as a valid parallel contact.
    contact_face_tol: qd.Tensor
    contact_edge_tol: qd.Tensor
    # Epsilon values for differentiable contact. [eps_boundary] denotes the maximum distance between the face
    # and the support point in the direction of the face normal. If this distance is 0, the face is on the
    # boundary of the Minkowski difference. For [eps_distance], the distance between the origin and the face
    # should not exceed this eps value plus the default EPA depth. For [eps_affine], the affine coordinates
    # of the origin's projection onto the face should not violate [0, 1] range by this eps value.
    # FIXME: Adjust these values based on the case study.
    diff_contact_eps_boundary: qd.Tensor
    diff_contact_eps_distance: qd.Tensor
    diff_contact_eps_affine: qd.Tensor
    # The minimum norm of the normal to be considered as a valid normal in the differentiable formulation.
    diff_contact_min_normal_norm: qd.Tensor
    # The minimum penetration depth to be considered as a valid contact in the differentiable formulation.
    # The contact with penetration depth smaller than this value is ignored in the differentiable formulation.
    # This should be large enough to be safe from numerical errors, because in the backward pass, the computed
    # penetration depth could be different from the forward pass due to the numerical errors. If this value is
    # too small, the non-zero penetration depth could be falsely computed to 0 in the backward pass and thus
    # produce nan values for the contact normal.
    diff_contact_min_penetration: qd.Tensor


def get_gjk_info(**kwargs):
    return GJKInfo(
        max_contacts_per_pair=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["max_contacts_per_pair"]),
        max_contact_polygon_verts=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["max_contact_polygon_verts"]),
        gjk_max_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["gjk_max_iterations"]),
        epa_max_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["epa_max_iterations"]),
        FLOAT_MIN=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MIN"]),
        FLOAT_MIN_SQ=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MIN"] ** 2),
        FLOAT_MAX=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MAX"]),
        FLOAT_MAX_SQ=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MAX"] ** 2),
        tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["tolerance"]),
        collision_eps=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["collision_eps"]),
        simplex_max_degeneracy_sq=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["simplex_max_degeneracy_sq"]),
        polytope_max_faces=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["polytope_max_faces"]),
        polytope_max_rel_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_rel_reprojection_error"]
        ),
        polytope_max_abs_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_abs_reprojection_error"]
        ),
        contact_face_tol=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_face_tol"]),
        contact_edge_tol=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_edge_tol"]),
        diff_contact_eps_boundary=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_boundary"]),
        diff_contact_eps_distance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_distance"]),
        diff_contact_eps_affine=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_affine"]),
        diff_contact_min_normal_norm=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_min_normal_norm"]),
        diff_contact_min_penetration=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_min_penetration"]),
    )


@qd.data_oriented
class GJKStaticConfig(metaclass=AutoInitMeta):
    enable_contact_patch: bool


# =========================================== SupportField ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SupportFieldInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    support_cell_start: qd.Tensor
    support_v: qd.Tensor
    support_vid: qd.Tensor
    support_res: qd.Tensor


def get_support_field_info(n_geoms, n_support_cells, support_res):
    return SupportFieldInfo(
        support_cell_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        support_v=V_VEC(3, dtype=gs.qd_float, shape=(max(n_support_cells, 1),)),
        support_vid=V(dtype=gs.qd_int, shape=(max(n_support_cells, 1),)),
        support_res=V_SCALAR_FROM(dtype=gs.qd_int, value=support_res),
    )


# =========================================== SDF ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SDFGeomInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    T_mesh_to_sdf: qd.Tensor
    sdf_res: qd.Tensor
    sdf_max: qd.Tensor
    sdf_cell_size: qd.Tensor
    sdf_cell_start: qd.Tensor
    # Coarse min-grid companion: per-block minima over grid nodes, a certified lower bound of the trilinear sd.
    sdf_coarse_res: qd.Tensor
    sdf_coarse_cell_start: qd.Tensor


def get_sdf_geom_info(n_geoms):
    return SDFGeomInfo(
        T_mesh_to_sdf=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_res=V_VEC(3, dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_max=V(dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_size=V_VEC(3, dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_start=V(dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_coarse_res=V_VEC(3, dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_coarse_cell_start=V(dtype=gs.qd_int, shape=(n_geoms,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SDFInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    geoms_info: SDFGeomInfo
    geoms_sdf_start: qd.Tensor
    geoms_sdf_val: qd.Tensor
    geoms_sdf_grad: qd.Tensor
    geoms_sdf_closest_vert: qd.Tensor
    geoms_sdf_coarse_val: qd.Tensor


def get_sdf_info(n_geoms, n_cells, n_coarse_cells):
    if math.prod((n_cells, 3)) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"SDF Gradient shape (n_cells={n_cells}, 3) is too large. Consider manually setting larger "
            "'sdf_cell_size' in 'gs.materials.Rigid' options."
        )

    return SDFInfo(
        geoms_info=get_sdf_geom_info(max(n_geoms, 1)),
        geoms_sdf_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        geoms_sdf_val=V(dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_grad=V_VEC(3, dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_closest_vert=V(dtype=gs.qd_int, shape=(max(n_cells, 1),)),
        geoms_sdf_coarse_val=V(dtype=gs.qd_float, shape=(max(n_coarse_cells, 1),)),
    )


# =========================================== ColliderInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ColliderInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    # Narrowphase sub-component descriptions, owned by their respective builders (MPR, GJK, SupportField, SDF) and
    # embedded here so kernels receive the whole collider description as one argument. Each has a single instance,
    # activated (which may reallocate it at its final size) before this struct is built; a reassignment after the
    # embedding would leave the reference here stale.
    mpr: MPRInfo
    gjk: GJKInfo
    support_field: SupportFieldInfo
    sdf: SDFInfo
    vert_neighbors: qd.Tensor
    vert_neighbor_start: qd.Tensor
    vert_n_neighbors: qd.Tensor
    verts_spatial_grid: VertsSpatialGrid
    # (i_ga, i_gb) -> dense pair index, or -1 if invalid. Used by SAP broadphase, narrowphase, and contact cache.
    collision_pair_idx: qd.Tensor
    max_possible_pairs: qd.Tensor
    max_collision_pairs: qd.Tensor
    max_candidate_contacts: qd.Tensor
    max_collision_pairs_broad: qd.Tensor
    # Post-pruning contact-point budget per environment, which sizes the contact constraint buffers (4 constraints per
    # contact point). Smaller than max_candidate_contacts when contact pruning is enabled or 'max_contacts' is set.
    max_contacts: qd.Tensor
    # Compact list of valid collision pairs. Used by all-vs-all broadphase to dispatch valid pairs to GPU threads.
    n_valid_pairs: qd.Tensor
    valid_collision_pairs: qd.Tensor
    # Terrain fields
    terrain_hf: qd.Tensor
    terrain_rc: qd.Tensor
    terrain_scale: qd.Tensor
    terrain_xyz_maxmin: qd.Tensor
    # multi contact perturbation and tolerance
    mc_perturbation: qd.Tensor
    mc_tolerance: qd.Tensor
    mpr_to_gjk_overlap_ratio: qd.Tensor
    mpr_to_gjk_overlap_ratio_valid: qd.Tensor
    mpr_to_gjk_penetration_ratio: qd.Tensor
    # differentiable contact tolerance
    diff_pos_tolerance: qd.Tensor
    diff_normal_tolerance: qd.Tensor
    # link-pair contact pruning
    contact_pruning_tolerance: qd.Tensor
    prune_deep_penetration_ratio: qd.Tensor


def get_collider_info(
    solver,
    n_vert_neighbors,
    n_valid_pairs,
    collider_static_config,
    mpr_info,
    gjk_info,
    support_field_info,
    sdf_info,
    **kwargs,
):
    for geom in solver.geoms:
        if geom.type == gs.GEOM_TYPE.TERRAIN:
            terrain_hf_shape = geom.entity.terrain_hf.shape
            break
    else:
        terrain_hf_shape = 1

    return ColliderInfo(
        mpr=mpr_info,
        gjk=gjk_info,
        support_field=support_field_info,
        sdf=sdf_info,
        vert_neighbors=V(dtype=gs.qd_int, shape=(max(n_vert_neighbors, 1),)),
        vert_neighbor_start=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        vert_n_neighbors=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        verts_spatial_grid=get_verts_spatial_grid(solver),
        collision_pair_idx=V(dtype=gs.qd_int, shape=(solver.n_geoms_, solver.n_geoms_)),
        max_possible_pairs=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs=V(dtype=gs.qd_int, shape=()),
        max_candidate_contacts=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs_broad=V(dtype=gs.qd_int, shape=()),
        max_contacts=V(dtype=gs.qd_int, shape=()),
        n_valid_pairs=V_SCALAR_FROM(dtype=gs.qd_int, value=n_valid_pairs),
        valid_collision_pairs=V(dtype=gs.qd_ivec2, shape=(max(n_valid_pairs, 1),)),
        terrain_hf=V(dtype=gs.qd_float, shape=terrain_hf_shape),
        terrain_rc=V(dtype=gs.qd_int, shape=(2,)),
        terrain_scale=V(dtype=gs.qd_float, shape=(2,)),
        terrain_xyz_maxmin=V(dtype=gs.qd_float, shape=(6,)),
        mc_perturbation=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_perturbation"]),
        mc_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_tolerance"]),
        mpr_to_gjk_overlap_ratio=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mpr_to_gjk_overlap_ratio"]),
        mpr_to_gjk_overlap_ratio_valid=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mpr_to_gjk_overlap_ratio_valid"]),
        mpr_to_gjk_penetration_ratio=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mpr_to_gjk_penetration_ratio"]),
        diff_pos_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_pos_tolerance"]),
        diff_normal_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_normal_tolerance"]),
        contact_pruning_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_pruning_tolerance"]),
        prune_deep_penetration_ratio=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["prune_deep_penetration_ratio"]),
    )


# =========================================== DofsInfo and DofsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DofsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    entity_idx: qd.Tensor
    stiffness: qd.Tensor
    invweight: qd.Tensor
    armature: qd.Tensor
    damping: qd.Tensor
    frictionloss: qd.Tensor
    motion_ang: qd.Tensor
    motion_vel: qd.Tensor
    limit: qd.Tensor
    act_gain: qd.Tensor
    act_bias: qd.Tensor
    force_range: qd.Tensor
    dof_length: qd.Tensor


def get_dofs_info(solver):
    shape = (solver.n_dofs_, solver._B) if solver._options.batch_dofs_info else (solver.n_dofs_,)

    return DofsInfo(
        entity_idx=V(dtype=gs.qd_int, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        invweight=V(dtype=gs.qd_float, shape=shape),
        armature=V(dtype=gs.qd_float, shape=shape),
        damping=V(dtype=gs.qd_float, shape=shape),
        frictionloss=V(dtype=gs.qd_float, shape=shape),
        motion_ang=V(dtype=gs.qd_vec3, shape=shape),
        motion_vel=V(dtype=gs.qd_vec3, shape=shape),
        limit=V(dtype=gs.qd_vec2, shape=shape),
        act_gain=V(dtype=gs.qd_float, shape=shape),
        act_bias=V(dtype=gs.qd_vec3, shape=shape),
        force_range=V(dtype=gs.qd_vec2, shape=shape),
        dof_length=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DofsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    # *_bw: Cache to avoid overwriting for backward pass
    force: qd.Tensor
    qf_bias: qd.Tensor
    qf_passive: qd.Tensor
    qf_actuator: qd.Tensor
    qf_applied: qd.Tensor
    act_length: qd.Tensor
    pos: qd.Tensor
    vel: qd.Tensor = of_kind(DataKind.STATE)
    vel_prev: qd.Tensor = of_kind(DataKind.SCRATCH)
    vel_next: qd.Tensor = of_kind(DataKind.SCRATCH)
    acc: qd.Tensor = of_kind(DataKind.STATE)
    acc_smooth: qd.Tensor
    acc_smooth_bw: qd.Tensor
    qf_smooth: qd.Tensor
    qf_constraint: qd.Tensor
    cdof_ang: qd.Tensor
    cdof_vel: qd.Tensor
    cdofvel_ang: qd.Tensor
    cdofvel_vel: qd.Tensor
    cdofd_ang: qd.Tensor
    cdofd_vel: qd.Tensor
    f_vel: qd.Tensor
    f_ang: qd.Tensor
    ctrl_force: qd.Tensor = of_kind(DataKind.STATE)
    ctrl_pos: qd.Tensor = of_kind(DataKind.STATE)
    ctrl_vel: qd.Tensor = of_kind(DataKind.STATE)
    ctrl_mode: qd.Tensor = of_kind(DataKind.STATE)
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)


def get_dofs_state(solver):
    shape = (solver.n_dofs_, solver._B)
    requires_grad = solver._requires_grad
    shape_bw = maybe_shape((2, *shape), requires_grad)

    return DofsState(
        force=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_bias=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_passive=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_actuator=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_applied=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        act_length=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        pos=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel_prev=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel_next=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc_smooth=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc_smooth_bw=V(dtype=gs.qd_float, shape=shape_bw, needs_grad=requires_grad),
        qf_smooth=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_constraint=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        cdof_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdof_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofvel_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofvel_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        f_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        f_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        ctrl_force=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_pos=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_vel=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_mode=V(dtype=gs.qd_int, shape=shape),
        is_hibernated=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== LinksState and LinksInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class LinksState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    # *_bw: Cache to avoid overwriting for backward pass
    cinr_inertial: qd.Tensor
    cinr_pos: qd.Tensor
    cinr_quat: qd.Tensor
    cinr_mass: qd.Tensor
    crb_inertial: qd.Tensor
    crb_pos: qd.Tensor
    crb_quat: qd.Tensor
    crb_mass: qd.Tensor
    cdd_vel: qd.Tensor
    cdd_ang: qd.Tensor
    pos: qd.Tensor
    quat: qd.Tensor
    pos_bw: qd.Tensor
    quat_bw: qd.Tensor
    i_pos: qd.Tensor
    i_pos_bw: qd.Tensor
    i_quat: qd.Tensor
    j_pos: qd.Tensor
    j_quat: qd.Tensor
    j_pos_bw: qd.Tensor
    j_quat_bw: qd.Tensor
    j_vel: qd.Tensor
    j_ang: qd.Tensor
    cd_ang: qd.Tensor
    cd_vel: qd.Tensor
    # Whether any contact or connect/weld equality row involves the link this step. Written by the constraint
    # assembly (the deterministic post-sort view of the contacts), consumed by the midpoint-integration eligibility;
    # the raw collider contact buffer is only valid through the contact_sort_idx indirection.
    is_constrained: qd.Tensor
    cd_ang_bw: qd.Tensor
    cd_vel_bw: qd.Tensor
    mass_sum: qd.Tensor
    root_COM: qd.Tensor  # COM of the kinematic tree
    root_COM_bw: qd.Tensor
    cacc_ang: qd.Tensor
    cacc_lin: qd.Tensor
    cfrc_ang: qd.Tensor
    cfrc_vel: qd.Tensor
    cfrc_applied_ang: qd.Tensor = of_kind(DataKind.STATE)
    cfrc_applied_vel: qd.Tensor = of_kind(DataKind.STATE)
    cfrc_coupling_ang: qd.Tensor
    cfrc_coupling_vel: qd.Tensor
    contact_force: qd.Tensor
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)
    awake_steps: qd.Tensor = of_kind(DataKind.STATE)


def get_links_state(solver):
    shape = (solver.n_links_, solver._B)
    requires_grad = solver._requires_grad
    # The backward joint buffers hold one slot per joint of a link plus the link itself; collapsed when grad is off.
    n_joints_bw = (max(link.n_joints for link in solver.links) + 1) if requires_grad and solver.n_links else 1
    shape_bw = (solver.n_links_, n_joints_bw, solver._B)

    return LinksState(
        cinr_inertial=V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        cinr_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cinr_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        cinr_mass=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        crb_inertial=V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        crb_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        crb_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        crb_mass=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        cdd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        pos_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        quat_bw=V(dtype=gs.qd_vec4, shape=shape_bw, needs_grad=requires_grad),
        i_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        i_pos_bw=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        i_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        j_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        j_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        j_pos_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        j_quat_bw=V(dtype=gs.qd_vec4, shape=shape_bw, needs_grad=requires_grad),
        j_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        j_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        is_constrained=V(dtype=gs.qd_bool, shape=shape),
        cd_ang_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        cd_vel_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        mass_sum=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        root_COM=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        root_COM_bw=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cacc_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cacc_lin=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_applied_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_applied_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_coupling_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_coupling_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        contact_force=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        is_hibernated=V(dtype=gs.qd_int, shape=shape),
        awake_steps=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class LinksInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    parent_idx: qd.Tensor
    root_idx: qd.Tensor
    q_start: qd.Tensor
    dof_start: qd.Tensor
    joint_start: qd.Tensor
    q_end: qd.Tensor
    dof_end: qd.Tensor
    joint_end: qd.Tensor
    n_dofs: qd.Tensor
    pos: qd.Tensor
    quat: qd.Tensor
    invweight: qd.Tensor
    is_fixed: qd.Tensor
    inertial_pos: qd.Tensor
    inertial_quat: qd.Tensor
    inertial_i: qd.Tensor
    inertial_mass: qd.Tensor
    entity_idx: qd.Tensor
    # Heterogeneous simulation support: per-link geom/vgeom index ranges
    geom_start: qd.Tensor
    geom_end: qd.Tensor
    vgeom_start: qd.Tensor
    vgeom_end: qd.Tensor


def get_links_info(solver):
    links_info_shape = (solver.n_links_, solver._B) if solver._options.batch_links_info else solver.n_links_

    return LinksInfo(
        parent_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        root_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        q_start=V(dtype=gs.qd_int, shape=links_info_shape),
        dof_start=V(dtype=gs.qd_int, shape=links_info_shape),
        joint_start=V(dtype=gs.qd_int, shape=links_info_shape),
        q_end=V(dtype=gs.qd_int, shape=links_info_shape),
        dof_end=V(dtype=gs.qd_int, shape=links_info_shape),
        joint_end=V(dtype=gs.qd_int, shape=links_info_shape),
        n_dofs=V(dtype=gs.qd_int, shape=links_info_shape),
        pos=V(dtype=gs.qd_vec3, shape=links_info_shape),
        quat=V(dtype=gs.qd_vec4, shape=links_info_shape),
        invweight=V(dtype=gs.qd_vec2, shape=links_info_shape),
        is_fixed=V(dtype=gs.qd_bool, shape=links_info_shape),
        inertial_pos=V(dtype=gs.qd_vec3, shape=links_info_shape),
        inertial_quat=V(dtype=gs.qd_vec4, shape=links_info_shape),
        inertial_i=V(dtype=gs.qd_mat3, shape=links_info_shape),
        inertial_mass=V(dtype=gs.qd_float, shape=links_info_shape),
        entity_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        # Heterogeneous simulation support: per-link geom/vgeom index ranges
        geom_start=V(dtype=gs.qd_int, shape=links_info_shape),
        geom_end=V(dtype=gs.qd_int, shape=links_info_shape),
        vgeom_start=V(dtype=gs.qd_int, shape=links_info_shape),
        vgeom_end=V(dtype=gs.qd_int, shape=links_info_shape),
    )


# =========================================== JointsInfo and JointsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class JointsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    type: qd.Tensor
    sol_params: qd.Tensor
    q_start: qd.Tensor
    dof_start: qd.Tensor
    q_end: qd.Tensor
    dof_end: qd.Tensor
    n_dofs: qd.Tensor
    pos: qd.Tensor


def get_joints_info(solver):
    shape = (solver.n_joints_, solver._B) if solver._options.batch_joints_info else (solver.n_joints_,)

    return JointsInfo(
        type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        q_start=V(dtype=gs.qd_int, shape=shape),
        dof_start=V(dtype=gs.qd_int, shape=shape),
        q_end=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        pos=V(dtype=gs.qd_vec3, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class JointsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    xanchor: qd.Tensor
    xaxis: qd.Tensor


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)
    requires_grad = solver._requires_grad

    return JointsState(
        xanchor=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        xaxis=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GeomsInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    pos: qd.Tensor
    center: qd.Tensor
    quat: qd.Tensor
    data: qd.Tensor
    link_idx: qd.Tensor
    type: qd.Tensor
    friction: qd.Tensor
    friction_torsional: qd.Tensor
    friction_rolling: qd.Tensor
    sol_params: qd.Tensor
    vert_num: qd.Tensor
    vert_start: qd.Tensor
    vert_end: qd.Tensor
    verts_state_start: qd.Tensor
    verts_state_end: qd.Tensor
    face_num: qd.Tensor
    face_start: qd.Tensor
    face_end: qd.Tensor
    edge_num: qd.Tensor
    edge_start: qd.Tensor
    edge_end: qd.Tensor
    is_convex: qd.Tensor
    contype: qd.Tensor
    conaffinity: qd.Tensor
    is_fixed: qd.Tensor
    is_decomposed: qd.Tensor
    is_hollow: qd.Tensor
    needs_coup: qd.Tensor
    coup_friction: qd.Tensor
    coup_softness: qd.Tensor
    coup_restitution: qd.Tensor


def get_geoms_info(solver, is_active=True):
    shape = (solver.n_geoms_,) if is_active else ()

    return GeomsInfo(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        center=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
        data=V(dtype=gs.qd_vec7, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        type=V(dtype=gs.qd_int, shape=shape),
        friction=V(dtype=gs.qd_float, shape=shape),
        friction_torsional=V(dtype=gs.qd_float, shape=shape),
        friction_rolling=V(dtype=gs.qd_float, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        vert_num=V(dtype=gs.qd_int, shape=shape),
        vert_start=V(dtype=gs.qd_int, shape=shape),
        vert_end=V(dtype=gs.qd_int, shape=shape),
        verts_state_start=V(dtype=gs.qd_int, shape=shape),
        verts_state_end=V(dtype=gs.qd_int, shape=shape),
        face_num=V(dtype=gs.qd_int, shape=shape),
        face_start=V(dtype=gs.qd_int, shape=shape),
        face_end=V(dtype=gs.qd_int, shape=shape),
        edge_num=V(dtype=gs.qd_int, shape=shape),
        edge_start=V(dtype=gs.qd_int, shape=shape),
        edge_end=V(dtype=gs.qd_int, shape=shape),
        is_convex=V(dtype=gs.qd_bool, shape=shape),
        contype=V(dtype=gs.qd_int, shape=shape),
        conaffinity=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
        is_decomposed=V(dtype=gs.qd_bool, shape=shape),
        is_hollow=V(dtype=gs.qd_bool, shape=shape),
        needs_coup=V(dtype=gs.qd_int, shape=shape),
        coup_friction=V(dtype=gs.qd_float, shape=shape),
        coup_softness=V(dtype=gs.qd_float, shape=shape),
        coup_restitution=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GeomsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    quat: qd.Tensor
    aabb_min: qd.Tensor
    aabb_max: qd.Tensor
    verts_updated: qd.Tensor
    min_buffer_idx: qd.Tensor
    max_buffer_idx: qd.Tensor
    is_hibernated: qd.Tensor = of_kind(DataKind.STATE)
    friction_ratio: qd.Tensor = of_kind(DataKind.INFO)


def get_geoms_state(solver, is_active=True):
    shape = (solver.n_geoms_, solver._B) if is_active else ()
    requires_grad = solver.rigid_config.requires_grad

    return GeomsState(
        pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        aabb_min=V(dtype=gs.qd_vec3, shape=shape),
        aabb_max=V(dtype=gs.qd_vec3, shape=shape),
        verts_updated=V(dtype=gs.qd_bool, shape=shape),
        min_buffer_idx=V(dtype=gs.qd_int, shape=shape),
        max_buffer_idx=V(dtype=gs.qd_int, shape=shape),
        is_hibernated=V(dtype=gs.qd_int, shape=shape),
        friction_ratio=V(dtype=gs.qd_float, shape=shape),
    )


# =========================================== VertsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    init_pos: qd.Tensor
    init_normal: qd.Tensor
    geom_idx: qd.Tensor
    init_center_pos: qd.Tensor
    verts_state_idx: qd.Tensor
    is_fixed: qd.Tensor


def get_verts_info(solver, is_active=True):
    shape = (solver.n_verts_,) if is_active else ()

    return VertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_normal=V(dtype=gs.qd_vec3, shape=shape),
        geom_idx=V(dtype=gs.qd_int, shape=shape),
        init_center_pos=V(dtype=gs.qd_vec3, shape=shape),
        verts_state_idx=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== FacesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FacesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    verts_idx: qd.Tensor
    geom_idx: qd.Tensor


def get_faces_info(solver, is_active=True):
    shape = (solver.n_faces_,) if is_active else ()

    return FacesInfo(verts_idx=V(dtype=gs.qd_ivec3, shape=shape), geom_idx=V(dtype=gs.qd_int, shape=shape))


# =========================================== EdgesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EdgesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    v0: qd.Tensor
    v1: qd.Tensor
    length: qd.Tensor


def get_edges_info(solver, is_active=True):
    shape = (solver.n_edges_,) if is_active else ()

    return EdgesInfo(
        v0=V(dtype=gs.qd_int, shape=shape), v1=V(dtype=gs.qd_int, shape=shape), length=V(dtype=gs.qd_float, shape=shape)
    )


# =========================================== VertsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor


def get_free_verts_state(solver, is_active=True):
    return VertsState(pos=V(dtype=gs.qd_vec3, shape=(solver.n_free_verts_, solver._B) if is_active else ()))


def get_fixed_verts_state(solver, is_active=True):
    return VertsState(pos=V(dtype=gs.qd_vec3, shape=(solver.n_fixed_verts_,) if is_active else ()))


# =========================================== VvertsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VVertsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    init_pos: qd.Tensor
    init_vnormal: qd.Tensor
    vgeom_idx: qd.Tensor
    vverts_state_idx: qd.Tensor


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return VVertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_vnormal=V(dtype=gs.qd_vec3, shape=shape),
        vgeom_idx=V(dtype=gs.qd_int, shape=shape),
        vverts_state_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== VVertsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VVertsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor


def get_vverts_state(solver, is_active=True):
    if not is_active:
        return VVertsState(pos=V(dtype=gs.qd_vec3, shape=()))
    if math.prod((solver.n_custom_vverts_, solver._B, 3)) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Custom-vverts state shape (n_custom_vverts={solver.n_custom_vverts_}, B={solver._B}, 3) is too large. "
            "Consider opting fewer kinematic entities into 'enable_custom_vverts=True', or reducing 'n_envs'."
        )
    return VVertsState(pos=V(dtype=gs.qd_vec3, shape=(solver.n_custom_vverts_, solver._B)))


# =========================================== VfacesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VFacesInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    vverts_idx: qd.Tensor
    vgeom_idx: qd.Tensor


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return VFacesInfo(vverts_idx=V(dtype=gs.qd_ivec3, shape=shape), vgeom_idx=V(dtype=gs.qd_int, shape=shape))


# =========================================== VgeomsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VGeomsInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    pos: qd.Tensor
    quat: qd.Tensor
    link_idx: qd.Tensor
    vvert_num: qd.Tensor
    vvert_start: qd.Tensor
    vvert_end: qd.Tensor
    vface_num: qd.Tensor
    vface_start: qd.Tensor
    vface_end: qd.Tensor
    color: qd.Tensor


def get_vgeoms_info(solver):
    shape = (solver.n_vgeoms_,)

    return VGeomsInfo(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        vvert_num=V(dtype=gs.qd_int, shape=shape),
        vvert_start=V(dtype=gs.qd_int, shape=shape),
        vvert_end=V(dtype=gs.qd_int, shape=shape),
        vface_num=V(dtype=gs.qd_int, shape=shape),
        vface_start=V(dtype=gs.qd_int, shape=shape),
        vface_end=V(dtype=gs.qd_int, shape=shape),
        color=V(dtype=gs.qd_vec4, shape=shape),
    )


# =========================================== VGeomsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VGeomsState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    pos: qd.Tensor
    quat: qd.Tensor


def get_vgeoms_state(solver, is_active=True):
    shape = (solver.n_vgeoms_, solver._B) if is_active else ()

    return VGeomsState(pos=V(dtype=gs.qd_vec3, shape=shape), quat=V(dtype=gs.qd_vec4, shape=shape))


# =========================================== EqualitiesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EqualitiesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    eq_obj1id: qd.Tensor
    eq_obj2id: qd.Tensor
    eq_data: qd.Tensor
    eq_type: qd.Tensor
    sol_params: qd.Tensor


def get_equalities_info(solver, is_active=True):
    shape = (solver.n_candidate_equalities_, solver._B) if is_active else ()

    return EqualitiesInfo(
        eq_obj1id=V(dtype=gs.qd_int, shape=shape),
        eq_obj2id=V(dtype=gs.qd_int, shape=shape),
        eq_data=V(dtype=gs.qd_vec11, shape=shape),
        eq_type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
    )


# =========================================== EntitiesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EntitiesInfo:
    kind: ClassVar[DataKind] = DataKind.INFO

    dof_start: qd.Tensor
    dof_end: qd.Tensor
    n_dofs: qd.Tensor
    link_start: qd.Tensor
    link_end: qd.Tensor
    n_links: qd.Tensor
    geom_start: qd.Tensor
    geom_end: qd.Tensor
    n_geoms: qd.Tensor
    gravity_compensation: qd.Tensor
    is_local_collision_mask: qd.Tensor


def get_entities_info(solver):
    shape = (solver.n_entities_,)

    return EntitiesInfo(
        dof_start=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        link_start=V(dtype=gs.qd_int, shape=shape),
        link_end=V(dtype=gs.qd_int, shape=shape),
        n_links=V(dtype=gs.qd_int, shape=shape),
        geom_start=V(dtype=gs.qd_int, shape=shape),
        geom_end=V(dtype=gs.qd_int, shape=shape),
        n_geoms=V(dtype=gs.qd_int, shape=shape),
        gravity_compensation=V(dtype=gs.qd_float, shape=shape),
        is_local_collision_mask=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== EntitiesState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EntitiesState:
    kind: ClassVar[DataKind] = DataKind.STATE

    is_hibernated: qd.Tensor


def get_entities_state(solver, is_active=True):
    return EntitiesState(is_hibernated=V(dtype=gs.qd_int, shape=(solver.n_entities_, solver._B) if is_active else ()))


# =========================================== RigidAdjointCache ===========================================
@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RigidAdjointCache:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    # This cache stores intermediate values during rigid body simulation to use Quadrants's AD. Quadrants's AD requires
    # us not to overwrite the values that have been read during the forward pass, so we need to store the intemediate
    # values in this cache to avoid overwriting them. Specifically, after we compute next frame's qpos, dofs_vel, and
    # dofs_acc, we need to store them in this cache because we overwrite the values in the next frame. See how
    # [kernel_save_adjoint_cache] is used in [rigid_solver.py] to store the values in this cache.
    qpos: qd.Tensor
    dofs_vel: qd.Tensor
    dofs_acc: qd.Tensor


def get_rigid_adjoint_cache(solver):
    substeps_local = solver._sim.substeps_local
    requires_grad = solver._requires_grad

    return RigidAdjointCache(
        qpos=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_qs_, solver._B), needs_grad=requires_grad),
        dofs_vel=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_dofs_, solver._B), needs_grad=requires_grad),
        dofs_acc=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_dofs_, solver._B), needs_grad=requires_grad),
    )


# =========================================== DynInfo and DynState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DynInfo:
    """Build-time model description of the kinematics+dynamics component, one leaf struct per element type.

    Kinematic-only solvers allocate the collision-related leaves (geoms, verts, faces, edges, equalities) as scalar
    dummies: the fields must exist for kernel argument flattening, while unused ones are pruned per compiled kernel.
    """

    entities: EntitiesInfo
    links: LinksInfo
    joints: JointsInfo
    dofs: DofsInfo
    geoms: GeomsInfo
    verts: VertsInfo
    faces: FacesInfo
    edges: EdgesInfo
    vverts: VVertsInfo
    vfaces: VFacesInfo
    vgeoms: VGeomsInfo
    equalities: EqualitiesInfo


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DynState:
    """Runtime state of the kinematics+dynamics component, one leaf struct per element type.

    Dummy-leaf allocation follows DynInfo. The adjoint cache is a second instance holding copies of the leaves
    overwritten during a forward substep (dofs, links, joints, geoms), with every other field a scalar dummy.
    """

    entities: EntitiesState
    links: LinksState
    joints: JointsState
    dofs: DofsState
    geoms: GeomsState
    free_verts: VertsState
    fixed_verts: VertsState
    vverts: VVertsState
    vgeoms: VGeomsState


def get_dyn_state_adjoint_cache(solver):
    return DynState(
        dofs=get_dofs_state(solver),
        links=get_links_state(solver),
        joints=get_joints_state(solver),
        geoms=get_geoms_state(solver),
        entities=get_entities_state(solver, is_active=False),
        free_verts=get_free_verts_state(solver, is_active=False),
        fixed_verts=get_fixed_verts_state(solver, is_active=False),
        vverts=get_vverts_state(solver, is_active=False),
        vgeoms=get_vgeoms_state(solver, is_active=False),
    )


# =================================== RigidSimStaticConfig ===================================


@qd.data_oriented
class RigidSimStaticConfig(metaclass=AutoInitMeta):
    backend: int
    para_level: int
    enable_collision: bool
    use_hibernation: bool
    batch_links_info: bool
    batch_dofs_info: bool
    batch_joints_info: bool
    enable_mujoco_compatibility: bool
    enable_elliptic_friction: bool
    # Whether contact friction is bounded by the disc of radius mu * f_n taken from the normal force developed at the
    # current iterate, instead of jointly with it through the coupled cone (see gs.contact_resolution). Implies
    # enable_elliptic_friction: the pyramidal rows do not separate into a normal row and a friction disc.
    enable_signorini_contact: bool
    enable_multi_contact: bool
    enable_joint_limit: bool
    box_box_detection: bool
    sparse_solve: bool
    integrator: int
    solver_type: int
    requires_grad: bool
    prefer_decomposed_solver: int = -1  # -1 = None (auto), 0 = False, 1 = True
    # Whether the cone-free assembled Hessian is persisted in nt_H's mirror slots (diagonal in nt_H_cone_free_diag);
    # see the nt_H declaration for the packed-storage mechanics and the rigid solver's resolution for the gating.
    enable_cone_free_hessian_reuse: bool = False
    # Whether contacts carry torsional friction rows resisting relative spin about the contact normal: one extra
    # opposing pyramid pair per contact with the pyramidal cone, one extra cone row with the elliptic cone.
    enable_torsional_friction: bool = False
    # Whether contacts also carry rolling friction rows resisting relative rotation about the two tangent axes: two
    # extra opposing pyramid pairs per contact with the pyramidal cone, two extra cone rows with the elliptic cone.
    # Requires enable_torsional_friction (the rolling rows sit after the spin row in the contact row layout).
    enable_rolling_friction: bool = False
    # Consecutive sub-tolerance steps a body's max DOF velocity must hold before it is ready to hibernate. Guards
    # against a body that is only momentarily slow (e.g. at the apex of a toss) sleeping prematurely.
    hibernation_min_steps: int = 10
    parallel_init: bool = False  # parallelize init over (constraints, envs) when GPU is not saturated by envs alone
    broadphase_traversal: int = 0
    enable_tiled_cholesky_mass_matrix: bool = False
    mass_matrix_fits_shared: bool = False
    # Register-tile width of the mass-matrix Cholesky kernels: 16 (Tile16x16) or 32 (Tile32x32), see
    # cholesky_tile_size_for.
    cholesky_tile_size: int = 32
    # Register-streaming tiled per-entity mass factor for the >shared-cap branch of func_factor_mass (GPU forward
    # only). When True, each entity's single-mass-block submatrix factors in registers via the same TileNxN Cholesky
    # primitive as the Hessian, instead of the shared-pivot cooperative LDL^T. Only enabled when every entity is a
    # single mass block (the common case: one kinematic tree). The tile width is always 32: the path is only taken
    # when the per-entity block exceeds shared memory, which on any real GPU means well over 48 DOFs.
    enable_register_tiled_mass: bool = False
    # When True, the constraint solver uses the GPU subgroup-cooperative kernel variants (warp-cooperative linesearch
    # refinement, per-friction constraint builder, cooperative mass-matrix assembly), together with the batch-first
    # tensor layouts they expect, eg (_B, len_constraints_) for Jaref / efc_D / ... which unlocks coalesced cross-lane
    # reads.
    enable_cooperative_constraint_kernels: bool = False
    # Purely descriptive layout flag: True whenever the layout-flippable constraint-state tensors are physically
    # batch-first, i.e. enable_cooperative_constraint_kernels or serialized execution (env loop outermost, so per-env
    # rows must be contiguous). Consumers that only need iteration order to follow the physical layout (ndrange axes,
    # flattened index decompositions) key on this flag, while algorithm selection (warp-cooperative vs serial
    # reductions) keys on enable_cooperative_constraint_kernels alone.
    constraint_layout_batch_first: bool = False
    # Whether the Newton system is Jacobi-equilibrated: see nt_jacobi in ConstraintState for the mechanics, the rigid
    # solver's resolution for the gating.
    enable_jacobi_equilibration: bool = False
    tiled_n_dofs_per_block: int = -1
    # Dof caps of the shared tiles of the cooperative per-island factor+solve, one launch per cap (see island_tile_caps
    # below): the doublings of the first cap, the tile size doubled up to the smallest dof-carrying tree, below
    # the last one, the largest tile-size multiple that fits GPU shared memory. Islands above the last cap factor in
    # global memory.
    island_tile_cap_first: int = 0
    island_tile_cap_last: int = 0
    # Number of lanes of each launch of the cooperative per-island factor+solve, grid-striding over its class's
    # (env, island) work-list in blocks of the class's tile size: static for CUDA-graph capture, independent of the env
    # count, and sized to keep several warps resident per streaming multiprocessor.
    island_factor_n_lanes: int = 32
    max_n_geoms_per_entity: int = -1
    n_entities: int = -1
    n_links: int = -1
    n_geoms: int = -1

    @property
    def rows_per_contact(self) -> int:
        """Constraint rows per contact.

        The pyramidal cone carries 2 opposing friction-mixed edges per friction axis, the elliptic cone the normal
        row plus one row per friction axis; torsional friction adds the spin axis and rolling friction the two
        tangent axes.
        """
        n_extra_axes = int(self.enable_torsional_friction) + 2 * int(self.enable_rolling_friction)
        if self.enable_elliptic_friction:
            return 3 + n_extra_axes
        return 4 + 2 * n_extra_axes

    @property
    def hessian_rank_update_batch(self) -> int:
        """Number of rank-1 Cholesky updates fused into one column sweep by the CPU incremental factor.

        Sizes the nt_vec slots and the static per-column unroll of func_rank_batch_update_island: 8 amortizes the
        active-set flip batching, widened when the coupled elliptic-cone update must stage 2 slots per cone row
        (see func_cone_rank_update_island).
        """
        return max(8, 2 * self.rows_per_contact)


# =========================================== DataManager ===========================================


def cholesky_tile_size_for(n_dofs):
    """Tile size of the register-streaming Cholesky kernels for a system of n_dofs dofs: 16 where one 16-lane tile
    holds it or where a 32-lane tile would pad it past its lane utilization (33 to 48 dofs), 32 otherwise."""
    return 16 if (n_dofs <= 16 or 32 < n_dofs <= 48) else 32


def island_tile_caps(rigid_config):
    """Ascending dof caps of the shared tiles of the cooperative per-island factor+solve, see island_tile_cap_first:
    the doublings of the first cap below the last one, then the last one."""
    caps = []
    cap = rigid_config.island_tile_cap_first
    while cap < rigid_config.island_tile_cap_last:
        caps.append(cap)
        cap *= 2
    caps.append(rigid_config.island_tile_cap_last)
    return tuple(caps)


@qd.data_oriented
class DataManager:
    def __init__(self, solver, kinematic_only):
        self.rigid_info = get_rigid_info(solver, kinematic_only)

        # Leaf structs are allocated one by one before assembling the per-component aggregates, to control the
        # allocation order of the underlying buffers (see the warning in get_constraint_state).
        is_dynamic = not kinematic_only
        dofs_info = get_dofs_info(solver)
        dofs_state = get_dofs_state(solver)
        links_info = get_links_info(solver)
        links_state = get_links_state(solver)
        joints_info = get_joints_info(solver)
        joints_state = get_joints_state(solver)

        entities_info = get_entities_info(solver)
        entities_state = get_entities_state(solver)

        vverts_info = get_vverts_info(solver)
        vverts_state = get_vverts_state(solver)
        vfaces_info = get_vfaces_info(solver)

        vgeoms_info = get_vgeoms_info(solver)
        vgeoms_state = get_vgeoms_state(solver)

        geoms_info = get_geoms_info(solver, is_dynamic)
        geoms_state = get_geoms_state(solver, is_dynamic)

        verts_info = get_verts_info(solver, is_dynamic)
        faces_info = get_faces_info(solver, is_dynamic)
        edges_info = get_edges_info(solver, is_dynamic)

        free_verts_state = get_free_verts_state(solver, is_dynamic)
        fixed_verts_state = get_fixed_verts_state(solver, is_dynamic)

        equalities_info = get_equalities_info(solver, is_dynamic)

        self.kinematics_scratch = get_kinematics_scratch(solver)

        self.dyn_info = DynInfo(
            entities=entities_info,
            links=links_info,
            joints=joints_info,
            dofs=dofs_info,
            geoms=geoms_info,
            verts=verts_info,
            faces=faces_info,
            edges=edges_info,
            vverts=vverts_info,
            vfaces=vfaces_info,
            vgeoms=vgeoms_info,
            equalities=equalities_info,
        )
        self.dyn_state = DynState(
            entities=entities_state,
            links=links_state,
            joints=joints_state,
            dofs=dofs_state,
            geoms=geoms_state,
            free_verts=free_verts_state,
            fixed_verts=fixed_verts_state,
            vverts=vverts_state,
            vgeoms=vgeoms_state,
        )

        if solver.rigid_config.requires_grad:
            # Data structures required for backward pass
            self.dyn_state_adjoint_cache = get_dyn_state_adjoint_cache(solver)

        self.rigid_adjoint_cache = get_rigid_adjoint_cache(solver)
        self.errno = V(dtype=gs.qd_int, shape=(solver._B,))

        self._solver = solver
        self._ik_scratch = None
        self._weight_scratch = None

    @property
    def weight_scratch(self) -> WeightScratch:
        """Scratch buffers for the inverse-weight computation, allocated the first time a setter triggers one.

        At build time the computation borrows the buffers of the constraint solver, so a scene that never writes a
        mass, a center of mass or an inertia allocates none of this.
        """
        if self._weight_scratch is None:
            self._weight_scratch = get_weight_scratch(self._solver)
        return self._weight_scratch

    @property
    def ik_scratch(self) -> IKScratch:
        """Inverse-kinematics scratch, allocated the first time a solve asks for it.

        The damped normal matrix it holds is quadratic in RigidOptions.IK_max_targets, so a scene that never solves
        inverse kinematics carries none of it.
        """
        if self._ik_scratch is None:
            self._ik_scratch = get_ik_scratch(self._solver)
        return self._ik_scratch


# =========================================== RaycastResult ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RaycastResult:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    distance: qd.Tensor
    geom_idx: qd.Tensor
    hit_point: qd.Tensor
    normal: qd.Tensor


def get_raycast_result(n_envs: int):
    return RaycastResult(
        distance=V(dtype=gs.qd_float, shape=(n_envs,)),
        geom_idx=V(dtype=gs.qd_int, shape=(n_envs,)),
        hit_point=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
    )


GeomsInitAABB = qd.Tensor
