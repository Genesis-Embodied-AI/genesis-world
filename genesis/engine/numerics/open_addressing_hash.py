import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field
from genesis.utils.misc import qd_debug_asserts

EMPTY_KEY = 0xFFFFFFFFFFFFFFFF

HASH_SLOTS = Size("hash_slots")


class OpenAddressingHashScratch(Data, kind=DataKind.SCRATCH):
    """Linear-probing hash table from unique 64-bit unsigned integer (u64) keys to 32-bit integer (i32) values.

    The capacity is a power of two at least twice the number of keys of a build (see open_addressing_hash_sizes). The
    keys of one phase are unique, so an insert is one compare-and-swap claim of the slot, and lookups run in later
    phases. A table is empty once func_reset_hash ran."""

    keys: qd.Tensor = field(qd.u64, shape=(HASH_SLOTS,), ndarray=True)
    vals: qd.Tensor = field(int, shape=(HASH_SLOTS,), ndarray=True)


def open_addressing_hash_sizes(max_keys: int) -> dict[Size, int]:
    """Return the sizes of a table of at most `max_keys` keys per build."""
    return {HASH_SLOTS: 1 << max((max(2 * max_keys, 2) - 1).bit_length(), 1)}


@qd.func
def func_hash_slot(key: qd.u64, table: OpenAddressingHashScratch):
    """Return the home slot of a key, a 64-bit avalanche mix masked to the capacity."""
    h = key
    h = h ^ (h >> qd.u64(33))
    h = h * qd.u64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> qd.u64(33))
    return gs.qd_int(h & qd.u64(table.keys.shape[0] - 1))


@qd.func
def func_insert_hash(key: qd.u64, value: int, table: OpenAddressingHashScratch):
    """Claim a slot for a key that this phase inserts once, publish its value, and return whether it found a slot, which
    a full table fails in debug mode."""
    capacity = table.keys.shape[0]
    i_slot = func_hash_slot(key, table)
    is_inserted = False
    probes = 0
    while not is_inserted and probes < capacity:
        previous_key = qd.atomic_cas(table.keys[i_slot], qd.u64(EMPTY_KEY), key)
        if previous_key == qd.u64(EMPTY_KEY) or previous_key == key:
            table.vals[i_slot] = value
            is_inserted = True
        else:
            i_slot = (i_slot + 1) & (capacity - 1)
            probes += 1
    if qd.static(qd_debug_asserts()):
        assert is_inserted, "OpenAddressingHash.insert: the table of %d slots is full" % capacity
    return is_inserted


@qd.func
def func_lookup_hash(key: qd.u64, table: OpenAddressingHashScratch):
    """Return the value of a key, or -1 for an absent key, the probes stopping at an empty slot or a full turn."""
    capacity = table.keys.shape[0]
    i_slot = func_hash_slot(key, table)
    result = gs.qd_int(-1)
    is_searching = True
    probes = 0
    while is_searching and probes < capacity:
        stored_key = table.keys[i_slot]
        if stored_key == key:
            result = table.vals[i_slot]
            is_searching = False
        elif stored_key == qd.u64(EMPTY_KEY):
            is_searching = False
        else:
            i_slot = (i_slot + 1) & (capacity - 1)
            probes += 1
    return result


@qd.func(requires_top_level=True)
def func_reset_hash(table: OpenAddressingHashScratch):
    qd.loop_config(name="hash_reset")
    for i_slot in range(table.keys.shape[0]):
        table.keys[i_slot] = qd.u64(EMPTY_KEY)
        table.vals[i_slot] = -1
