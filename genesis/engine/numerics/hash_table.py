"""Hash table from unique 64-bit keys to 32-bit values, by open addressing with linear probing."""

import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

HASH_SLOTS = Size("hash_slots")


class HashTableScratch(Data, kind=DataKind.SCRATCH):
    """The slots of a hash table, the largest unsigned 64-bit key marking an empty one."""

    keys: qd.Tensor = field(qd.u64, shape=(HASH_SLOTS,), ndarray=True)
    values: qd.Tensor = field(int, shape=(HASH_SLOTS,), ndarray=True)


def hash_table_sizes(capacity: int) -> dict[Size, int]:
    """
    Return the sizes of a table holding a bounded number of keys.

    The table holds at most `capacity` keys in a power of two of at least twice as many slots.
    """
    return {HASH_SLOTS: 1 << max((max(2 * capacity, 2) - 1).bit_length(), 1)}


@qd.func
def func_hash_slot(table: HashTableScratch, key: qd.u64):
    """Return the home slot of a key, a 64-bit avalanche mix of it masked to the slots."""
    h = key
    h = h ^ (h >> qd.u64(33))
    h = h * qd.u64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> qd.u64(33))
    return gs.qd_int(h & qd.u64(table.keys.shape[0] - 1))


@qd.func
def func_insert_hash(table: HashTableScratch, key: qd.u64, value: int):
    """Insert a key with its value, the table holding at least one empty slot."""
    n_slots = table.keys.shape[0]
    i_slot = func_hash_slot(table, key)
    is_inserted = False
    while not is_inserted:
        key_previous = qd.atomic_cas(table.keys[i_slot], qd.u64(0xFFFFFFFFFFFFFFFF), key)
        if key_previous == qd.u64(0xFFFFFFFFFFFFFFFF) or key_previous == key:
            table.values[i_slot] = value
            is_inserted = True
        else:
            i_slot = (i_slot + 1) & (n_slots - 1)


@qd.func
def func_lookup_hash(table: HashTableScratch, key: qd.u64):
    """Return the value of a key, -1 for a key the table does not hold."""
    n_slots = table.keys.shape[0]
    i_slot = func_hash_slot(table, key)
    value = gs.qd_int(-1)
    is_searching = True
    while is_searching:
        key_stored = table.keys[i_slot]
        if key_stored == key:
            value = table.values[i_slot]
            is_searching = False
        elif key_stored == qd.u64(0xFFFFFFFFFFFFFFFF):
            is_searching = False
        else:
            i_slot = (i_slot + 1) & (n_slots - 1)
    return value


@qd.func(requires_top_level=True)
def func_clear_hash(table: HashTableScratch):
    qd.loop_config(name="hash_table_clear")
    for i_slot in range(table.keys.shape[0]):
        table.keys[i_slot] = qd.u64(0xFFFFFFFFFFFFFFFF)
        table.values[i_slot] = -1
