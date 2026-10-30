import numpy as np
import quadrants as qd

from genesis.utils.misc import qd_debug_asserts

EMPTY_KEY = 0xFFFFFFFFFFFFFFFF


@qd.data_oriented
class OpenAddressingHash:
    """Linear-probing hash table from unique u64 keys to i32 values.

    The capacity is a power of two at least twice the number of keys inserted per build. Keys inserted in one phase
    are unique, so an insert is a single compare-and-swap claim of the slot followed by the value write, and lookups
    run in later phases only. Device code reads the capacity from the shape of the table, so that a table of another
    capacity runs the same compiled kernels.
    """

    def __init__(self, max_keys: int) -> None:
        self.capacity = 1 << max(int(np.ceil(np.log2(max(2 * max_keys, 2)))), 1)
        self.keys = qd.ndarray(qd.u64, shape=(self.capacity,))
        self.vals = qd.ndarray(qd.i32, shape=(self.capacity,))
        self.keys.from_numpy(np.full(self.capacity, EMPTY_KEY, dtype=np.uint64))
        self.vals.from_numpy(np.full(self.capacity, -1, dtype=np.int32))

    @qd.func
    def slot_of(self, key):
        """Home slot of a key: a 64-bit avalanche mix masked to the capacity."""
        h = key
        h = h ^ (h >> qd.u64(33))
        h = h * qd.u64(0xFF51AFD7ED558CCD)
        h = h ^ (h >> qd.u64(33))
        return qd.i32(h & qd.u64(self.keys.shape[0] - 1))

    @qd.func
    def insert(self, key, value):
        """Claim a slot for a key that this phase inserts once, then publish its value. Return whether the key found
        a slot: a full table, which the capacity rule excludes, fails the assertion in debug mode and inserts nothing
        otherwise, the caller raising the error."""
        capacity = self.keys.shape[0]
        i = self.slot_of(key)
        done = False
        probes = 0
        while not done and probes < capacity:
            previous = qd.atomic_cas(self.keys[i], qd.u64(EMPTY_KEY), key)
            if previous == qd.u64(EMPTY_KEY) or previous == key:
                self.vals[i] = value
                done = True
            else:
                i = (i + 1) & (capacity - 1)
                probes += 1
        if qd.static(qd_debug_asserts()):
            assert done, "OpenAddressingHash.insert: the table of %d slots is full" % capacity
        return done

    @qd.func
    def lookup(self, key):
        """Value of a key, or -1 when the key is absent, the probes stopping at an empty slot or after a full turn."""
        capacity = self.keys.shape[0]
        i = self.slot_of(key)
        result = qd.i32(-1)
        searching = True
        probes = 0
        while searching and probes < capacity:
            stored = self.keys[i]
            if stored == key:
                result = self.vals[i]
                searching = False
            elif stored == qd.u64(EMPTY_KEY):
                searching = False
            else:
                i = (i + 1) & (capacity - 1)
                probes += 1
        return result


@qd.func(requires_top_level=True)
def func_reset_hash(
    table: qd.template(),  # OpenAddressingHash
):
    qd.loop_config(name="hash_reset")
    for i in range(table.keys.shape[0]):
        table.keys[i] = qd.u64(EMPTY_KEY)
        table.vals[i] = -1
