"""Log the state of a scene as it steps, one frame per step, and read the log back to seek and replay it.

A trajectory file holds the scene as 'Scene.export' writes it, then its frames in compressed chunks. Every part carries
its own length and checksum, so a run that crashes leaves every chunk written before readable, and a reader stops at the
first part it cannot trust. A checkpoint file (see 'Scene.save_checkpoint') is a trajectory file of one frame holding
every array, scratch included.
"""

import bisect
import dataclasses
import enum
import functools
import io
import itertools
import json
import logging
import os
import queue
import struct
import threading
import zlib
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import numba as nb
import zstandard

import quadrants as qd

import genesis as gs
from genesis.engine.state_change import StateChange, Subscriber
from genesis.engine.states.solvers import SimulatorCheckpoint, SolverCheckpoint
from genesis.options.recorders import TrajectoryFile
from genesis.utils import serialization
from genesis.utils.array_class import ArrayItem, ConfigItem, DataKind, ScalarItem
from genesis.utils.misc import data_to_array, tensor_to_array
from genesis.utils.tools import Rate

from .file_writers import BaseFileWriter
from .recorder_manager import register_recording

if TYPE_CHECKING:
    from genesis.engine.scene import Scene, TrajectorySource
    from genesis.engine.simulator import Simulator
    from genesis.options.renderers import RendererOptions
    from genesis.options.vis import ViewerOptions, VisOptions

LOGGER = logging.getLogger(__name__)

TRAJECTORY_FORMAT = ".gstraj"
MAGIC = b"GSTRAJ4\n"
# The magic of any version of the format, which a file from another version shares up to its version number
MAGIC_PREFIX = b"GSTRAJ"
# The fastest level of zstd, so that the compression thread keeps pace with the fastest stepping scene. The levels
# above shrink frame differences by a few percent only.
ZSTD_LEVEL = 1
# The worker threads of zstd for the frames the device encodes (see DEVICE_ENCODED_FRAME_BYTES), which compress their
# changed blocks in parallel, so that the compression of the largest frames keeps pace with the steps. Beyond a few, it
# shortens little. Smaller frames compress on the compression thread alone.
ZSTD_THREADS = 4
# The slots of frames a writer holds at once: the one being filled, the one the compression thread is encoding, the one
# holding the frame it takes the next difference against, and one more queued so that a compression slower than a step
# for a while does not hold the step back. The file is written by a thread of its own, so a slow disk delays the records
# only, never the release of a slot.
N_FRAME_SLOTS = 4
# The bytes of the frames a slot holds at most, when frames are small enough for several to fit. A slot costs the
# compression thread the same work under the interpreter lock whatever its size, and the stepping thread waits on that
# lock, so small frames share their slot. The slots left to the stepping thread while the compression thread closes a
# chunk must also hold the frames of the steps that closing it takes, or a step waits for a slot.
SLOT_BYTES = 2**22
# The bytes of a frame above which CUDA takes its difference with the previous frame and gathers the blocks it changed
# on the device, a frame per slot, so that the copy carries them alone. Smaller frames cross to the host whole, for less
# than the device takes to encode them.
DEVICE_ENCODED_FRAME_BYTES = 2**19
# Whether the CUDA graphs of the frame copies record the events of the slots, which spares a launch per frame. Torch
# records an event inside a graph capture on ROCm from torch 2.13 with ROCm 7.0 on.
IS_GRAPH_EVENT_SUPPORTED = not torch.version.hip or (
    torch.torch_version.TorchVersion(torch.__version__.split("+", 1)[0]) >= (2, 13)
    and int(torch.version.hip.split(".", 1)[0]) >= 7
)
# The bytes of the completed chunks waiting for the I/O thread at most, one chunk being let through whatever its size. A
# chunk holds many frames compressed, so this absorbs a slow disk far more cheaply than frame slots would, and the
# compression thread waits once it is reached. Bytes rather than chunks, since a chunk grows with the environments.
MAX_PENDING_BYTES = 2**28
CHUNK_MAGIC = b"CHNK"
# A chunk record: frame count, payload length, payload checksum.
CHUNK_HEADER = struct.Struct("<IQI")
INFO_MAGIC = b"INFO"
# A record of the model parameters the frames from a given one on hold: that frame, payload length, payload checksum.
INFO_HEADER = struct.Struct("<QQI")
LAST_MAGIC = b"LAST"
# The record a trajectory ends with: payload length, payload checksum.
LAST_HEADER = struct.Struct("<QI")
LENGTH = struct.Struct("<Q")
# What a frame holds in each mode (see TrajectoryFile). The model parameters change through setters alone, so they go in
# a record of their own whenever one changes them (see INFO_MAGIC). A checkpoint file and the record a trajectory ends
# with hold every array but the static configs and constants, the scratch of the solvers included. They serve the
# analysis of a failure, and a failure may depend on the platform and never reproduce once shared. What the last step
# computed is therefore kept whole.
MODES_KINDS = {
    "exact": frozenset((DataKind.STATE, DataKind.DERIVED)),
    "minimal": frozenset((DataKind.STATE,)),
}
INFO_KINDS = frozenset((DataKind.INFO,))
CHECKPOINT_FILE_KINDS = frozenset(DataKind) - {DataKind.CONSTANT}
# The largest frame a mode is resolved to (see TrajectoryFile). Every recorded step copies a whole frame, and the
# compression thread goes through it, so a larger frame holds the step back once the compression falls behind.
MAX_FRAME_BYTES = 2**22
# A chunk stores each frame as its difference with the previous frame of the chunk, block-sparse: a bitmap of the blocks
# of BLOCK_SIZE bytes holding a non-zero byte, then these blocks alone. A step changes a small part of a frame, so the
# compressor goes through that part alone.
BLOCK_SIZE = 256
# The host-side end of a frame starts on a whole byte of the bitmap, so that the device encodes the rest on its own
FRAME_ALIGNMENT = 8 * BLOCK_SIZE
# The name a frame holds the step count of each environment under (see 'Simulator.data')
STEPS_FIELD = "Simulator.steps"
# The wall time of each phase of the step a frame was reached by, which closes a frame (see 'StepTimer.timings')
TIMINGS_PREFIX = "timings/"


class FrameField(NamedTuple):
    """Where one array stands in a frame: its name, kind, canonical shape, numpy dtype string, byte offset, byte length
    and memory order.

    The memory order lists the canonical axes from the slowest-varying to the fastest-varying one. A frame stores each
    array in the memory order of the scene that wrote it.
    """

    name: str
    kind: DataKind
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int
    axes: tuple[int, ...]


class SlotFrames(NamedTuple):
    """The frames of a slot handed to the compression thread: the slot and how many frames it holds."""

    i_slot: int
    n_frames: int


class FileRecord(NamedTuple):
    """A record to append to a trajectory file: its buffers in order, their byte count and the frames it holds."""

    buffers: list[bytes]
    n_bytes: int
    n_frames: int


class InfoRecord(NamedTuple):
    """The model parameters of the frames from a given one on, as a flat uint8 buffer.

    The payload is laid out as 'info_fields' states it in the manifest (see '_arrays_fields').
    """

    i_frame: int
    payload: np.ndarray


def _json_value(value):
    """Return a value as JSON holds it: the name of an enum member, a native for a numpy scalar, a list for a tuple."""
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _fields_to_json(fields: list[FrameField]) -> list:
    """Return the frame layout as the JSON rows the file header holds."""
    return [
        [field.name, field.kind.name, list(field.shape), field.dtype, field.offset, field.nbytes, list(field.axes)]
        for field in fields
    ]


def _fields_from_json(rows: list) -> list[FrameField]:
    """Return the frame layout the JSON rows of a file header hold."""
    return [
        FrameField(name, DataKind[kind], tuple(shape), dtype, offset, nbytes, tuple(axes))
        for name, kind, shape, dtype, offset, nbytes, axes in rows
    ]


def _frame_arrays(frame: np.ndarray, fields: list[FrameField], kinds: frozenset[DataKind]) -> dict[str, np.ndarray]:
    """Return the arrays of the given kinds that a frame holds, by name, shaped as the scene holds them."""
    arrays = {}
    for field in fields:
        if field.kind in kinds:
            array = frame[field.offset : field.offset + field.nbytes].view(field.dtype)
            arrays[field.name] = array.reshape([field.shape[i] for i in field.axes]).transpose(np.argsort(field.axes))
    return arrays


def _arrays_fields(sim: "Simulator", kinds: frozenset[DataKind], drop_unused: bool) -> list[FrameField]:
    """Lay out the arrays of the given kinds back to back, from the start of a frame.

    With 'drop_unused', an unused array is listed empty. Where zero-copy views exist, an array keeps the memory order
    of its view, a permutation of its canonical axes. Otherwise it is canonical.
    """
    fields = []
    offset = 0
    for item in sim.data(kinds):
        if isinstance(item, ArrayItem):
            array = data_to_array(item.array)
            # Every unused array of the kinds a step frame holds is the placeholder of a feature switched off (see
            # 'maybe_shape'), which a restore leaves as built (see 'check_data')
            if drop_unused and not item.array.shape and not isinstance(item.array, torch.Tensor):
                fields.append(FrameField(item.name, item.kind, (0,), array.dtype.str, offset, 0, (0,)))
                continue
            axes = tuple(range(array.ndim))
            # The memory order of the view, so that a frame is copied from the views as they lie, layout-tagged arrays
            # included (see 'TrajectoryFileWriter.build')
            if gs.use_zerocopy and item.array.shape:
                view = data_to_array(item.array, to_torch=True, copy=False)
                # Sorting is stable, so the axes of equal strides, which only a size-1 axis can share, keep their order
                axes = tuple(sorted(axes, key=view.stride, reverse=True))
                assert view.permute(axes).is_contiguous()
            fields.append(FrameField(item.name, item.kind, array.shape, array.dtype.str, offset, array.nbytes, axes))
            offset += array.nbytes
    return fields


def _scalars_fields(sim: "Simulator", kinds: frozenset[DataKind], offset: int) -> list[FrameField]:
    """Lay out the scalars of the given kinds back to back, from the given byte offset (see 'ScalarItem')."""
    fields = []
    for item in sim.data(kinds):
        if isinstance(item, ScalarItem):
            dtype = np.dtype(type(getattr(item.owner, item.attr)))
            fields.append(FrameField(item.name, item.kind, (), dtype.str, offset, dtype.itemsize, ()))
            offset += dtype.itemsize
    return fields


def _write_scalars(scalars: list[ScalarItem], scalars_fields: list[FrameField], buffer: np.ndarray) -> None:
    """Write the given scalars into a flat uint8 buffer, where their fields lay them out.

    A frame is laid out once, so a scalar whose type changed since raises.
    """
    for item, field in zip(scalars, scalars_fields):
        scalar = np.array(getattr(item.owner, item.attr))
        if scalar.dtype.str != field.dtype:
            gs.raise_exception(
                f"'{field.name}' holds a {scalar.dtype} value, where a {np.dtype(field.dtype)} one was laid out."
            )
        buffer[field.offset : field.offset + field.nbytes] = scalar.reshape(-1).view(np.uint8)


def _frame_fields(sim: "Simulator", kinds: frozenset[DataKind], drop_unused: bool) -> tuple[list[FrameField], int]:
    """Lay out one frame of the given kinds (see '_read_frame'), and return its fields and its size.

    A frame holds every array (see '_arrays_fields'), the step count of each environment included, then the scalars
    (see 'ScalarItem') and the wall time of every phase of a step. The scalars and the timings start on a multiple of
    FRAME_ALIGNMENT, and so does the size of the frame. The bytes in between are zeros.
    """
    fields = _arrays_fields(sim, kinds, drop_unused)
    offset = sum(field.nbytes for field in fields)
    offset = -(-offset // FRAME_ALIGNMENT) * FRAME_ALIGNMENT
    scalars_fields = _scalars_fields(sim, kinds, offset)
    fields += scalars_fields
    offset += sum(field.nbytes for field in scalars_fields)
    for phase in sim.step_timer.phases:
        fields.append(FrameField(f"{TIMINGS_PREFIX}{phase}", DataKind.STATE, (), "<f8", offset, 8, ()))
        offset += 8
    return fields, -(-offset // FRAME_ALIGNMENT) * FRAME_ALIGNMENT


def _write_frame_tail(
    sim: "Simulator",
    scalars: list[ScalarItem],
    scalars_fields: list[FrameField],
    timings_field: FrameField,
    frame: np.ndarray,
) -> None:
    """Write into a frame, a flat uint8 buffer, its end that lives on the host.

    It holds the given scalars, where their fields lay them out, then from 'timings_field' on the wall time of every
    phase of the step the state was reached by, NaN for a phase that step did not time.
    """
    phases = sim.step_timer.phases
    step_timings = sim.step_timer.timings
    # A phase timed without being declared would be lost, as the frame is laid out once (see 'StepTimer.add_phases')
    assert step_timings.keys() <= set(phases)
    _write_scalars(scalars, scalars_fields, frame)
    timings = frame[timings_field.offset : timings_field.offset + timings_field.nbytes * len(phases)]
    timings.view(timings_field.dtype)[:] = [step_timings.get(phase, np.nan) for phase in phases]


def _read_arrays(arrays: dict[str, ArrayItem], fields: list[FrameField]) -> np.ndarray:
    """Return the given arrays as a flat uint8 buffer, laid out as their fields state (see '_arrays_fields')."""
    parts = [(arrays[field.name].array, field.axes) for field in fields if field.name in arrays and field.nbytes]
    if not parts:
        return np.empty((0,), dtype=np.uint8)
    # Where zero-copy views exist the arrays are concatenated on the device and cross to the host once. Otherwise each
    # array crosses on its own, to be concatenated on the host.
    if gs.use_zerocopy:
        views = [
            data_to_array(array, to_torch=True).permute(axes).reshape(-1).view(torch.uint8) for array, axes in parts
        ]
        return tensor_to_array(torch.cat(views))
    return np.concatenate([data_to_array(array).transpose(axes).reshape(-1).view(np.uint8) for array, axes in parts])


def _read_frame(sim: "Simulator", kinds: frozenset[DataKind], fields: list[FrameField], frame_size: int) -> np.ndarray:
    """Return one frame of the given kinds as a flat uint8 buffer laid out as 'fields' states it.

    See '_frame_fields'.
    """
    items = list(sim.data(kinds))
    arrays = {item.name: item for item in items if isinstance(item, ArrayItem)}
    scalars = [item for item in items if isinstance(item, ScalarItem)]
    # Every array of the given kinds has a field, and these fields come first, followed by the ones of the scalars in
    # the order of their items
    device_part = _read_arrays(arrays, fields[: len(arrays)])
    frame = np.zeros((frame_size,), dtype=np.uint8)
    frame[: device_part.size] = device_part
    scalars_fields = fields[len(arrays) : len(arrays) + len(scalars)]
    _write_frame_tail(sim, scalars, scalars_fields, fields[len(arrays) + len(scalars)], frame)
    return frame


def _encode_device_frame(
    frame_views: list[torch.Tensor],
    staging: torch.Tensor,
    device_end: int,
    i_staging: int,
    is_delta: bool,
    blocks: torch.Tensor,
    is_block_changed: torch.Tensor,
    event: torch.cuda.Event | None,
) -> None:
    """Gather on the device the blocks of a frame that hold a non-zero byte, of its difference with the previous frame.

    The frame is concatenated from its views into the buffer 'i_staging' of the two in 'staging'. With 'is_delta', the
    other buffer, which holds the previous frame, takes the difference. 'blocks', as large as the frame, receives the
    changed blocks at its front, in order, and 'is_block_changed' which blocks they are. 'event', unless None, is
    recorded once both have landed.
    """
    frame = staging[i_staging]
    torch.cat(frame_views, out=frame[:device_end])
    if is_delta:
        frame_prev = staging[1 - i_staging]
        frame = torch.bitwise_xor(frame_prev, frame, out=frame_prev)
    frame_blocks = frame.view(torch.int64).view((-1, BLOCK_SIZE // 8))
    is_frame_block_changed = frame_blocks.any(dim=1)
    # The indices of the changed blocks are padded with the first block up to the number of blocks, so that nothing
    # waits on the device for their count and a CUDA graph can capture the whole. Only the leading gathered blocks are
    # changed ones, as many as the mask marks.
    blocks_idx = torch.nonzero_static(is_frame_block_changed, size=frame_blocks.shape[0], fill_value=0)[:, 0]
    torch.index_select(frame_blocks, 0, blocks_idx, out=blocks)
    is_block_changed.copy_(is_frame_block_changed, non_blocking=True)
    if event is not None:
        event.record()


def _copy_device_frame(
    frame_views: list[torch.Tensor],
    staging: torch.Tensor,
    device_end: int,
    slot: torch.Tensor,
    event: torch.cuda.Event | None,
) -> None:
    """Concatenate a frame on the device from its views, then copy it to the host without waiting for it to land.

    'staging' receives the concatenated frame and 'slot', page-locked, its copy. 'event', unless None, is recorded once
    the copy has landed.
    """
    torch.cat(frame_views, out=staging[:device_end])
    slot.copy_(staging, non_blocking=True)
    if event is not None:
        event.record()


@nb.jit(nopython=True, cache=True, nogil=True)
def _encode_frames(
    frames: np.ndarray, frame_prev: np.ndarray, is_delta: bool, encoded: np.ndarray, blocks: np.ndarray
) -> int:
    """Encode consecutive frames block-sparse (see BLOCK_SIZE), each as its difference with the one before it.

    'frames' holds the frames as rows of bytes, each a whole number of bitmap bytes long (see FRAME_ALIGNMENT), and
    'frame_prev' the frame before the first, whose difference the first takes if 'is_delta' is set. Write into
    'encoded' the frames back to back, each as a bitmap of one bit per block, set if it holds a non-zero byte, the
    lowest bit first, then these blocks. 'blocks' is a uint64 scratch the size of a frame. Return the bytes written.
    """
    n_words = BLOCK_SIZE // 8
    n_bitmap_bytes = frames.shape[1] // FRAME_ALIGNMENT
    blocks_bytes = blocks.view(np.uint8)
    i_encoded = 0
    for i_frame in range(frames.shape[0]):
        frame_ref = frame_prev if i_frame == 0 else frames[i_frame - 1]
        # The previous frame drops out of the difference of a frame stored whole
        words_ref_mask = np.uint64(0xFFFFFFFFFFFFFFFF) if is_delta or i_frame > 0 else np.uint64(0)
        words = frames[i_frame].view(np.uint64)
        words_ref = frame_ref.view(np.uint64)
        n_blocks = 0
        for i_byte in range(n_bitmap_bytes):
            byte = 0
            for i_bit in range(8):
                i_word_start = (8 * i_byte + i_bit) * n_words
                # Most blocks of a frame are unchanged, so a block is reduced first and copied only if it holds a
                # non-zero word, which spares the writes of the others
                words_or = np.uint64(0)
                for i_word in range(i_word_start, i_word_start + n_words):
                    words_or |= words[i_word] ^ (words_ref[i_word] & words_ref_mask)
                if words_or != 0:
                    for i_word in range(n_words):
                        i_word_frame = i_word_start + i_word
                        word = words[i_word_frame] ^ (words_ref[i_word_frame] & words_ref_mask)
                        blocks[n_blocks * n_words + i_word] = word
                    byte |= 1 << i_bit
                    n_blocks += 1
            encoded[i_encoded + i_byte] = byte
        # The blocks follow the bitmap at any byte, so they are gathered as aligned words first, then copied
        i_encoded += n_bitmap_bytes
        n_blocks_bytes = n_blocks * BLOCK_SIZE
        encoded[i_encoded : i_encoded + n_blocks_bytes] = blocks_bytes[:n_blocks_bytes]
        i_encoded += n_blocks_bytes
    return i_encoded


def _header(
    source: "TrajectorySource",
    mode: str,
    chunk_size: int,
    fields: list[FrameField],
    frame_size: int,
    info_fields: list[FrameField],
    last_fields: list[FrameField],
    last_frame_size: int,
) -> bytes:
    """Return the magic, the JSON manifest and the scene description that open a trajectory file.

    'fields' lays out its frames of 'frame_size' bytes, 'info_fields' its records of model parameters and 'last_fields'
    the record of 'last_frame_size' bytes it ends with.
    """
    sim, desc, layout = source
    description = io.BytesIO()
    serialization.export(description, {"scene": desc})
    manifest = {
        "genesis": gs.__version__,
        "source": serialization.source_digest(),
        "backend": gs.backend.name,
        "precision": np.dtype(gs.np_float).name,
        "mode": mode,
        "chunk_size": chunk_size,
        "dt": sim.dt,
        "layout": {key: _json_value(value) for key, value in dataclasses.asdict(layout).items()},
        "configs": {
            item.name: {key: _json_value(value) for key, value in vars(item.config).items() if not key.startswith("_")}
            for solver in sim.active_solvers
            for item in solver.data
            if isinstance(item, ConfigItem)
        },
        "fields": _fields_to_json(fields),
        "frame_size": frame_size,
        "info_fields": _fields_to_json(info_fields),
        "last_fields": _fields_to_json(last_fields),
        "last_frame_size": last_frame_size,
    }
    text = json.dumps(manifest).encode()
    return MAGIC + LENGTH.pack(len(text)) + text + LENGTH.pack(description.getbuffer().nbytes) + description.getvalue()


def save_checkpoint(path: str | os.PathLike, source: "TrajectorySource") -> None:
    """Write the whole state of a built scene as a trajectory file of one frame, the record a trajectory ends with (see
    'Scene.save_checkpoint').
    """
    fields, frame_size = _frame_fields(source.sim, CHECKPOINT_FILE_KINDS, drop_unused=False)
    with open(path, "wb") as file:
        file.write(
            _header(
                source,
                mode="exact",
                chunk_size=1,
                fields=fields,
                frame_size=frame_size,
                info_fields=[],
                last_fields=fields,
                last_frame_size=frame_size,
            )
        )
        payload = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            _read_frame(source.sim, CHECKPOINT_FILE_KINDS, fields, frame_size)
        )
        file.write(LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload)) + payload)


@register_recording(TrajectoryFile)
class TrajectoryFileWriter(BaseFileWriter):
    """Write the frames of a scene to a trajectory file (see TrajectoryFile), one chunk at a time.

    Each frame captures the state a step starts from, control inputs included, since it is read before the step. The
    stepping thread copies each frame into one of a few slots allocated at build (see SLOT_BYTES) and hands full slots
    to a compression thread, which streams their frames through the compressor as their XOR with the previous frame of
    their chunk. Unchanged bytes thus become zeros the compressor removes. A completed chunk goes to a third thread,
    which appends it to the file. A step costs one copy of the frame, issued without waiting for a GPU to finish the
    step: the compression thread waits for the copy to land instead. The step waits only when every slot is taken,
    which happens when the frames are compressed slower than the scene steps. The model parameters change through
    setters alone, so they are recorded ahead of the first frame and of the first frame after a setter changed them, in
    a record of their own.
    """

    def build(self):
        self._source: TrajectorySource = self._data_func()
        sim = self._source.sim
        # The frames of the steps leave the unused arrays out, as they are read once per step. The last record holds
        # them, with the scratch.
        # The exact frame holds the kinds of every mode, so it sizes them all.
        exact_fields, _ = _frame_fields(sim, MODES_KINDS["exact"], drop_unused=True)
        modes_frame_size = {
            mode: sum(field.nbytes for field in exact_fields if field.kind in kinds)
            for mode, kinds in MODES_KINDS.items()
        }
        self._mode = self._options.mode
        if self._mode is None:
            self._mode = "exact" if modes_frame_size["exact"] <= MAX_FRAME_BYTES else "minimal"
        elif self._mode == "exact" and modes_frame_size["exact"] > MAX_FRAME_BYTES:
            gs.logger.warning(
                f"[TrajectoryFileWriter] Recording in exact mode copies {modes_frame_size['exact'] / 1e6:.1f} MB at "
                f"every recorded step, which slows the simulation down a lot. A minimal frame holds "
                f"{modes_frame_size['minimal'] / 1e6:.1f} MB (see TrajectoryFile)."
            )
        self._kinds = MODES_KINDS[self._mode]
        self._fields, self._frame_size = _frame_fields(sim, self._kinds, drop_unused=True)
        self._last_fields, self._last_frame_size = _frame_fields(sim, CHECKPOINT_FILE_KINDS, drop_unused=False)
        # The model parameters are recorded with the first frame, then with the first frame after a setter changes them
        self._info_fields = _arrays_fields(sim, INFO_KINDS, drop_unused=True)
        self._info_scalars_fields = _scalars_fields(sim, INFO_KINDS, sum(field.nbytes for field in self._info_fields))
        self._info_fields += self._info_scalars_fields
        # The data of the solvers is walked once, as walking it costs more than a step of a small scene
        info_items = list(sim.data(INFO_KINDS))
        self._info_arrays = {item.name: item for item in info_items if isinstance(item, ArrayItem)}
        self._info_scalars = [item for item in info_items if isinstance(item, ScalarItem)]
        self._info_size = sum(field.nbytes for field in self._info_fields)
        self._has_info = self._info_size > 0
        self._info_subscriber = Subscriber(frozenset((StateChange.MODEL,)))
        for solver in sim.active_solvers:
            solver.subscribe(self._info_subscriber)
        self._i_frame = 0
        self._steps_field = next(field for field in self._fields if field.name == STEPS_FIELD)
        # The scalars and the step timings close the frame (see '_frame_fields'). They live on the host, so a frame copy
        # stops short of them and they are written beside it.
        items = list(sim.data(self._kinds))
        arrays = {item.name: item.array for item in items if isinstance(item, ArrayItem)}
        self._device_end = sum(field.nbytes for field in self._fields if field.name in arrays)
        self._tail_offset = self._fields[len(arrays)].offset
        self._scalars = [item for item in items if isinstance(item, ScalarItem)]
        self._scalars_fields = self._fields[len(arrays) : len(arrays) + len(self._scalars)]
        self._timings_field = self._fields[len(arrays) + len(self._scalars)]

        # The arrays of a built scene keep their memory, so the byte view each array is copied from is taken once, in
        # the memory order the frame stores it in (see '_frame_fields'). Without zero-copy views each frame is read
        # whole (see '_read_frame').
        # FIXME: quadrants#970 - a 0-d field has no zero-copy view, and reading one per frame runs a copy kernel. The
        # step frames leave out the 0-d arrays (see '_frame_fields'), whose content the step frames therefore lack.
        self._frame_views: list[np.ndarray | torch.Tensor] | None = None
        self._staging: torch.Tensor | None = None
        # See DEVICE_ENCODED_FRAME_BYTES
        self._is_device_encoded = (
            gs.use_zerocopy and gs.device.type == "cuda" and self._frame_size > DEVICE_ENCODED_FRAME_BYTES
        )
        # A slot holds several frames when they are small, at most a chunk of them, and a slot is handed over at the end
        # of a chunk (see '_get_frame'), so that its frames all belong to one chunk
        self._n_slot_frames = 1
        if not self._is_device_encoded:
            self._n_slot_frames = max(1, min(self._options.chunk_size, SLOT_BYTES // self._frame_size))
        self._device_blocks: torch.Tensor | None = None
        self._device_blocks_bytes: list[torch.Tensor] = []
        self._slots_mask_tc: torch.Tensor | None = None
        self._slots_mask: np.ndarray | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._frame_graphs: dict[tuple[int, ...], torch.cuda.CUDAGraph] = {}
        if gs.use_zerocopy:
            self._frame_views = []
            for field in self._fields[: len(arrays)]:
                if field.nbytes:
                    array = data_to_array(arrays[field.name], to_torch=True, copy=False)
                    view = array.permute(field.axes).view(-1).view(torch.uint8)
                    # The views of the CPU backend are host memory already, and numpy concatenates them on the calling
                    # thread. Torch would run the copy on its thread pool, whose wake-up costs more than the copy.
                    self._frame_views.append(view.numpy() if gs.backend == gs.cpu else view)
            if gs.backend != gs.cpu:
                # The frame on its way to the host, or the two latest frames in turn when the device encodes them, the
                # one before the latest taking its difference with it (see '_get_frame'). The bytes between the arrays
                # and the scalars stay zeros.
                n_staging = 2 if self._is_device_encoded else 1
                self._staging = torch.zeros((n_staging, self._tail_offset), dtype=torch.uint8, device=gs.device)
        # Page-locked on CUDA, so that the copy of a frame from the device runs without blocking the stepping thread.
        # The bytes between the arrays and the scalars stay zeros.
        slots = torch.zeros(
            (N_FRAME_SLOTS, self._n_slot_frames, self._frame_size),
            dtype=torch.uint8,
            pin_memory=gs.device.type == "cuda",
        )
        self._slots_tc = slots
        self._slots = slots.numpy()
        if self._is_device_encoded:
            # The changed blocks of each frame, gathered on the device, and which blocks they are on the host
            n_device_blocks = self._tail_offset // BLOCK_SIZE
            self._device_blocks = torch.empty(
                (N_FRAME_SLOTS, n_device_blocks, BLOCK_SIZE // 8), dtype=torch.int64, device=gs.device
            )
            # The bytes of each slot, which the compression thread copies (see '_compress_frames')
            self._device_blocks_bytes = [blocks.view(-1).view(torch.uint8) for blocks in self._device_blocks]
            slots_mask = torch.empty((N_FRAME_SLOTS, n_device_blocks), dtype=torch.bool, pin_memory=True)
            self._slots_mask_tc = slots_mask
            self._slots_mask = slots_mask.numpy()
            self._copy_stream = torch.cuda.Stream()
        # The compression thread waits on the event of a slot for the copies of its frames to land (see
        # '_compress_frames'). The frame graphs record it where they can, which takes external events for the host to
        # see the record of each replay (see IS_GRAPH_EVENT_SUPPORTED).
        self._is_slot_event_captured = gs.use_zerocopy and gs.device.type == "cuda" and IS_GRAPH_EVENT_SUPPORTED
        self._slots_event = [
            torch.cuda.Event(external=self._is_slot_event_captured) if gs.device.type == "cuda" else None
            for _ in range(N_FRAME_SLOTS)
        ]
        slots_event_captured = self._slots_event if self._is_slot_event_captured else [None] * N_FRAME_SLOTS
        if gs.use_zerocopy and gs.device.type == "cuda":
            # Launching each operation of a frame copy costs more host time than the device takes to run them all, so
            # they are captured once in a CUDA graph per place of the frame in the slots, and per staging buffer and
            # encoding when the device encodes it, a frame launching one. The graphs run one after the other, so they
            # share their memory pool. A first run outside capture lets torch initialize what the operations need.
            if self._is_device_encoded:
                captures = {
                    (i_slot, i_staging, is_delta): functools.partial(
                        _encode_device_frame,
                        self._frame_views,
                        self._staging,
                        self._device_end,
                        i_staging,
                        is_delta,
                        self._device_blocks[i_slot],
                        self._slots_mask_tc[i_slot],
                        slots_event_captured[i_slot],
                    )
                    for i_slot, i_staging, is_delta in itertools.product(range(N_FRAME_SLOTS), range(2), (False, True))
                }
            else:
                captures = {
                    (i_slot, i_slot_frame): functools.partial(
                        _copy_device_frame,
                        self._frame_views,
                        self._staging[0],
                        self._device_end,
                        self._slots_tc[i_slot, i_slot_frame, : self._tail_offset],
                        slots_event_captured[i_slot],
                    )
                    for i_slot, i_slot_frame in itertools.product(range(N_FRAME_SLOTS), range(self._n_slot_frames))
                }
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            graphs_pool = torch.cuda.graph_pool_handle()
            for key, capture in captures.items():
                with torch.cuda.stream(warmup_stream):
                    capture()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=graphs_pool):
                    capture()
                self._frame_graphs[key] = graph

        # The block-sparse encoding of the frames of a slot and its scratch (see '_encode_frames'), of the host-side end
        # of the frame alone when the device encodes the rest
        encoded_size = self._frame_size - (self._tail_offset if self._is_device_encoded else 0)
        self._encoded = np.empty((self._n_slot_frames * (encoded_size // FRAME_ALIGNMENT + encoded_size),), np.uint8)
        self._encoded_blocks = np.empty((encoded_size // 8,), dtype=np.uint64)
        self._free_slots: queue.SimpleQueue[int] = queue.SimpleQueue()
        for i_slot in range(N_FRAME_SLOTS):
            self._free_slots.put(i_slot)
        # The slot being filled, its frame count, and whether it takes no more frames
        self._i_slot: int | None = None
        self._n_slot_filled = 0
        self._is_slot_full = False
        # The slots handed to the compression thread in step order, each preceded by the model parameters of its frames
        # that changed, then None at the stop
        self._frames: queue.SimpleQueue[SlotFrames | InfoRecord | None] = queue.SimpleQueue()

        # The chunk the compression thread is filling: its compressor, its compressed pieces and its frame count
        self._compressor = zstandard.ZstdCompressor(
            level=ZSTD_LEVEL, threads=ZSTD_THREADS if self._is_device_encoded else 0
        )
        self._info_compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._chunk_stream = self._compressor.compressobj()
        self._chunk_pieces: list[bytes] = []
        # The records of model parameters for the frames of the open chunk, which go to the file ahead of it
        self._info_records: list[bytes] = []
        self._n_chunk_frames = 0
        self._n_frames = 0
        self._written = 0
        self._fault: str | None = None
        self._error: Exception | None = None
        # The completed chunks handed to the I/O thread, in order, then None at the stop
        self._records: queue.SimpleQueue[FileRecord | None] = queue.SimpleQueue()
        self._pending_bytes = 0
        self._pending_bytes_changed = threading.Condition()
        self._compression_thread: threading.Thread | None = None
        self._io_thread: threading.Thread | None = None
        self._file: io.BufferedWriter | None = None

        super().build()

    def _initialize_writer(self):
        # The header goes out with the first record. A run that ends before any leaves an empty file, which no reader
        # takes for a trajectory.
        self._file = open(self._get_filename(), "wb")
        self._header = _header(
            self._source,
            self._mode,
            self._options.chunk_size,
            self._fields,
            self._frame_size,
            self._info_fields,
            self._last_fields,
            self._last_frame_size,
        )
        self._written = 0
        self._compression_thread = threading.Thread(target=self._compress_frames, daemon=True)
        self._compression_thread.start()
        self._io_thread = threading.Thread(target=self._write_records, daemon=True)
        self._io_thread.start()

    def _get_frame(self):
        """Copy the frame of the current state into the slot being filled and return the slot.

        The model parameters go to the compression thread ahead of the slot of the frame, when it is the first or a
        setter changed them since the previous one.
        """
        sim = self._source.sim
        # A full slot goes to the compression thread when the next frame comes, so that the stop can drop the frame it
        # recorded (see 'cleanup')
        if self._is_slot_full:
            self._frames.put(SlotFrames(self._i_slot, self._n_slot_filled))
            self._i_slot = None
        if self._i_slot is None:
            self._i_slot = self._free_slots.get()
            self._n_slot_filled = 0
        i_slot = self._i_slot
        slot = self._slots[i_slot, self._n_slot_filled]
        # On Metal, the arrays are written by a queue torch does not wait for
        if self._frame_views is not None and gs.backend == gs.metal:
            qd.sync()
        if self._has_info and (self._i_frame == 0 or self._info_subscriber.pending):
            self._info_subscriber.clear()
            info_arrays = _read_arrays(self._info_arrays, self._info_fields)
            info = np.empty((self._info_size,), dtype=np.uint8)
            info[: info_arrays.size] = info_arrays
            _write_scalars(self._info_scalars, self._info_scalars_fields, info)
            self._frames.put(InfoRecord(self._i_frame, info))
        if self._frame_views is not None and gs.backend == gs.cpu:
            np.concatenate(self._frame_views, out=slot[: self._device_end])
        elif self._is_device_encoded:
            # The first frame of a chunk is stored whole, the others as their difference with the previous frame (see
            # '_compress_frames'). The buffer of the previous frame takes the difference, and the next frame is
            # concatenated into it once the work queued before has read it. The graph is launched behind the step on
            # the stream both run on, and the compression thread waits for it to land (see '_compress_frames').
            is_delta = self._i_frame % self._options.chunk_size != 0
            self._frame_graphs[(i_slot, self._i_frame % 2, is_delta)].replay()
        elif self._frame_views is not None and gs.device.type == "cuda":
            # The copy runs behind the step on the stream both run on, and the next frame is concatenated once it has
            # read the staging buffer
            self._frame_graphs[(i_slot, self._n_slot_filled)].replay()
        elif self._frame_views is not None:
            frame = self._staging[0]
            torch.cat(self._frame_views, out=frame[: self._device_end])
            self._slots_tc[i_slot, self._n_slot_filled, : self._tail_offset].copy_(frame)
        else:
            frame = _read_frame(sim, self._kinds, self._fields, self._frame_size)
            slot[: self._tail_offset] = frame[: self._tail_offset]
        _write_frame_tail(sim, self._scalars, self._scalars_fields, self._timings_field, slot)
        self._i_frame += 1
        self._n_slot_filled += 1
        self._is_slot_full = self._n_slot_filled == self._n_slot_frames or self._i_frame % self._options.chunk_size == 0
        if self._is_slot_full and self._slots_event[i_slot] is not None and not self._is_slot_event_captured:
            self._slots_event[i_slot].record()
        return i_slot

    def step(self, global_step: int):
        # A failed write ends the recorder, as a failed sample does (see 'Recorder.record')
        if self._fault is not None:
            self._is_alive = False
            fault, self._fault = self._fault, None
            gs.raise_exception(fault)
        if self._error is not None:
            self._is_alive = False
            error, self._error = self._error, None
            gs.raise_exception_from(f"[TrajectoryFileWriter] Writing '{self._get_filename()}' failed.", error)
        super().step(global_step)

    def process(self, data, cur_time):
        # The frames reach the compression thread by slot (see '_get_frame')
        pass

    def _compress_frames(self):
        """Append the frames of the queued slots to the open chunk in step order, on the compression thread.

        A failure is raised on the stepping thread (see 'step'), and the slots that follow are released unwritten so the
        stepping thread never waits on one.
        """
        # The changed blocks are copied on a stream of their own, made current for this thread once, as switching to it
        # for each frame costs more host time than the copy
        if self._is_device_encoded:
            torch.cuda.set_stream(self._copy_stream)
        i_slot_prev: int | None = None
        n_frames_prev = 0
        while True:
            item = self._frames.get()
            if item is None:
                break
            if isinstance(item, InfoRecord):
                if self._error is None and self._fault is None:
                    try:
                        payload = self._info_compressor.compress(item.payload)
                        record = INFO_MAGIC + INFO_HEADER.pack(item.i_frame, len(payload), zlib.crc32(payload))
                        self._info_records.append(record + payload)
                    except Exception as error:
                        self._error = error
                continue
            i_slot, n_frames = item
            # A failure of the copy surfaces here, and is reported like a failure of the compression
            if self._slots_event[i_slot] is not None and self._error is None:
                try:
                    self._slots_event[i_slot].synchronize()
                except Exception as error:
                    self._error = error
            if self._error is None and self._fault is None:
                try:
                    # The first frame of a chunk is stored whole, the others as their difference with the previous
                    # frame, block-sparse either way (see BLOCK_SIZE)
                    is_delta = self._n_chunk_frames > 0
                    frames = self._slots[i_slot, :n_frames]
                    frame_prev = self._slots[i_slot_prev, n_frames_prev - 1] if is_delta else frames[0]
                    if self._is_device_encoded:
                        # The device took the difference of the part before the scalars and gathered its changed
                        # blocks (see '_get_frame'), so only these cross to the host
                        frame = frames[0]
                        is_block_changed = self._slots_mask[i_slot]
                        n_device_bytes = np.count_nonzero(is_block_changed) * BLOCK_SIZE
                        blocks = self._device_blocks_bytes[i_slot][:n_device_bytes]
                        self._slots_tc[i_slot, 0, :n_device_bytes].copy_(blocks, non_blocking=True)
                        self._copy_stream.synchronize()
                        n_encoded = _encode_frames(
                            frame[np.newaxis, self._tail_offset :],
                            frame_prev[self._tail_offset :],
                            is_delta,
                            self._encoded,
                            self._encoded_blocks,
                        )
                        # The bitmap of the whole frame precedes its blocks
                        n_tail_bitmap_bytes = (self._frame_size - self._tail_offset) // FRAME_ALIGNMENT
                        bitmap = np.packbits(is_block_changed, bitorder="little")
                        self._chunk_pieces.append(self._chunk_stream.compress(bitmap))
                        self._chunk_pieces.append(self._chunk_stream.compress(self._encoded[:n_tail_bitmap_bytes]))
                        self._chunk_pieces.append(self._chunk_stream.compress(frame[:n_device_bytes]))
                        encoded = self._encoded[n_tail_bitmap_bytes:n_encoded]
                    else:
                        n_encoded = _encode_frames(frames, frame_prev, is_delta, self._encoded, self._encoded_blocks)
                        encoded = self._encoded[:n_encoded]
                    self._chunk_pieces.append(self._chunk_stream.compress(encoded))
                    self._n_chunk_frames += n_frames
                    if self._n_chunk_frames == self._options.chunk_size:
                        self._close_chunk()
                except Exception as error:
                    self._error = error
            # The slot holding the previous frame is kept until the next slot takes its difference with it
            if i_slot_prev is not None:
                self._free_slots.put(i_slot_prev)
            i_slot_prev, n_frames_prev = i_slot, n_frames
        self._close_chunk()
        if i_slot_prev is not None:
            self._free_slots.put(i_slot_prev)

    def _close_chunk(self):
        """Close the open chunk and hand it to the I/O thread, on the compression thread."""
        if self._n_chunk_frames == 0 or self._error is not None or self._fault is not None:
            return
        try:
            self._chunk_pieces.append(self._chunk_stream.flush())
            payload_crc = 0
            for piece in self._chunk_pieces:
                payload_crc = zlib.crc32(piece, payload_crc)
            payload_size = sum(len(piece) for piece in self._chunk_pieces)
            chunk_header = CHUNK_MAGIC + CHUNK_HEADER.pack(self._n_chunk_frames, payload_size, payload_crc)
            # The records of model parameters go ahead of the chunk whose frames hold them, in one record, so that a
            # file closed at its max_size never ends on parameters without their frames. The pieces are written one
            # after the other, as joining them would copy the chunk while holding the interpreter lock the stepping
            # thread waits on.
            buffers = [*self._info_records, chunk_header, *self._chunk_pieces]
            record = FileRecord(buffers, sum(len(buffer) for buffer in buffers), self._n_chunk_frames)
            self._info_records.clear()
            with self._pending_bytes_changed:
                self._pending_bytes_changed.wait_for(
                    lambda: self._pending_bytes == 0 or self._pending_bytes + record.n_bytes <= MAX_PENDING_BYTES
                )
                self._pending_bytes += record.n_bytes
            self._records.put(record)
        except Exception as error:
            self._error = error
        self._chunk_pieces.clear()
        self._chunk_stream = self._compressor.compressobj()
        self._n_chunk_frames = 0

    def _write_records(self):
        """Append the completed chunks to the file in order, on the I/O thread.

        A failure is raised on the stepping thread (see 'step'), and the chunks that follow are dropped.
        """
        while True:
            record = self._records.get()
            if record is None:
                break
            if self._error is None and self._file is not None:
                try:
                    self._write(record)
                except Exception as error:
                    self._error = error
            with self._pending_bytes_changed:
                self._pending_bytes -= record.n_bytes
                self._pending_bytes_changed.notify()

    def _write(self, record: FileRecord):
        """Append one record to the file, with the header ahead of the first, if the file may still grow by that much.

        A record the file cannot hold closes it and records the fault.
        """
        n_bytes = len(self._header) + record.n_bytes
        if self._options.max_size is not None and self._written + n_bytes > self._options.max_size:
            self._file.close()
            self._file = None
            if self._written == 0:
                # A limit below the header and the first record leaves nothing to read, so the empty file goes too
                os.remove(self._get_filename())
                self._fault = (
                    f"The max_size of {self._options.max_size} bytes cannot hold the header and the first record of "
                    f"'{self._get_filename()}'."
                )
            else:
                self._fault = (
                    f"'{self._get_filename()}' reached its max_size of {self._options.max_size} bytes after "
                    f"{self._n_frames} frames, and was closed."
                )
            return
        # The header holds the description of the scene, megabytes with its meshes, so it is written on its own too
        self._file.write(self._header)
        for buffer in record.buffers:
            self._file.write(buffer)
        self._file.flush()
        self._header = b""
        self._written += n_bytes
        self._n_frames += record.n_frames

    def cleanup(self):
        sim = self._source.sim
        if self._compression_thread is not None:
            # The frame the stop recorded carries the step counts of the live state. The whole state ends the file in
            # its place (see CHECKPOINT_FILE_KINDS), so it is dropped. It is the last of the slot being filled, which
            # the compression thread has not been handed yet.
            if self._i_slot is not None:
                if self._slots_event[self._i_slot] is not None:
                    self._slots_event[self._i_slot].record()
                    self._slots_event[self._i_slot].synchronize()
                field = self._steps_field
                frame = self._slots[self._i_slot, self._n_slot_filled - 1]
                # The slot holds the changed blocks when the device encodes the frame, which keeps the last frame
                if self._is_device_encoded:
                    frame = tensor_to_array(self._staging[(self._i_frame - 1) % 2])
                steps = frame[field.offset : field.offset + field.nbytes].view(field.dtype).reshape(field.shape)
                n_frames = self._n_slot_filled - np.array_equal(steps, tensor_to_array(sim.steps))
                if n_frames > 0:
                    self._frames.put(SlotFrames(self._i_slot, n_frames))
                else:
                    self._free_slots.put(self._i_slot)
                self._i_slot = None
                self._is_slot_full = False
            self._frames.put(None)
            self._compression_thread.join()
            self._compression_thread = None
            self._records.put(None)
            self._io_thread.join()
            self._io_thread = None
        try:
            if self._error is None and self._file is not None:
                payload = self._compressor.compress(
                    _read_frame(sim, CHECKPOINT_FILE_KINDS, self._last_fields, self._last_frame_size)
                )
                last_header = LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload))
                self._write(FileRecord([last_header, payload], len(last_header) + len(payload), 1))
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
        # No step follows to raise what the last writes left, so it is raised here, the file closed
        if self._error is not None:
            gs.raise_exception_from(f"[TrajectoryFileWriter] Writing '{self._get_filename()}' failed.", self._error)
        if self._fault is not None:
            gs.raise_exception(self._fault)
        (gs.logger or LOGGER).info(
            f'[TrajectoryFileWriter] Saved {self._n_frames} frames to ~<"{self._get_filename()}">~.'
        )

    @property
    def run_in_thread(self) -> bool:
        return False


class Trajectory:
    """A recorded trajectory and the built scene it plays in (see 'Scene.load_trajectory').

    Frame i is the state the scene stood in after i steps, and the last frame is the state the recording stopped at,
    whole: every array but the static configs and constants (see CHECKPOINT_FILE_KINDS).

    Parameters
    ----------
    path : str or os.PathLike
        The file to read, as written by recording with 'TrajectoryFile' or by 'Scene.save_checkpoint'.
    scene : Scene, optional
        A built scene to play the trajectory in, from the same description and environment layout as the recorded
        one. If None, the recorded scene is created and built. Defaults to None.
    show_viewer : bool, optional
        Whether the created scene opens an interactive viewer. Defaults to False.
    viewer_options : ViewerOptions, optional
        Viewer options overriding the recorded ones in the created scene, for the fields they set.
    vis_options : VisOptions, optional
        Visualizer options overriding the recorded ones in the created scene, for the fields they set.
    renderer : RendererOptions, optional
        Renderer replacing the recorded one in the created scene. If None, the recorded one stands.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        scene: "Scene | None" = None,
        show_viewer: bool = False,
        viewer_options: "ViewerOptions | None" = None,
        vis_options: "VisOptions | None" = None,
        renderer: "RendererOptions | None" = None,
    ):
        # The scene module imports the recorders, so its classes are reached here rather than at import.
        from genesis.engine.scene import EnvironmentLayout, SceneCheckpoint, SceneDescription, description_digest

        self._path = path
        with open(path, "rb") as file:
            magic = file.read(len(MAGIC))
            if magic != MAGIC:
                if magic.startswith(MAGIC_PREFIX):
                    gs.raise_exception(
                        f"'{path}' was written in version {magic[len(MAGIC_PREFIX) : -1].decode()} of the trajectory "
                        f"format, and this Genesis reads version {MAGIC[len(MAGIC_PREFIX) : -1].decode()} only."
                    )
                gs.raise_exception(f"'{path}' is not a Genesis trajectory.")
            (length,) = LENGTH.unpack(file.read(LENGTH.size))
            self._manifest = json.loads(file.read(length))
            (length,) = LENGTH.unpack(file.read(LENGTH.size))
            description = file.read(length)
            self._fields = _fields_from_json(self._manifest["fields"])
            self._info_fields = _fields_from_json(self._manifest["info_fields"])
            self._last_fields = _fields_from_json(self._manifest["last_fields"])
            self._frame_size = self._manifest["frame_size"]
            self._last_frame_size = self._manifest["last_frame_size"]
            # Each record is trusted by its checksum, so the offsets stop at the first one a crash left incomplete.
            self._chunk_offsets: list[int] = []
            self._chunk_ends: list[int] = []
            self._last_offset: int | None = None
            # The records of model parameters, by the first frame each holds for, in the order they were written
            self._info_frames: list[int] = []
            self._info_offsets: list[int] = []
            n_frames = 0
            while True:
                offset = file.tell()
                magic = file.read(len(CHUNK_MAGIC))
                if magic == CHUNK_MAGIC:
                    header = file.read(CHUNK_HEADER.size)
                    if len(header) < CHUNK_HEADER.size:
                        break
                    n_chunk_frames, length, crc = CHUNK_HEADER.unpack(header)
                    payload = file.read(length)
                    if len(payload) < length or zlib.crc32(payload) != crc:
                        break
                    n_frames += n_chunk_frames
                    self._chunk_offsets.append(offset)
                    self._chunk_ends.append(n_frames)
                elif magic == INFO_MAGIC:
                    header = file.read(INFO_HEADER.size)
                    if len(header) < INFO_HEADER.size:
                        break
                    i_frame, length, crc = INFO_HEADER.unpack(header)
                    payload = file.read(length)
                    if len(payload) < length or zlib.crc32(payload) != crc:
                        break
                    self._info_frames.append(i_frame)
                    self._info_offsets.append(offset)
                elif magic == LAST_MAGIC:
                    header = file.read(LAST_HEADER.size)
                    if len(header) < LAST_HEADER.size:
                        break
                    length, crc = LAST_HEADER.unpack(header)
                    payload = file.read(length)
                    if len(payload) < length or zlib.crc32(payload) != crc:
                        break
                    self._last_offset = offset
                else:
                    break
        self._n_frames = n_frames + (self._last_offset is not None)
        if self._n_frames == 0:
            gs.raise_exception(f"'{path}' holds no frame.")
        self._i_cached_chunk: int | None = None
        self._cached_frames: np.ndarray | None = None
        self._i_cached_info: int | None = None
        self._cached_info: np.ndarray | None = None
        self._decompressor = zstandard.ZstdDecompressor()

        layout = self._manifest["layout"]
        self._layout = EnvironmentLayout(
            layout["n_envs"], tuple(layout["env_spacing"]), layout["n_envs_per_row"], layout["center_envs_at_origin"]
        )
        self._checkpoint_class = SceneCheckpoint
        if scene is None:
            scene = gs.Scene.load(io.BytesIO(description), show_viewer, viewer_options, vis_options, renderer)
            scene.build(**dataclasses.asdict(self._layout))
            description = None
        self._scene = scene
        self._digest = description_digest(scene.desc)
        if description is not None:
            recorded = serialization.load(io.BytesIO(description), {"scene": SceneDescription})["scene"]
            if description_digest(recorded) != self._digest:
                gs.raise_exception(f"'{path}' was recorded from another scene: its entities or physics options differ.")

    def __len__(self) -> int:
        return self._n_frames

    @property
    def scene(self) -> "Scene":
        """The built scene the trajectory plays in."""
        return self._scene

    @property
    def mode(self) -> str:
        """What the frames hold: 'exact' or 'minimal' (see TrajectoryFile)."""
        return self._manifest["mode"]

    @property
    def n_envs(self) -> int:
        """The number of environments the scene was recorded with."""
        return self._layout.n_envs

    def frame(self, index: int, kinds: frozenset[DataKind] = frozenset(DataKind)) -> dict[str, np.ndarray]:
        """Return the arrays of the given kinds that a frame holds, by name, shaped as the scene holds them.

        'index' is the frame, counted from the end when negative.
        """
        try:
            index = range(len(self))[index]
        except IndexError:
            gs.raise_exception(f"Frame {index} is out of the {len(self)} frames of '{self._path}'.")
        if self._last_offset is not None and index == len(self) - 1:
            with open(self._path, "rb") as file:
                file.seek(self._last_offset + len(LAST_MAGIC))
                length, _ = LAST_HEADER.unpack(file.read(LAST_HEADER.size))
                payload = self._decompressor.decompress(file.read(length), max_output_size=self._last_frame_size)
                # A writable buffer, as the restore wraps the arrays as tensors
                frame = np.frombuffer(bytearray(payload), np.uint8)
            return _frame_arrays(frame, self._last_fields, kinds)
        # The chunk holding the frame is decoded whole, undoing its block-sparse encoding then its chain of
        # differences (see BLOCK_SIZE), and kept for the frames of the same chunk read next.
        i_chunk = bisect.bisect_right(self._chunk_ends, index)
        if i_chunk != self._i_cached_chunk:
            n_blocks = self._frame_size // BLOCK_SIZE
            n_bitmap_bytes = n_blocks // 8
            with open(self._path, "rb") as file:
                file.seek(self._chunk_offsets[i_chunk] + len(CHUNK_MAGIC))
                n_chunk_frames, length, _ = CHUNK_HEADER.unpack(file.read(CHUNK_HEADER.size))
                # A chunk is compressed as a stream, whose header states no size, so the decompression is bounded by
                # the size the chunk holds at most
                payload = self._decompressor.decompress(
                    file.read(length), max_output_size=n_chunk_frames * (n_bitmap_bytes + self._frame_size)
                )
            encoded = np.frombuffer(payload, np.uint8)
            frames = np.zeros((n_chunk_frames, n_blocks, BLOCK_SIZE), dtype=np.uint8)
            offset = 0
            for frame in frames:
                is_block_changed = np.unpackbits(encoded[offset : offset + n_bitmap_bytes], bitorder="little")
                offset += n_bitmap_bytes
                n_frame_bytes = np.count_nonzero(is_block_changed) * BLOCK_SIZE
                blocks = encoded[offset : offset + n_frame_bytes]
                frame[is_block_changed.view(np.bool_)] = blocks.reshape((-1, BLOCK_SIZE))
                offset += n_frame_bytes
            frames = frames.reshape((n_chunk_frames, self._frame_size))
            np.bitwise_xor.accumulate(frames, axis=0, out=frames)
            self._i_cached_chunk = i_chunk
            self._cached_frames = frames
        i_frame_start = self._chunk_ends[i_chunk - 1] if i_chunk > 0 else 0
        arrays = _frame_arrays(self._cached_frames[index - i_frame_start], self._fields, kinds)
        if DataKind.INFO in kinds and self._info_frames:
            # The model parameters of a frame are the ones of the latest record written for it or a frame before it,
            # which the writer puts ahead of every frame they hold for
            i_info = bisect.bisect_right(self._info_frames, index) - 1
            if i_info != self._i_cached_info:
                with open(self._path, "rb") as file:
                    file.seek(self._info_offsets[i_info] + len(INFO_MAGIC))
                    _, length, _ = INFO_HEADER.unpack(file.read(INFO_HEADER.size))
                    payload = self._decompressor.decompress(
                        file.read(length), max_output_size=sum(field.nbytes for field in self._info_fields)
                    )
                self._i_cached_info = i_info
                # A writable buffer, as the restore wraps the arrays as tensors
                self._cached_info = np.frombuffer(bytearray(payload), np.uint8)
            arrays.update(_frame_arrays(self._cached_info, self._info_fields, kinds))
        return arrays

    def time(self, index: int) -> np.ndarray:
        """The simulated time of each environment at frame 'index', in seconds, as 'Scene.get_time' reports it."""
        # Multiplied as the simulator multiplies its own clock, so the two agree to the bit.
        steps = torch.as_tensor(self.frame(index)[STEPS_FIELD], dtype=gs.tc_float, device="cpu")
        return tensor_to_array(steps * self._manifest["dt"])

    def timings(self, index: int) -> dict[str, float]:
        """The wall time in seconds of every phase of the step a frame was reached by, by phase.

        'index' is the frame, counted from the end when negative. The phases are named as 'Scene.timings' names them. A
        phase the step did not time is NaN, and so is every phase of a frame no step led to, such as the first.
        """
        return {
            name[len(TIMINGS_PREFIX) :]: value[()]
            for name, value in self.frame(index, frozenset((DataKind.STATE,))).items()
            if name.startswith(TIMINGS_PREFIX)
        }

    def seek(self, index: int) -> None:
        """Put the scene in the state of frame 'index', counted from the end when negative.

        An exact frame puts back everything a step reads or writes, so the scene steps on from there bit-for-bit. A
        minimal frame puts back the state and recomputes the link and geom poses by forward kinematics, and the next
        step finds the contacts and constraint forces anew. Either puts back the model parameters the frame was recorded
        with. The scratch that a checkpoint file also holds is for inspection, shaped by the backend of the recording
        that wrote it.
        """
        kinds = MODES_KINDS[self.mode] | INFO_KINDS
        values = self.frame(index, kinds)
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for name, value in values.items():
            if name != STEPS_FIELD and not name.startswith(TIMINGS_PREFIX):
                solver_name, array_name = name.split(".", 1)
                arrays.setdefault(solver_name, {})[array_name] = value
        solvers = {
            solver_name: SolverCheckpoint(solver_arrays, {}, kinds) for solver_name, solver_arrays in arrays.items()
        }
        sim = SimulatorCheckpoint(steps=values[STEPS_FIELD], solvers=solvers)
        self._scene.__setstate__(
            self._checkpoint_class(scene=self._scene.desc, digest=self._digest, layout=self._layout, sim=sim)
        )

    def play(self, loop: bool = False) -> None:
        """Seek every frame in turn, which redraws the scene at each and plays the run in the viewer at its pace.

        With 'loop', the run starts over until the interactive viewer is closed, so the scene must have one.
        """
        viewer = self._scene.viewer
        if loop and viewer is None:
            gs.raise_exception("Looping a trajectory needs an interactive viewer.")
        # Frames are paced like the simulation steps they replay (see ViewerOptions.realtime_factor). The factor is read
        # at every frame because the user can change it from the viewer during the replay.
        pacer = Rate(1.0 / self._manifest["dt"])
        while True:
            steps_prev = None
            for index in range(len(self)):
                # A closed viewer ends the replay before a redraw reaches it
                if viewer is not None and not viewer.is_alive():
                    return
                steps = self.frame(index)[STEPS_FIELD]
                realtime_factor = None if viewer is None else viewer.realtime_factor
                if realtime_factor is not None:
                    pacer.period = self._manifest["dt"] / realtime_factor
                    viewer.sleep(pacer.advance(n_periods=0 if steps_prev is None else (steps - steps_prev).max()))
                self.seek(index)
                steps_prev = steps
            if not loop:
                return
