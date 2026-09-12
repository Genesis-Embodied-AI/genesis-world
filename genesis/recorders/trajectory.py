"""Log the state of a scene as it steps, one frame per step, and read the log back to seek and replay it.

A trajectory file holds the scene as 'Scene.export' writes it, then its frames in compressed chunks. Every part carries
its own length and checksum, so a run that crashes leaves every chunk written before readable, and a reader stops at the
first part it cannot trust. A checkpoint file (see 'Scene.save_checkpoint') is a trajectory file of one frame holding
every array, scratch included.
"""

import bisect
import dataclasses
import enum
import io
import json
import logging
import os
import struct
import threading
import time
import zlib
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers import KinematicSolver
from genesis.engine.states.solvers import KinematicSolverCheckpoint, SimulatorCheckpoint, SolverCheckpoint
from genesis.options.recorders import TrajectoryFile
from genesis.utils import serialization
from genesis.utils.array_class import CHECKPOINT_KINDS, DataKind
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from .file_writers import BaseFileWriter
from .recorder_manager import register_recording

if TYPE_CHECKING:
    from genesis.engine.scene import Scene, TrajectorySource
    from genesis.engine.simulator import Simulator
    from genesis.options.renderers import RendererOptions
    from genesis.options.vis import ViewerOptions, VisOptions

LOGGER = logging.getLogger(__name__)

TRAJECTORY_FORMAT = ".gstraj"
MAGIC = b"GSTRAJ1\n"
CHUNK_MAGIC = b"CHNK"
# A chunk record: frame count, payload length, payload checksum.
CHUNK_HEADER = struct.Struct("<IQI")
LAST_MAGIC = b"LAST"
# The record a trajectory ends with: payload length, payload checksum.
LAST_HEADER = struct.Struct("<QI")
LENGTH = struct.Struct("<Q")
# What a frame holds in each mode (see TrajectoryFile). A checkpoint file and the record a trajectory ends with hold
# every array but the static configs and constants, the scratch of the solvers included. They serve the analysis of a
# failure, and a failure may depend on the platform and never reproduce once shared. What the last step computed is
# therefore kept whole.
COMPRESSED_KINDS = frozenset((DataKind.INFO, DataKind.STATE))
EXACT_KINDS = CHECKPOINT_KINDS
CHECKPOINT_FILE_KINDS = frozenset(DataKind) - {DataKind.CONFIG, DataKind.CONSTANT}
# The per-environment step counts and the forward-kinematics flags of a solver, which follow the arrays in a frame.
STEPS_FIELD = "steps"
FLAGS_SUFFIX = ".forward_flags"


class FrameField(NamedTuple):
    """Where one array stands in a frame: its name, kind, canonical shape, numpy dtype string, byte offset and byte
    length.
    """

    name: str
    kind: DataKind
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int


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
        [field.name, field.kind.name, list(field.shape), field.dtype, field.offset, field.nbytes] for field in fields
    ]


def _fields_from_json(rows: list) -> list[FrameField]:
    """Return the frame layout the JSON rows of a file header hold."""
    return [
        FrameField(name, DataKind[kind], tuple(shape), dtype, offset, nbytes)
        for name, kind, shape, dtype, offset, nbytes in rows
    ]


def _frame_arrays(frame: np.ndarray, fields: list[FrameField], kinds: frozenset[DataKind]) -> dict[str, np.ndarray]:
    """Return the arrays of the given kinds that a frame holds, by name, shaped as the scene holds them."""
    return {
        field.name: frame[field.offset : field.offset + field.nbytes].view(field.dtype).reshape(field.shape)
        for field in fields
        if field.kind in kinds
    }


def _frame_fields(sim: "Simulator", kinds: frozenset[DataKind]) -> list[FrameField]:
    """Lay out one frame of the given kinds (see '_read_frame'): every array, then the step count of each environment
    and the forward-kinematics flags of every kinematic or rigid solver.
    """
    fields = []
    offset = 0
    for name, value, kind in sim.data(kinds):
        array = qd_to_numpy(value)
        fields.append(FrameField(name, kind, array.shape, array.dtype.str, offset, array.nbytes))
        offset += array.nbytes
    steps = tensor_to_array(sim.steps)
    fields.append(FrameField(STEPS_FIELD, DataKind.STATE, steps.shape, steps.dtype.str, offset, steps.nbytes))
    offset += steps.nbytes
    for solver in sim.active_solvers:
        if isinstance(solver, KinematicSolver):
            name = f"{type(solver).__name__}{FLAGS_SUFFIX}"
            fields.append(FrameField(name, DataKind.STATE, (2,), "|u1", offset, 2))
            offset += 2
    return fields


def _read_frame(sim: "Simulator", kinds: frozenset[DataKind]) -> np.ndarray:
    """Return one frame as a flat uint8 buffer laid out as '_frame_fields' states.

    Where zero-copy views exist the arrays are concatenated on the device and cross to the host once. Otherwise each
    array crosses on its own and the concatenation runs on the host.
    """
    flags = [
        flag
        for solver in sim.active_solvers
        if isinstance(solver, KinematicSolver)
        for flag in (solver.is_forward_pos_updated, solver.is_forward_vel_updated)
    ]
    if gs.use_zerocopy:
        parts = [qd_to_torch(value).contiguous().reshape(-1).view(torch.uint8) for _, value, _ in sim.data(kinds)]
        parts.append(sim.steps.reshape(-1).view(torch.uint8))
        parts.append(torch.tensor(flags, dtype=torch.uint8, device=gs.device))
        return tensor_to_array(torch.cat(parts))
    parts = [qd_to_numpy(value).reshape(-1).view(np.uint8) for _, value, _ in sim.data(kinds)]
    parts.append(tensor_to_array(sim.steps).reshape(-1).view(np.uint8))
    parts.append(np.array(flags, dtype=np.uint8))
    return np.concatenate(parts)


def _header(
    source: "TrajectorySource", exact: bool, chunk_size: int, fields: list[FrameField], last_fields: list[FrameField]
) -> bytes:
    """Return the magic, the JSON manifest and the scene description that open a trajectory file, 'fields' laying out
    its frames and 'last_fields' the record it ends with.
    """
    sim, desc, layout = source
    description = io.BytesIO()
    serialization.export(description, {"scene": desc})
    manifest = {
        "genesis": gs.__version__,
        "source": serialization.source_digest(),
        "backend": gs.backend.name,
        "precision": np.dtype(gs.np_float).name,
        "exact": exact,
        "chunk_size": chunk_size,
        "dt": sim.dt,
        "layout": {key: _json_value(value) for key, value in dataclasses.asdict(layout).items()},
        "configs": {
            name: {key: _json_value(value) for key, value in vars(config).items() if not key.startswith("_")}
            for solver in sim.active_solvers
            for name, config, kind in solver.data
            if kind == DataKind.CONFIG
        },
        "fields": _fields_to_json(fields),
        "last_fields": _fields_to_json(last_fields),
    }
    text = json.dumps(manifest).encode()
    return MAGIC + LENGTH.pack(len(text)) + text + LENGTH.pack(description.getbuffer().nbytes) + description.getvalue()


def save_checkpoint(path: str | os.PathLike, source: "TrajectorySource") -> None:
    """Write the whole state of a built scene as a trajectory file of one frame, the record a trajectory ends with (see
    'Scene.save_checkpoint').
    """
    fields = _frame_fields(source.sim, CHECKPOINT_FILE_KINDS)
    with open(path, "wb") as file:
        file.write(_header(source, exact=True, chunk_size=1, fields=fields, last_fields=fields))
        payload = zlib.compress(_read_frame(source.sim, CHECKPOINT_FILE_KINDS).tobytes())
        file.write(LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload)) + payload)


@register_recording(TrajectoryFile)
class TrajectoryFileWriter(BaseFileWriter):
    """Write the frames of a scene to a trajectory file (see TrajectoryFile), one chunk at a time.

    Each frame captures the state a step starts from, control inputs included, since it is read before the step.
    Within a chunk the writer stores each frame as its XOR with the previous frame. Unchanged bytes thus become zeros
    for zlib to remove, and an unchanged info array costs nothing. The frames are appended on the stepping thread; a
    completed chunk is compressed and written by a thread of its own, one chunk in flight at a time, so memory use is
    the open chunk and the one being written, whatever the file size.
    """

    def build(self):
        self._source: TrajectorySource = self._data_func()
        sim = self._source.sim
        self._is_exact = self._options.exact if self._options.exact is not None else sim.n_envs <= 1
        if not self._is_exact:
            gs.logger.warning(
                "[TrajectoryFileWriter] Recording the state alone: link and geom poses are recomputed at load and may "
                "differ at the last bits, and the log cannot resume the simulation bit-for-bit. Set 'exact=True' to "
                "log everything a step reads or writes."
            )
        self._kinds = EXACT_KINDS if self._is_exact else COMPRESSED_KINDS
        self._fields = _frame_fields(sim, self._kinds)
        self._last_fields = _frame_fields(sim, CHECKPOINT_FILE_KINDS)
        self._frame_size = sum(field.nbytes for field in self._fields)
        self._steps_field = next(field for field in self._fields if field.name == STEPS_FIELD)
        self._chunk = bytearray()
        self._n_chunk_frames = 0
        self._previous: np.ndarray | None = None
        self._n_frames = 0
        self._written = 0
        self._fault: str | None = None
        self._error: Exception | None = None
        self._writer: threading.Thread | None = None
        self._file: io.BufferedWriter | None = None

        super().build()

    def _initialize_writer(self):
        # The header goes out with the first record. A run that ends before any leaves an empty file, which no reader
        # takes for a trajectory.
        self._file = open(self._get_filename(), "wb")
        self._header = _header(self._source, self._is_exact, self._options.chunk_size, self._fields, self._last_fields)
        self._written = 0

    def _get_frame(self):
        return _read_frame(self._source.sim, self._kinds)

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
        if self._file is None:
            return
        # A full chunk is written as the next frame arrives, so the open chunk holds the latest frame. The stop replaces
        # that frame with the whole state (see 'cleanup').
        if self._n_chunk_frames == self._options.chunk_size:
            self._start_chunk_write()
        delta = data if self._previous is None else np.bitwise_xor(data, self._previous)
        self._chunk += delta.tobytes()
        self._previous = data
        self._n_chunk_frames += 1

    def _start_chunk_write(self):
        """Hand the open chunk to a writer thread, once the write of the previous chunk has ended."""
        if self._writer is not None:
            self._writer.join()
            self._writer = None
        if self._n_chunk_frames == 0 or self._file is None:
            return
        chunk, n_chunk_frames = bytes(self._chunk), self._n_chunk_frames
        self._chunk.clear()
        self._n_chunk_frames = 0
        self._previous = None
        self._writer = threading.Thread(target=self._write_chunk, args=(chunk, n_chunk_frames), daemon=True)
        self._writer.start()

    def _write(self, record: bytes, n_record_frames: int):
        """Append one record holding 'n_record_frames' frames, with the header ahead of the first, if the file may still
        grow by that much. Otherwise the file is closed and the fault recorded.
        """
        record = self._header + record
        if self._options.max_size is not None and self._written + len(record) > self._options.max_size:
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
        self._file.write(record)
        self._file.flush()
        self._header = b""
        self._written += len(record)
        self._n_frames += n_record_frames

    def _write_chunk(self, chunk: bytes, n_chunk_frames: int):
        """Compress and append one chunk on the writer thread. A failure is raised on the stepping thread (see 'step')."""
        try:
            payload = zlib.compress(chunk)
            chunk_header = CHUNK_MAGIC + CHUNK_HEADER.pack(n_chunk_frames, len(payload), zlib.crc32(payload))
            self._write(chunk_header + payload, n_chunk_frames)
        except Exception as error:
            self._error = error

    def cleanup(self):
        sim = self._source.sim
        if self._file is not None and self._n_chunk_frames > 0:
            # The frame the stop recorded carries the step counts of the live state. The whole state ends the file in
            # its place (see CHECKPOINT_FILE_KINDS), so it leaves the chunk.
            field = self._steps_field
            steps = self._previous[field.offset : field.offset + field.nbytes].view(field.dtype).reshape(field.shape)
            if np.array_equal(steps, tensor_to_array(sim.steps)):
                del self._chunk[-self._frame_size :]
                self._n_chunk_frames -= 1
        self._start_chunk_write()
        self._previous = None
        if self._writer is not None:
            self._writer.join()
            self._writer = None
        try:
            if self._error is None and self._file is not None:
                payload = zlib.compress(_read_frame(sim, CHECKPOINT_FILE_KINDS).tobytes())
                self._write(LAST_MAGIC + LAST_HEADER.pack(len(payload), zlib.crc32(payload)) + payload, 1)
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
        Viewer options replacing the recorded ones in the created scene. If None, the recorded ones stand.
    vis_options : VisOptions, optional
        Visualizer options replacing the recorded ones in the created scene. If None, the recorded ones stand.
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
            if file.read(len(MAGIC)) != MAGIC:
                gs.raise_exception(f"'{path}' is not a Genesis trajectory.")
            (length,) = LENGTH.unpack(file.read(LENGTH.size))
            self._manifest = json.loads(file.read(length))
            (length,) = LENGTH.unpack(file.read(LENGTH.size))
            description = file.read(length)
            self._fields = _fields_from_json(self._manifest["fields"])
            self._last_fields = _fields_from_json(self._manifest["last_fields"])
            self._frame_size = sum(field.nbytes for field in self._fields)
            # Each record is trusted by its checksum, so the offsets stop at the first one a crash left incomplete.
            self._chunk_offsets: list[int] = []
            self._chunk_ends: list[int] = []
            self._last_offset: int | None = None
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
    def is_exact(self) -> bool:
        """Whether the frames hold everything a step reads or writes, or the state alone (see TrajectoryFile)."""
        return self._manifest["exact"]

    @property
    def n_envs(self) -> int:
        """The number of environments the scene was recorded with."""
        return self._layout.n_envs

    def frame(self, index: int, kinds: frozenset[DataKind] = frozenset(DataKind)) -> dict[str, np.ndarray]:
        """Return the arrays of frame 'index' of the given kinds by name, shaped as the scene holds them. A negative
        'index' counts from the end.
        """
        try:
            index = range(len(self))[index]
        except IndexError:
            gs.raise_exception(f"Frame {index} is out of the {len(self)} frames of '{self._path}'.")
        if self._last_offset is not None and index == len(self) - 1:
            with open(self._path, "rb") as file:
                file.seek(self._last_offset + len(LAST_MAGIC))
                length, _ = LAST_HEADER.unpack(file.read(LAST_HEADER.size))
                # A writable buffer, as the restore wraps the arrays as tensors
                frame = np.frombuffer(bytearray(zlib.decompress(file.read(length))), np.uint8)
            return _frame_arrays(frame, self._last_fields, kinds)
        # The chunk holding the frame is decoded whole, undoing its chain of differences, and kept for the frames of
        # the same chunk read next.
        i_chunk = bisect.bisect_right(self._chunk_ends, index)
        if i_chunk != self._i_cached_chunk:
            with open(self._path, "rb") as file:
                file.seek(self._chunk_offsets[i_chunk] + len(CHUNK_MAGIC))
                n_chunk_frames, length, _ = CHUNK_HEADER.unpack(file.read(CHUNK_HEADER.size))
                deltas = np.frombuffer(zlib.decompress(file.read(length)), np.uint8)
            frames = deltas.reshape((n_chunk_frames, self._frame_size)).copy()
            np.bitwise_xor.accumulate(frames, axis=0, out=frames)
            self._i_cached_chunk = i_chunk
            self._cached_frames = frames
        i_frame_start = self._chunk_ends[i_chunk - 1] if i_chunk > 0 else 0
        return _frame_arrays(self._cached_frames[index - i_frame_start], self._fields, kinds)

    def time(self, index: int) -> np.ndarray:
        """The simulated time of each environment at frame 'index', in seconds, as 'Scene.get_time' reports it."""
        # Multiplied as the simulator multiplies its own clock, so the two agree to the bit.
        return tensor_to_array(torch.as_tensor(self.frame(index)[STEPS_FIELD], device="cpu") * self._manifest["dt"])

    def seek(self, index: int) -> None:
        """Put the scene in the state of frame 'index', counted from the end when negative.

        An exact frame puts back everything a step reads or writes, so the scene steps on from there bit-for-bit. A
        compressed frame puts back the state and recomputes the link and geom poses by forward kinematics. The scratch a
        checkpoint file also holds is for inspection: its shape follows the backend of the recording.
        """
        kinds = EXACT_KINDS if self.is_exact else COMPRESSED_KINDS
        values = self.frame(index, kinds)
        arrays: dict[str, dict[str, np.ndarray]] = {}
        flags: dict[str, tuple[bool, bool]] = {}
        for name, value in values.items():
            if name == STEPS_FIELD:
                continue
            solver_name, array_name = name.split(".", 1)
            if name.endswith(FLAGS_SUFFIX):
                # Native booleans: the step hands these to kernels as template arguments.
                flags[solver_name] = tuple(map(bool, value))
            else:
                arrays.setdefault(solver_name, {})[array_name] = value
        solvers = {}
        for solver_name, solver_arrays in arrays.items():
            if solver_name in flags:
                is_pos_updated, is_vel_updated = flags[solver_name]
                solvers[solver_name] = KinematicSolverCheckpoint(
                    solver_arrays, {}, kinds, is_pos_updated, is_vel_updated
                )
            else:
                solvers[solver_name] = SolverCheckpoint(solver_arrays, {}, kinds)
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
        # The viewer holds each redraw to one step of wall time (see ViewerOptions.realtime_factor). A frame spanning
        # more steps than one, as frames recorded every few steps do, waits for the steps beyond that one as well.
        is_paced = viewer is not None and viewer.realtime_factor is not None
        while True:
            steps_prev = None
            for index in range(len(self)):
                # A closed viewer ends the replay before a redraw reaches it
                if viewer is not None and not viewer.is_alive():
                    return
                steps = self.frame(index)[STEPS_FIELD]
                if is_paced and steps_prev is not None:
                    time.sleep(max((steps - steps_prev).max() - 1, 0) * self._manifest["dt"] / viewer.realtime_factor)
                self.seek(index)
                steps_prev = steps
            if not loop:
                return
