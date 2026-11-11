"""Cross-vendor GPU information for the test infrastructure.

Each backend queries the GPUs through the vendor management library - NVIDIA Management Library (NVML) via
nvidia-ml-py for NVIDIA, AMD SMI (amdsmi) for AMD - rather than parsing command-line tools or reading the
driver proc/sysfs interface. The management libraries reach the devices through the same driver path as the
compute runtime, so they report the GPUs actually allocated to the current process even inside a container
whose proc/sysfs interface does not reflect that allocation (e.g. an attached or namespaced container). The
management libraries are optional dependencies, so they are imported lazily and each backend is gated on
is_available(). An AMD GPU whose ROCm install lacks AMD SMI is read from the topology of the ROCm kernel driver (KFD)
instead, the one ROCm itself enumerates its devices from.
"""

import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import psutil


class GpuBackend(ABC):
    """Vendor-specific accessor for the GPUs visible to the current process."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this backend's management library loads and its GPUs are usable in the current process."""

    @abstractmethod
    def get_device_count(self) -> int:
        """Number of GPUs visible to the current process."""

    @abstractmethod
    def get_device_vram_mib(self) -> tuple[int, ...]:
        """Total VRAM in MiB of each visible GPU, ordered by device index."""

    @abstractmethod
    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        """Device index of the GPU whose UUID matches, or -1 if no visible GPU does."""

    @abstractmethod
    def get_per_process_vram_mib(self) -> dict[int, int]:
        """VRAM in MiB used across the visible GPUs by each process, keyed by process id."""


class NvidiaBackend(GpuBackend):
    """NVIDIA backend backed by the NVIDIA Management Library (NVML) through nvidia-ml-py."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pynvml
        except ImportError:
            return False
        # Validate the same calls __init__ relies on, so a driver that loads but cannot enumerate reports the
        # backend as unavailable here instead of raising when it is later constructed.
        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return False
        return True

    def __init__(self):
        import pynvml

        # NVML reference-counts initialization and the test worker is short-lived, so the matching shutdown is
        # left to process exit; the handles stay valid for the lifetime of this backend.
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handles = tuple(pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount()))

    def get_device_count(self) -> int:
        return len(self._handles)

    def get_device_vram_mib(self) -> tuple[int, ...]:
        # A GPU sharing the system memory (e.g. NVIDIA GB10) has no dedicated memory for NVML to report, so it gets the
        # size of the system memory, as an Apple Silicon GPU does (see pytest_xdist_auto_num_workers).
        devices_vram_mib = []
        for handle in self._handles:
            try:
                devices_vram_mib.append(self._pynvml.nvmlDeviceGetMemoryInfo(handle).total >> 20)
            except self._pynvml.NVMLError_NotSupported:
                devices_vram_mib.append(psutil.virtual_memory().total >> 20)
        return tuple(devices_vram_mib)

    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        target = device_uuid.replace("-", "").lower()
        for index, handle in enumerate(self._handles):
            uuid = self._pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            # NVML reports the UUID as 'GPU-<uuid>' while torch reports the bare UUID.
            if uuid.removeprefix("GPU-").replace("-", "").lower() == target:
                return index
        return -1

    def get_per_process_vram_mib(self) -> dict[int, int]:
        usage: dict[int, int] = {}
        for handle in self._handles:
            # Genesis test workers use the GPU both for compute (Quadrants) and for rendering (EGL/OpenGL), which
            # the driver reports as two separate process lists. usedGpuMemory is the process' total memory on the
            # device in either list, so a process doing both appears in both with the same footprint: take it once
            # per device (max), then sum across devices for genuine multi-GPU use.
            per_device: dict[int, int] = {}
            for get_processes in (
                self._pynvml.nvmlDeviceGetComputeRunningProcesses,
                self._pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                for proc in get_processes(handle):
                    # usedGpuMemory is None when the driver cannot attribute memory to the process.
                    if proc.usedGpuMemory is not None:
                        per_device[proc.pid] = max(per_device.get(proc.pid, 0), proc.usedGpuMemory >> 20)
            for pid, mem in per_device.items():
                usage[pid] = usage.get(pid, 0) + mem
        return usage


class AmdBackend(GpuBackend):
    """AMD backend backed by AMD SMI (amdsmi), the management library shipped with ROCm."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import amdsmi
        except ImportError:
            return False
        # Validate the same calls __init__ relies on, so a ROCm setup that loads but cannot enumerate (e.g. a
        # container without /dev/kfd access) reports the backend as unavailable here instead of raising when it
        # is later constructed.
        try:
            amdsmi.amdsmi_init()
            amdsmi.amdsmi_get_processor_handles()
        except amdsmi.AmdSmiException:
            return False
        return True

    def __init__(self):
        import amdsmi

        # Shutdown is left to process exit, mirroring the NVIDIA backend.
        amdsmi.amdsmi_init()
        self._amdsmi = amdsmi
        self._handles = tuple(amdsmi.amdsmi_get_processor_handles())

    def get_device_count(self) -> int:
        return len(self._handles)

    def get_device_vram_mib(self) -> tuple[int, ...]:
        return tuple(
            self._amdsmi.amdsmi_get_gpu_memory_total(handle, self._amdsmi.AmdSmiMemoryType.VRAM) >> 20
            for handle in self._handles
        )

    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        target = device_uuid.replace("-", "").lower()
        for index, handle in enumerate(self._handles):
            uuid = self._amdsmi.amdsmi_get_gpu_device_uuid(handle)
            if uuid.replace("-", "").lower() == target:
                return index
        return -1

    def get_per_process_vram_mib(self) -> dict[int, int]:
        usage: dict[int, int] = {}
        for handle in self._handles:
            for proc in self._amdsmi.amdsmi_get_gpu_process_list(handle):
                # amdsmi_get_gpu_process_list returns process info dicts on recent ROCm and opaque handles on
                # older ones, which must be resolved to a dict through amdsmi_get_gpu_process_info.
                info = proc if isinstance(proc, dict) else self._amdsmi.amdsmi_get_gpu_process_info(handle, proc)
                mem = info.get("memory_usage", {}).get("vram_mem") or info.get("mem")
                if mem is not None:
                    pid = int(info["pid"])
                    usage[pid] = usage.get(pid, 0) + (int(mem) >> 20)
        return usage


def _read_kfd_properties(path: Path) -> dict[str, int]:
    return {name: int(value) for name, value in [line.split() for line in path.read_text().splitlines() if line]}


def _get_kfd_gpu_nodes() -> list[Path]:
    """Return the KFD topology nodes of the GPUs the current process can use, in the order HIP numbers them.

    The topology lists every GPU of the host, inside a container too. A GPU counts only if its render node exists, which
    a container holds for the GPUs passed to it alone, and if ROCR_VISIBLE_DEVICES lists it, by index or by its 'GPU-'
    UUID, when set. The nodes of the CPUs have no SIMD units. The selection of the test workers, which they make through
    CUDA_VISIBLE_DEVICES, applies on top of these indices, as it does on those of AMD SMI.
    """
    nodes = []
    for node in sorted(Path("/sys/class/kfd/kfd/topology/nodes").iterdir(), key=lambda node: int(node.name)):
        properties = _read_kfd_properties(node / "properties")
        if properties.get("simd_count", 0) > 0 and Path(f"/dev/dri/renderD{properties['drm_render_minor']}").exists():
            nodes.append(node)
    rocr_visible_devices = os.environ.get("ROCR_VISIBLE_DEVICES")
    if rocr_visible_devices is not None:
        nodes_unique_id = [_read_kfd_properties(node / "properties").get("unique_id") for node in nodes]
        visible_nodes = []
        for device in rocr_visible_devices.split(","):
            if device.startswith("GPU-") and int(device[4:], 16) in nodes_unique_id:
                visible_nodes.append(nodes[nodes_unique_id.index(int(device[4:], 16))])
            elif device.isdigit() and int(device) < len(nodes):
                visible_nodes.append(nodes[int(device)])
        nodes = visible_nodes
    return nodes


class AmdKfdBackend(GpuBackend):
    """AMD backend reading the topology of the ROCm kernel driver (KFD), for a ROCm install without AMD SMI."""

    @classmethod
    def is_available(cls) -> bool:
        return (
            Path("/sys/class/kfd/kfd/topology/nodes").is_dir()
            and os.access("/dev/kfd", os.R_OK | os.W_OK)
            and not AmdBackend.is_available()
            and bool(_get_kfd_gpu_nodes())
        )

    def __init__(self):
        self._nodes = _get_kfd_gpu_nodes()

    def get_device_count(self) -> int:
        return len(self._nodes)

    def get_device_vram_mib(self) -> tuple[int, ...]:
        devices_vram_mib = []
        for node in self._nodes:
            banks = [_read_kfd_properties(bank / "properties") for bank in (node / "mem_banks").iterdir()]
            devices_vram_mib.append(sum([bank["size_in_bytes"] for bank in banks]) >> 20)
        return tuple(devices_vram_mib)

    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        # The UUID HIP reports holds the hexadecimal digits of the unique id of the node as ASCII codes
        try:
            unique_id = int(bytes.fromhex(device_uuid.replace("-", "")).decode("ascii"), 16)
        except ValueError:
            return -1
        for index, node in enumerate(self._nodes):
            if _read_kfd_properties(node / "properties").get("unique_id") == unique_id:
                return index
        return -1

    def get_per_process_vram_mib(self) -> dict[int, int]:
        gpus_id = [(node / "gpu_id").read_text().strip() for node in self._nodes]
        usage: dict[int, int] = {}
        for proc in Path("/sys/class/kfd/kfd/proc").iterdir():
            for gpu_id in gpus_id:
                vram = proc / f"vram_{gpu_id}"
                if vram.exists():
                    usage[int(proc.name)] = usage.get(int(proc.name), 0) + (int(vram.read_text()) >> 20)
        return usage


def detect_gpu_backend() -> GpuBackend | None:
    """Return a fresh instance of the first available GPU backend, or None when no GPU backend is usable.

    A new instance is built on every call so the management library is initialized in the calling process,
    which keeps the backend valid in forked children (pytest-forked runs every test in one) where the parent's
    session and device handles would be stale.
    """
    backend_classes = (NvidiaBackend, AmdBackend, AmdKfdBackend)
    available_backends = [backend_cls for backend_cls in backend_classes if backend_cls.is_available()]

    if not available_backends:
        return None

    if len(available_backends) > 1:
        warnings.warn("Multiple GPU backends were detected on the current system; using the first one.", stacklevel=2)

    return available_backends[0]()
