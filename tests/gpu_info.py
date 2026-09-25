"""Cross-vendor GPU information for the test infrastructure.

Each backend queries the GPUs through the vendor management library - NVIDIA Management Library (NVML) via
nvidia-ml-py for NVIDIA, AMD SMI (amdsmi) for AMD - rather than parsing command-line tools or reading the
driver proc/sysfs interface. The management libraries reach the devices through the same driver path as the
compute runtime, so they report the GPUs actually allocated to the current process even inside a container
whose proc/sysfs interface does not reflect that allocation (e.g. an attached or namespaced container). The
management libraries are optional dependencies, so they are imported lazily and each backend is gated on
is_available().
"""

import warnings
from abc import ABC, abstractmethod


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
        return tuple(self._pynvml.nvmlDeviceGetMemoryInfo(handle).total >> 20 for handle in self._handles)

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


def detect_gpu_backend() -> GpuBackend | None:
    """Return a fresh instance of the first available GPU backend, or None when no GPU backend is usable.

    A new instance is built on every call so the management library is initialized in the calling process,
    which keeps the backend valid in forked children (pytest-forked runs every test in one) where the parent's
    session and device handles would be stale.
    """
    backend_classes = (NvidiaBackend, AmdBackend)
    available_backends = [backend_cls for backend_cls in backend_classes if backend_cls.is_available()]

    if not available_backends:
        return None

    if len(available_backends) > 1:
        warnings.warn("Multiple GPU backends were detected on the current system; using the first one.", stacklevel=2)

    return available_backends[0]()
