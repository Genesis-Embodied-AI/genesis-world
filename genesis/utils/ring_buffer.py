import torch

import genesis as gs


class TensorRingBuffer:
    """
    A helper class for storing a buffer of `torch.Tensor`s without allocating new tensors.

    Parameters
    ----------
    N : int
        The number of tensors to store.
    shape : tuple[int, ...]
        The shape of the tensors to store.
    dtype : torch.dtype
        The dtype of the tensors to store.
    buffer : torch.Tensor | None, optional
        The buffer tensor where all the data is stored. If not provided, a new tensor is allocated.
    idx : torch.Tensor, optional
        The index reference to the most recently updated position in the ring buffer as a mutable 0D torch.Tensor of
        integer dtype. If not provided, it is initialized to -1.
    """

    def __init__(
        self,
        N: int,
        shape: tuple[int, ...],
        dtype=torch.float32,
        buffer: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
    ):
        if buffer is None:
            self.buffer = torch.empty((N, *shape), dtype=dtype, device=gs.device)
        else:
            assert buffer.shape == (N, *shape)
            self.buffer = buffer
        self.N = N
        if idx is None:
            # Host memory is critical for performance and must be preserved: this index only ever selects a slot, and
            # reading a device-resident one back on the host would block on the device queue at every access
            self._idx = torch.tensor(-1, dtype=gs.tc_int, device="cpu")
        else:  # torch.Tensor
            self._idx = idx

    def at(
        self,
        idx: int | torch.Tensor,
        *others_idx: int | slice | torch.Tensor,
        copy: bool | None = None,
        per_row: bool = False,
    ) -> torch.Tensor:
        """
        Get the value of the tensor at the given index.

        Parameters
        ----------
        idx : int | torch.Tensor
            Index of the element to get from most recent to least recent (that has not been discarded yet).
            Passing a 1D tensor for advanced (aka fancy) indexing is supported, but this is requiring allocating fresh
            memory instead of returning a view, which is less efficient.
        others_idx : int | slice | torch.Tensor, optional
            Index of the elements to extract from the selected tensor. In case of advanced indexing, this is equivalent
            but significantly more efficient than doing this extraction in a latter stage.
        copy: bool | None, optional
            If `None`, then memory will be allocated only if necessary. If `True`, then memory will be allocated
            systematically instead of returning a view. If `False`, then allocating memory is forbidden and will raise
            an exception if returning a view is impossible.
        per_row : bool, optional
            If True, `idx` must be a 1D tensor of length self.buffer.shape[1] selecting one ring slot per row of
            the second buffer dimension. `others_idx` applies to the remaining dimensions. Result shape:
            (self.buffer.shape[1], *trailing_slice_shape). Always allocates memory.
        """
        # 'idx' is free to be a device tensor: a 0-dim host tensor is a valid scalar operand of a device tensor, so the
        # result follows the device of 'idx' and the selection below stays where the buffer lives
        rel_idx = (self._idx - idx) % self.N
        if per_row:
            sub = self.buffer[(slice(None), slice(None), *others_idx)] if others_idx else self.buffer
            idx_expanded = rel_idx.view(1, -1, *([1] * (sub.ndim - 2))).expand(1, *sub.shape[1:])
            return sub.gather(0, idx_expanded).squeeze(0)

        assert len(others_idx) < self.buffer.ndim
        tensor = self.buffer[(rel_idx, *others_idx)]
        if tensor.untyped_storage().data_ptr() == self.buffer.untyped_storage().data_ptr():
            if copy:
                tensor = tensor.clone()
        elif copy == False:
            gs.raise_exception("Allocating memory is necessary but 'copy=False'.")
        return tensor

    def get(self, idx: int) -> torch.Tensor:
        """
        Get a clone of the tensor at the given index.

        Parameters
        ----------
        idx : int
            Index of the element to get from most recent to least recent (that has not been discarded yet).
        """
        return self.buffer[idx].clone()

    def set(self, tensor: torch.Tensor):
        """
        Set the current position of the ring buffer.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to copy into the ring buffer.
        """
        self.buffer[self._idx] = tensor

    def rotate(self):
        """
        , and advance the index pointer
        """
        self._idx[()] = (self._idx + 1) % self.N

    def clone(self) -> "TensorRingBuffer":
        return TensorRingBuffer(
            self.N,
            self.buffer.shape[1:],
            dtype=self.buffer.dtype,
            buffer=self.buffer.clone(),
            idx=self._idx.clone(),
        )

    def __getitem__(self, key: int | slice | tuple) -> "TensorRingBuffer":
        """
        Enable slicing of the tensor ring buffer.

        Parameters
        ----------
        key : int | slice | tuple
            Slice object (e.g., 3:6) or integer index or tuple of indices

        Returns
        -------
        TensorRingBuffer
            A new ring buffer containing a view of the sliced data
        """
        if isinstance(key, int):
            sliced_buffer = self.buffer[:, key : key + 1]
        elif isinstance(key, slice):
            sliced_buffer = self.buffer[:, key]
        elif isinstance(key, tuple):
            indexes = (slice(None),) + key
            sliced_buffer = self.buffer[indexes]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        return TensorRingBuffer(
            self.N,
            sliced_buffer.shape[1:],
            dtype=sliced_buffer.dtype,
            buffer=sliced_buffer,
            idx=self._idx,
        )
