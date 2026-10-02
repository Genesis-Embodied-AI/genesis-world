import builtins
import sys
from functools import wraps

import torch

import genesis as gs

from .tensor import Tensor

# Ops given the requested dtype and 'device=gs.device', so that they allocate there directly and random ones draw from
# its generator at that precision. The other ops follow the device of their input.
_torch_factory_ops = (
    torch.tensor,
    torch.asarray,
    torch.as_tensor,
    torch.zeros,
    torch.ones,
    torch.arange,
    torch.range,
    torch.linspace,
    torch.logspace,
    torch.eye,
    torch.empty,
    torch.empty_strided,
    torch.full,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
)
_torch_ops = (
    *_torch_factory_ops,
    torch.as_strided,
    torch.from_numpy,
    torch.zeros_like,
    torch.ones_like,
    torch.empty_like,
    torch.full_like,
    torch.rand_like,
    torch.randn_like,
    torch.randint_like,
)


def _to_gs_dtype(dtype):
    """Map a Python or torch scalar type to the torch dtype that Genesis uses for this kind of scalar."""
    match dtype:
        case builtins.float | torch.float32 | torch.float64:
            return gs.tc_float
        case builtins.int | torch.int32 | torch.int64:
            return gs.tc_int
        case builtins.bool | torch.bool:
            return torch.bool
    gs.raise_exception(f"Unsupported dtype: {dtype}")


def torch_op_wrapper(torch_op):
    @wraps(torch_op)
    def _wrapper(*args, dtype=None, requires_grad=False, scene=None, **kwargs):
        if "device" in kwargs:
            gs.raise_exception("Device selection not supported. All genesis tensors are on GPU.")

        if not gs._initialized:
            gs.raise_exception("Genesis not initialized yet.")

        # Allocating at the requested scalar type before conversion would fail for float64 on devices lacking it (MPS)
        if dtype is not None:
            dtype = _to_gs_dtype(dtype)

        if torch_op is torch.from_numpy:
            torch_tensor = torch_op(*args)
        elif torch_op is torch.tensor:
            torch_tensor = torch_op(*args, dtype=dtype, requires_grad=requires_grad, device=gs.device)
        elif torch_op in _torch_factory_ops:
            torch_tensor = torch_op(*args, dtype=dtype, device=gs.device, **kwargs)
        else:
            torch_tensor = torch_op(*args, **kwargs)

        return from_torch(torch_tensor, dtype, requires_grad, detach=True, scene=scene)

    _wrapper.__doc__ = (
        f"This method is the genesis wrapper of `torch.{torch_op.__name__}`.\n\n------------------\n{_wrapper.__doc__}"
    )

    return _wrapper


def from_torch(torch_tensor, dtype=None, requires_grad=False, detach=True, scene=None):
    """
    By default, detach is True, meaning that this function returns a new leaf tensor which is not connected to torch_tensor's computation gragh.
    """
    dtype = _to_gs_dtype(torch_tensor.dtype if dtype is None else dtype)

    if torch_tensor.requires_grad and (not detach) and (not requires_grad):
        gs.logger.warning(
            "The parent torch tensor requires grad and detach is set to False. Ignoring requires_grad=False."
        )
        requires_grad = True

    gs_tensor = Tensor(torch_tensor.to(device=gs.device, dtype=dtype), scene=scene).clone()

    if detach:
        gs_tensor = gs_tensor.detach(sceneless=False)

    if requires_grad:
        gs_tensor = gs_tensor.requires_grad_()

    return gs_tensor


for _torch_op in _torch_ops:
    setattr(sys.modules[__name__], _torch_op.__name__, torch_op_wrapper(_torch_op))
