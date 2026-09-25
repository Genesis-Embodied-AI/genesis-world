import base64
import io
from types import GeneratorType

import numpy as np
import torch
from PIL import Image

from genesis.utils.misc import tensor_to_array

IMG_STD_ERR_THR = 1.0
IMG_NUM_ERR_THR = 0.001
IMG_BLUR_KERNEL_SIZE = 1  # Size of the blur kernel (must be odd)


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    # Determine absolute and relative tolerance from input arguments
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    # Convert input arguments as numpy arrays
    args = [actual, desired]
    for i, arg in enumerate(args):
        if isinstance(arg, (GeneratorType, map)):
            arg = tuple(arg)
        if isinstance(arg, (tuple, list)):
            arg = np.stack([tensor_to_array(val) for val in arg], axis=0)
        args[i] = tensor_to_array(arg)

    # Early return without checking anything is both arrays are empty (0D arrays have size 1).
    if all(e.size == 0 for e in args):
        return

    # Try to make sure both arrays have the exact same shape.
    # First, try to broadcast both matrices. Then it is does not work, squeeze them before trying again.
    try:
        args = np.broadcast_arrays(*args)
    except ValueError as e:
        try:
            args = np.broadcast_arrays(*map(np.squeeze, args))
        except ValueError:
            raise e

    np.testing.assert_allclose(*args, atol=atol, rtol=rtol, err_msg=err_msg)


def assert_equal(actual, desired, *, err_msg=None):
    assert_allclose(actual, desired, atol=0.0, rtol=0.0, err_msg=err_msg)


def assert_pixel_match(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    err_msg: str = "Images do not match",
    verbose: bool = True,
    std_err_threshold: float = IMG_STD_ERR_THR,
    ratio_err_threshold: float = IMG_NUM_ERR_THR,
    blurred_kernel_size: int = IMG_BLUR_KERNEL_SIZE,
) -> None:
    """Assert two RGB image arrays match.

    The images match unless the per-channel standard deviation of their blurred difference exceeds
    ``std_err_threshold`` AND the number of differing pixels exceeds ``ratio_err_threshold`` of the total size.
    This tolerates the few-pixel jitter that software renderers produce on any platform while still catching a
    real difference. On mismatch, raise ``AssertionError``; unless ``verbose`` is False, also print the error
    metrics and a base64-encoded PNG of the per-pixel delta (so the failing frame can be recovered from CI logs).
    """
    img_a = np.atleast_3d(np.asarray(img_a)).astype(np.float32)
    img_b = np.atleast_3d(np.asarray(img_b)).astype(np.float32)
    if img_a.shape != img_b.shape:
        raise AssertionError(f"{err_msg} (shape {img_a.shape} != {img_b.shape})")

    # Blur both images with a normalized box kernel to smooth anti-aliasing edges before comparing.
    blurred = []
    for img_arr in (img_a, img_b):
        if blurred_kernel_size == 1:
            blurred.append(img_arr)
            continue
        pad_size = blurred_kernel_size // 2
        h, w = img_arr.shape[:2]
        padded = np.pad(img_arr, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="edge")
        # A box kernel is separable, so the window sum accumulates one row shift at a time, then one column shift at a
        # time over that, which costs a handful of whole-array additions instead of a pass per pixel.
        rows = np.zeros((h, padded.shape[1], img_arr.shape[-1]), dtype=np.float32)
        for i in range(blurred_kernel_size):
            rows += padded[i : i + h]
        window_sum = np.zeros_like(img_arr, dtype=np.float32)
        for j in range(blurred_kernel_size):
            window_sum += rows[:, j : j + w]
        blurred.append(window_sum / blurred_kernel_size**2)

    img_err = np.minimum(np.abs(blurred[1] - blurred[0]), 255).astype(np.uint8)
    std_err = float(np.max(np.std(img_err.reshape((-1, img_err.shape[-1])), axis=0)))
    ratio_err = int((np.abs(img_err) > np.finfo(np.float32).eps).sum())
    if not (std_err > std_err_threshold and ratio_err > ratio_err_threshold * img_err.size):
        return

    if verbose:
        print(
            f"Image mismatch [std_err={std_err:.2f} (thr={std_err_threshold:.2f}), "
            f"ratio_err={ratio_err} (thr={ratio_err_threshold * img_err.size})]:"
        )
        raw_bytes = io.BytesIO()
        img_delta = np.minimum(np.abs(img_b - img_a), 255).astype(np.uint8)
        img_obj = Image.fromarray(img_delta.squeeze(-1) if img_delta.shape[-1] == 1 else img_delta)
        img_obj.save(raw_bytes, "PNG")
        raw_bytes.seek(0)
        print(base64.b64encode(raw_bytes.read()))
    raise AssertionError(err_msg)


def rgb_array_to_png_bytes(rgb_arr: np.ndarray | torch.Tensor) -> bytes:
    img = Image.fromarray(tensor_to_array(rgb_arr))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
