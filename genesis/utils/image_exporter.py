import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, Executor
from functools import partial

import torch
import numpy as np

import genesis as gs
from genesis.constants import IMAGE_TYPE
from genesis.utils.misc import tensor_to_array


def as_grayscale_image(
    data: np.ndarray, clip_max: float | None = None, enable_log_scale: bool = False, black_to_white: bool = False
) -> np.ndarray:
    """Convert a batched 2D array of numeric dtype as 8 bits single channel (grayscale) image array for visualization.

    Internally, this method clips non-finite values, optionally applies log scaling (i.e. `log(1.0 + data)`), then
    normalizes values between 0.0 and 1.0, to finally convert to grayscale.

    Parameters
    ----------
    data : ndarray [(N x) H x W]
        The data to normalize as a batched 2D array with any numeric dtype.
    clip_max : float, optional
        The maximum valid value if any. Default to None.
    enable_log_scale: bool, optional
        Wether to apply log scaling before normalization. Default to False.
    black_to_white: bool, optional
        Whether the color is transitioning from black to white as value increases or conversely. Default to False.
    """
    # Cast data to float32
    data_float = data.astype(np.float32)

    # Clip data, with special handling for non-finite values only if necessary for efficiency
    valid_mask = np.isfinite(data_float)
    if np.all(valid_mask):
        data_min = np.min(data_float, axis=(-2, -1), keepdims=True)
        data_max = np.max(data_float, axis=(-2, -1), keepdims=True)
    else:
        data_min = np.min(data_float, axis=(-2, -1), keepdims=True, initial=float("+inf"), where=valid_mask)
        data_max = np.max(data_float, axis=(-2, -1), keepdims=True, initial=float("-inf"), where=valid_mask)
    data_min = np.maximum(data_min, 0.0)
    if clip_max is not None:
        data_max = np.minimum(data_max, clip_max)
    data_float = np.clip(data_float, data_min, data_max)

    # Apply log scaling if requested
    if enable_log_scale:
        data_float = np.log(1.0 + data_float)

    # Normalize values between 0.0 and 1.0
    data_delta = data_max - data_min
    data_normalized = data_float - data_min if black_to_white else data_max - data_float
    np.divide(data_normalized, data_delta, where=data_delta > gs.EPS, out=data_normalized)

    # Discretize as unsigned int8
    return (data_normalized * 255.0).astype(np.uint8)


class FrameImageExporter:
    """
    This class enables exporting images from multiple cameras and environments in batch and in parallel, unlike
    `Camera.(start|stop)_recording` API, which only allows for exporting images from a single camera and environment.
    """

    def __init__(self, export_dir: str, depth_clip_max: float = 100.0, enable_depth_log_scale: bool = False):
        self.depth_clip_max = depth_clip_max
        self.enable_depth_log_scale = enable_depth_log_scale
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_frame_all_cameras(
        self,
        i_step: int,
        cameras_idx: Iterable | None = None,
        rgb: Sequence[np.ndarray] | None = None,
        depth: Sequence[np.ndarray] | None = None,
        segmentation: Sequence[np.ndarray] | None = None,
        normal: Sequence[np.ndarray] | None = None,
    ):
        """
        Export multiple frames from different cameras and environments in parrallel as PNG files.

        Note
        ----
        All specified sequences of images must have the same length.

        Parameters
        ----------
        i_step : int
            The current step index.
        cameras_idx: Iterable, optional
            Sequence of indices of cameras to export. If None, all cameras are exported.
        rgb: Sequence[ndarray[np.floating]], optional
            RGB image is a sequence of arrays of shape ([n_envs,] H, W, 3).
        depth: Sequence[ndarray[np.floating]], optional
            Depth image is a sequence of arrays of shape ([n_envs,] H, W).
        segmentation: Sequence[ndarray[np.integer]], optional
            Segmentation image is a sequence of arrays of shape ([n_envs,] H, W).
        normal: Sequence[ndarray[np.floating]], optional
            Normal image is a sequence of arrays of shape ([n_envs,] H, W, 3).
        """
        # Pack frames data for convenience
        frames_data = (rgb, depth, segmentation, normal)

        # Early return if nothing to do
        if all(e is None for e in frames_data):
            gs.logger.debug("No images to export.")
            return

        # Make sure that all image sequences are valid
        try:
            (num_cameras,) = set(map(len, (e for e in frames_data if e is not None)))
        except ValueError as e:
            for img_type, imgs_data in zip(IMAGE_TYPE, frames_data):
                if imgs_data is not None and len(imgs_data) == 0:
                    gs.raise_exception_from(f"'{img_type}' must be a non-empty sequence of arrays.", e)
            gs.raise_exception_from("Specified image sequences have inconsistent length.", e)

        # Set default camera indices if undefined
        if cameras_idx is None:
            cameras_idx = range(num_cameras)
        if num_cameras != len(cameras_idx):
            gs.raise_exception("Camera indices and image sequences have inconsistent length.")

        # Loop over single camera data asynchronously
        with ThreadPoolExecutor() as executor:
            for i_cam, frame_data in zip(
                cameras_idx, zip(*(e if e is not None else (None,) * num_cameras for e in frames_data))
            ):
                self.export_frame_single_camera(i_step, i_cam, *frame_data, executor=executor)

    def export_frame_single_camera(
        self,
        i_step,
        i_cam,
        rgb=None,
        depth=None,
        segmentation=None,
        normal=None,
        *,
        compress_level: int | None = None,
        executor: Executor | None = None,
    ):
        """
        Export multiple frames from a single camera but different environments in parallel as PNG files.

        Parameters
        ----------
        i_step: int
            The current step index.
        i_cam: int
            The index of the camera.
        rgb: ndarray[np.floating], optional
            RGB image array of shape ([n_envs,] H, W, 3).
        depth: ndarray[np.floating], optional
            Depth image array of shape ([n_envs,] H, W).
        segmentation: ndarray[np.integer], optional
            Segmentation image array of shape ([n_envs,] H, W).
        normal: ndarray[np.floating], optional
            Normal image array of shape ([n_envs,] H, W, 3).
        compress_level: int, optional
            Compression level when exporting images as PNG. Default to 3.
        executor: Executor, optional
            Executor to which I/O bounded jobs (saving to PNG) will be submitted. A local executor will be instantiated
            if none is provided.
        """
        # Postpone import of OpenCV at runtime to reduce hard system dependencies
        import cv2

        # Pack frames data for convenience
        frame_data = (rgb, depth, segmentation, normal)

        # Early return if nothing to do
        if all(e is None for e in frame_data):
            gs.logger.debug("No images to export.")
            return

        # Instantiate a new executor if none is provided
        is_local_executor = False
        if executor is None:
            is_local_executor = True
            executor = ThreadPoolExecutor()

        # Loop over each image type
        exported_types = []
        for img_type, imgs_data in zip(IMAGE_TYPE, frame_data):
            if imgs_data is None:
                continue

            # Convert data to numpy
            if isinstance(imgs_data, torch.Tensor):
                imgs_data = tensor_to_array(imgs_data)
            else:
                imgs_data = np.asarray(imgs_data)

            # Make sure that image data has shape `(n_env, H, W [, C>1])``
            if imgs_data.shape[-1] == 1:
                imgs_data = imgs_data[..., 0]
            if imgs_data.ndim == (3 if imgs_data.shape[-1] <= 4 else 2):
                imgs_data = imgs_data[None]
            if imgs_data.ndim not in (3, 4):
                gs.raise_exception("'{imgs_data}' images must be arrays of shape ([n_envs,] H, W [, C>1])")

            # Convert image data to grayscale array if necessary
            if img_type == IMAGE_TYPE.DEPTH:
                imgs_data = as_grayscale_image(
                    imgs_data, self.depth_clip_max, self.enable_depth_log_scale, black_to_white=False
                )
            elif img_type == IMAGE_TYPE.SEGMENTATION and imgs_data.ndim == 3:
                imgs_data = as_grayscale_image(imgs_data, None, enable_log_scale=False, black_to_white=True)
            imgs_data = imgs_data.astype(np.uint8)

            # Flip channel order if necessary
            if imgs_data.ndim == 4:
                imgs_data = np.flip(imgs_data, axis=-1)

            # Export image array as (compressed) PNG file.
            # Note that 'pillow>=11' is now consistently faster than 'cv2' when compression level is explicitly
            # specified, yet slower for (implicit) default compression level, namely 3.
            cv2_params = [cv2.IMWRITE_PNG_COMPRESSION, compress_level] if compress_level is not None else None
            for i_env, img_data in enumerate(imgs_data):
                frame_path = os.path.join(self.export_dir, f"{img_type}_cam{i_cam}_env{i_env}_{i_step:03d}.png")
                executor.submit(partial(cv2.imwrite, params=cv2_params), frame_path, img_data)
            exported_types.append((len(imgs_data), img_type.name.lower()))

        if exported_types:
            types_str = ", ".join(f"{num} {type}" for num, type in exported_types)
            gs.logger.info(
                f"Exported ~<{sum(num for num, _ in exported_types)} frame(s) ({types_str})>~ "
                f"from camera ~<{i_cam}>~ at step ~<{i_step}>~ to ~<{self.export_dir}>~"
            )

        # Shutdown executor if necessary
        if is_local_executor:
            executor.shutdown(wait=True)
