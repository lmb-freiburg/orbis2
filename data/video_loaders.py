import json
import os
import random
import signal
import warnings

from tqdm import tqdm
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
import torch.distributed as dist

from .l2_context import L2ContextMixin
from util import instantiate_from_config

try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False



def _normalize_size(size):
    return (size, size) if isinstance(size, int) else tuple(size)


class FrameAdapter:
    def __call__(self, frames):
        raise NotImplementedError


class PILFrameAdapter(FrameAdapter):
    def __call__(self, frames):
        arrays = np.stack([np.asarray(frame) for frame in frames], axis=0)
        return torch.from_numpy(arrays).permute(0, 3, 1, 2).float() / 255.0


class DecordFrameAdapter(FrameAdapter):
    def __call__(self, frames):
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


class TensorFrameAdapter(FrameAdapter):
    def __call__(self, frames):
        if not torch.is_tensor(frames):
            raise TypeError(f"Expected torch.Tensor frames, got {type(frames).__name__}")
        frames = frames.float()
        if frames.max().item() > 1.0:
            frames = frames / 255.0
        return frames


class SpatialTransformPolicy:
    def __init__(self, size):
        self.size = _normalize_size(size)

    def sample_params(self, frames, context=None):
        return None

    def apply(self, frames, params, context=None):
        raise NotImplementedError


class ResizeCenterPolicy(SpatialTransformPolicy):
    def apply(self, frames, params, context=None):
        frames = transforms.functional.resize(frames, [min(self.size)], antialias=True)
        return transforms.functional.center_crop(frames, list(self.size))


class RandomShiftPolicy(SpatialTransformPolicy):
    def __init__(self, size, max_shift_horizontal=60, max_shift_vertical=60):
        super().__init__(size)
        self.max_shift_horizontal = max_shift_horizontal
        self.max_shift_vertical = max_shift_vertical

    def sample_params(self, frames, context=None):
        frames = transforms.functional.resize(frames, [min(self.size)], antialias=True)
        crop_height, crop_width = self.size
        width, height = frames.shape[-1], frames.shape[-2]
        center_left = (width - crop_width) // 2
        center_top = (height - crop_height) // 2

        shift_horizontal = random.randint(-self.max_shift_horizontal, self.max_shift_horizontal)
        shift_vertical = random.randint(-self.max_shift_vertical, self.max_shift_vertical)

        left = max(0, min(center_left + shift_horizontal, width - crop_width))
        top = max(0, min(center_top + shift_vertical, height - crop_height))
        return left, top

    def apply(self, frames, params, context=None):
        frames = transforms.functional.resize(frames, [min(self.size)], antialias=True)
        left, top = self.sample_params(frames) if params is None else params
        crop_height, crop_width = self.size
        return frames[..., top:top + crop_height, left:left + crop_width]


class RandomResizedCenterPolicy(SpatialTransformPolicy):
    def __init__(self, size, scale=(0.5, 1.0)):
        super().__init__(size)
        self.scale = scale

    def sample_params(self, frames, context=None):
        height, width = frames.shape[-2], frames.shape[-1]
        area = height * width
        aspect_ratio = width / height
        target_area = random.uniform(*self.scale) * area

        new_width = int(round((target_area * aspect_ratio) ** 0.5))
        new_height = int(round((target_area / aspect_ratio) ** 0.5))
        return new_height, new_width

    def apply(self, frames, params, context=None):
        new_height, new_width = self.sample_params(frames) if params is None else params
        frames = transforms.functional.resize(frames, [new_height, new_width], antialias=True)
        return transforms.functional.center_crop(frames, list(self.size))


class CameraCalibrationPolicy(SpatialTransformPolicy):
    def __init__(
        self,
        size,
        transform=None,
        transform_config=None,
        calibration_key="intrinsics",
        fallback_policy=None,
    ):
        super().__init__(size)
        self.transform = transform if transform is not None else instantiate_from_config(transform_config)
        self.calibration_key = calibration_key
        self.fallback_policy = fallback_policy or ResizeCenterPolicy(size)

    def apply(self, frames, params, context=None):
        if context is None or self.calibration_key not in context:
            # print(f"[CamCalib] FALLBACK — key={self.calibration_key!r} context_keys={list(context.keys()) if context else None}", flush=True)
            return self.fallback_policy.apply(frames, params, context=context)
        return self.transform(frames, context[self.calibration_key])


class ClipAugmenter:
    def __init__(self, adapter, spatial_policy):
        self.adapter = adapter
        self.spatial_policy = spatial_policy

    def __call__(self, frames, context=None):
        frames = self.adapter(frames)
        params = self.spatial_policy.sample_params(frames, context=context)
        frames = self.spatial_policy.apply(frames, params, context=context)
        return frames * 2 - 1


def _build_frame_adapter(input_format):
    if input_format == "pil":
        return PILFrameAdapter()
    if input_format == "decord":
        return DecordFrameAdapter()
    if input_format in {"tensor", "torchcodec"}:
        return TensorFrameAdapter()
    raise ValueError(f"Unknown input format: {input_format}")


def _build_spatial_policy(aug, size, scale_min=0.15, scale_max=0.5):
    if aug == "resize_center":
        return ResizeCenterPolicy(size)
    if aug == "random_resize_center":
        return RandomResizedCenterPolicy(size, scale=(scale_min, scale_max))
    if aug == "random_shift":
        return RandomShiftPolicy(size, max_shift_horizontal=60, max_shift_vertical=30)
    raise ValueError(f"Unknown augmentation type: {aug}")


def build_clip_augmenter(*, aug, size, input_format, scale_min=0.15, scale_max=0.5, spatial_transform_config=None):
    spatial_policy = (
        instantiate_from_config(spatial_transform_config)
        if spatial_transform_config is not None
        else _build_spatial_policy(aug, size, scale_min=scale_min, scale_max=scale_max)
    )
    return ClipAugmenter(
        adapter=_build_frame_adapter(input_format),
        spatial_policy=spatial_policy,
    )


def concat_frame_batches(*frame_batches):
    first_batch = frame_batches[0]
    if isinstance(first_batch, np.ndarray):
        return np.concatenate(frame_batches, axis=0)
    if torch.is_tensor(first_batch):
        return torch.cat(frame_batches, dim=0)
    raise TypeError(f"Unsupported frame batch type: {type(first_batch).__name__}")


def _read_h5_value(node):
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str) and value[:1] in "{[":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value.item() if hasattr(value, "item") else value
    return {key: _read_h5_value(node[key]) for key in node.keys()}


class DatasetMultiFrameIdxMapping(Dataset):
    """
    Template dataset that maps each index to a specific frame in a specific video.
    Subclasses must implement get_images_and_indices() and apply_transforms().
    """
    def __init__(self):
        super().__init__()
        self.index_to_starting_frame_map = []

    def __len__(self):
        return len(self.index_to_starting_frame_map)

    def get_images_and_indices(self, idx):
        raise NotImplementedError

    def apply_transforms(self, images, context=None):
        raise NotImplementedError

    def build_context(self, metadata):
        return None

    def __getitem__(self, idx):
        images, metadata = self.get_images_and_indices(idx)
        images = self.apply_transforms(images, context=self.build_context(metadata))
        return {"images": images, "frame_rate": self.frame_rate}


class MultiHDF5DatasetMultiFrameIdxMapping(DatasetMultiFrameIdxMapping):
    """
    This dataset maps each index to a specific frame in a specific video.
    Useful for validation and selecting subsets of frames.
    """
    def __init__(self, size, hdf5_paths_file, num_frames, stored_data_frame_rate=5, frame_rate=5, aug="resize_center", scale_min=0.15, scale_max=0.5):
        super().__init__()
        self.frame_interval = int(stored_data_frame_rate / frame_rate)
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate

        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.hdf5_paths_file = hdf5_paths_file
        with open(os.path.expandvars(hdf5_paths_file), "r") as f:
            self.hdf5_paths = f.read().splitlines()

        self.hdf5_files = [h5py.File(path, "r") for path in self.hdf5_paths]

        self.scan_h5_files()

        self.aug = aug
        self.augmenter = build_clip_augmenter(
            aug=self.aug,
            size=self.size,
            input_format="pil",
            scale_min=scale_min,
            scale_max=scale_max,
        )

    def scan_h5_files(self):
        self.index_to_starting_frame_map = []
        for file in self.hdf5_files:
            keys = list(file.keys())
            for key in keys:
                video_length = len(file[key])
                max_frame_index = video_length - self.num_frames * self.frame_interval - 1
                for i in range(0, max_frame_index + 1):
                    self.index_to_starting_frame_map.append((file, key, i))

    def __str__(self):
        return f"MultiHDF5DatasetMultiFrameIdxMapping({self.hdf5_paths_file}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})"

    def get_images_and_indices(self, idx):
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}")
        file, key, start_frame = self.index_to_starting_frame_map[idx]
        images = [Image.fromarray(file[key][start_frame + i * self.frame_interval]) for i in range(self.num_frames)]
        return images, (file.filename, key, start_frame)

    def apply_transforms(self, images, context=None):
        return self.augmenter(images, context=context)

    def close(self):
        for file in self.hdf5_files:
            file.close()


class MultiMP4DatasetMultiFrameIdxMapping(DatasetMultiFrameIdxMapping):
    def __init__(self, size, mp4_paths_file, num_frames, stored_data_frame_rate=5, frame_rate=5, aug="resize_center", backend=None, subsample_interval=None, intrinsics_h5_path=None, spatial_transform_config=None):
        super().__init__()
        self.frame_interval = int(stored_data_frame_rate / frame_rate)
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.mp4_paths_file = mp4_paths_file
        self.subsample_interval = subsample_interval
        self.intrinsics_h5_path = os.path.expandvars(intrinsics_h5_path) if intrinsics_h5_path is not None else None
        self.intrinsics_file = None
        
        with open(os.path.expandvars(mp4_paths_file), "r") as f:
            # the list is either a list of mp4 file paths, or a csv with two columns: file_path,length_in_frames, with a header row.
            lines = f.read().splitlines()
            if len(lines) > 1 and len(lines[0].split(",")) == 2:
                # CSV format
                self.mp4_paths = [self._normalize_path(p.split(",")[0]) for p in lines[1:]]
                mp4_lengths = [int(p.split(",")[1]) for p in lines[1:]]
                # CSV format: create a {'path': length_in_frames} dict
                self.mp4_lengths_in_frames = dict(zip(self.mp4_paths, mp4_lengths))
            else:
                # Plain list of file paths
                self.mp4_paths = [self._normalize_path(p) for p in lines]
                self.mp4_lengths_in_frames = None


        if backend is not None:
            self.backend = backend
            assert self.backend in ["decord", "torchcodec"], f"Unknown backend {self.backend}"
            assert (self.backend == "decord" and DECORD_AVAILABLE) or (self.backend == "torchcodec" and TORCHCODEC_AVAILABLE), f"Specified backend {self.backend} is not available"
        else:
            if DECORD_AVAILABLE:
                self.backend = "decord"
            elif TORCHCODEC_AVAILABLE:
                self.backend = "torchcodec"
            else:
                raise ImportError("Either Decord or TorchCodec must be installed to use MultiMP4DatasetMultiFrameIdxMapping")

        self._validate_frame_rate_sample()
        self.scan_mp4_files_distributed()

        if self.backend == "decord":
            self.read_frames = self.read_frames_decord
        elif self.backend == "torchcodec":
            self.read_frames = self.read_frames_torchcodec
        else:
            raise ImportError("Either Decord or TorchCodec must be installed to use MultiMP4DatasetMultiFrameIdxMapping")

        self.aug = aug
        self.augmenter = build_clip_augmenter(
            aug=self.aug,
            size=self.size,
            input_format=self.backend,
            spatial_transform_config=spatial_transform_config,
        )

        # Decode robustness knobs (override via environment if needed).
        self.decode_max_retries = max(int(os.environ.get("ORBIS_DECODE_MAX_RETRIES", "4")), 0)
        self.decode_timeout_s = max(int(os.environ.get("ORBIS_DECODE_TIMEOUT_S", "20")), 0)
        self.decode_warn_limit = max(int(os.environ.get("ORBIS_DECODE_WARN_LIMIT", "50")), 0)
        self._decode_warn_count = 0

    def _validate_frame_rate_sample(self, sample_size=100):
        paths = random.sample(self.mp4_paths, min(sample_size, len(self.mp4_paths)))
        for path in paths:
            if self.backend == "decord":
                fps = VideoReader(path, ctx=decord_cpu(0), num_threads=-1).get_avg_fps()
            else:
                fps = VideoDecoder(path).metadata.average_fps
            assert self.stored_data_frame_rate == fps, (
                f"Stored data frame rate {self.stored_data_frame_rate} does not match "
                f"actual frame rate {fps} for file {path}"
            )

    def get_video_length(self, path):
        if self.backend == "decord":
            file = VideoReader(path, ctx=decord_cpu(0), num_threads=-1)
            assert self.stored_data_frame_rate == file.get_avg_fps(), f"Stored data frame rate {self.stored_data_frame_rate} does not match actual frame rate {file.get_avg_fps()} for file {file}"
            video_length = len(file)
        elif self.backend == "torchcodec":
            file = VideoDecoder(path)
            assert self.stored_data_frame_rate == file.metadata.average_fps, f"Stored data frame rate {self.stored_data_frame_rate} does not match actual frame rate {file.metadata.average_fps} for file {file}"
            video_length = file.metadata.num_frames
        else:
            raise ImportError("Either Decord or TorchCodec must be installed to use MultiMP4DatasetMultiFrameIdxMapping")
        return video_length

    def _normalize_path(self, path):
        return os.path.normpath(os.path.expandvars(os.path.expanduser(path)))

    def _get_video_id(self, path):
        return os.path.basename(path).split(".")[0]

    def _get_intrinsics_file(self):
        if self.intrinsics_h5_path is None:
            return None
        if self.intrinsics_file is None:
            self.intrinsics_file = h5py.File(self.intrinsics_h5_path, "r")
        return self.intrinsics_file

    def _load_intrinsics(self, path):
        intrinsics_file = self._get_intrinsics_file()
        if intrinsics_file is None:
            return None
        video_id = self._get_video_id(path)
        if video_id not in intrinsics_file or "intrinsics" not in intrinsics_file[video_id]:
            return None
        return _read_h5_value(intrinsics_file[video_id]["intrinsics"])

    def build_context(self, metadata):
        _, path, start_frame = metadata
        context = {"path": path, "video_id": self._get_video_id(path), "start_frame": start_frame}
        intrinsics = self._load_intrinsics(path)
        if intrinsics is not None:
            context["intrinsics"] = intrinsics
        return context

    def scan_mp4_files(self):                
        is_rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        self.index_to_starting_frame_map = []
        for path in tqdm(
            self.mp4_paths,
            desc=f"Scanning MP4 files in {self.__class__.__name__}",
            disable=not is_rank0,
        ):
            if self.mp4_lengths_in_frames is not None and path in self.mp4_lengths_in_frames:
                video_length = self.mp4_lengths_in_frames[path]
            else:
                video_length = self.get_video_length(path)

            frame_interval = self.frame_interval if self.subsample_interval is None else self.subsample_interval*self.frame_interval
            
            max_frame_index = video_length - self.num_frames * self.frame_interval - 1
            # assert max_frame_index > 0
            if not max_frame_index > 0: continue
            for i in range(0, max_frame_index + 1, frame_interval):
                self.index_to_starting_frame_map.append((path, i))

    def scan_mp4_files_distributed(self):
        if not (dist.is_available() and dist.is_initialized()):
            self.scan_mp4_files()
            return

        # Scan only on rank 0 and broadcast index mapping to all other ranks.
        if dist.get_rank() == 0:
            self.scan_mp4_files()
            index_map = self.index_to_starting_frame_map
        else:
            index_map = None

        obj_list = [index_map]
        dist.broadcast_object_list(obj_list, src=0)
        self.index_to_starting_frame_map = obj_list[0]

    def __str__(self):
        return f"MultiMP4DatasetMultiFrameIdxMapping({self.mp4_paths_file}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})"

    def apply_transforms(self, images, context=None):
        return self.augmenter(images, context=context)

    def read_frames_decord(self, path, start_frame):
        indices = list(range(start_frame, start_frame + self.num_frames * self.frame_interval, self.frame_interval))
        file = VideoReader(path, ctx=decord_cpu(0))
        frames = file.get_batch(indices).asnumpy()
        return frames

    def read_frames_torchcodec(self, path, start_frame):
        file = VideoDecoder(path)
        frames = file[start_frame:start_frame + self.num_frames * self.frame_interval:self.frame_interval]
        return frames

    def _decode_with_timeout(self, path, start_frame):
        if self.decode_timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
            return self.read_frames(path, start_frame)

        def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
            raise TimeoutError(
                f"Decode timed out after {self.decode_timeout_s}s for {path} at frame {start_frame}"
            )

        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(self.decode_timeout_s))
        try:
            return self.read_frames(path, start_frame)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _warn_decode_error(self, msg):
        if self._decode_warn_count < self.decode_warn_limit:
            warnings.warn(msg)
            self._decode_warn_count += 1
        elif self._decode_warn_count == self.decode_warn_limit:
            warnings.warn(
                "Reached ORBIS_DECODE_WARN_LIMIT; suppressing further decode warnings."
            )
            self._decode_warn_count += 1

    def get_images_and_indices(self, idx):
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}")
        map_len = len(self.index_to_starting_frame_map)
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1

        last_exc = None
        for attempt in range(self.decode_max_retries + 1):
            attempt_idx = (idx + attempt) % map_len
            path, start_frame = self.index_to_starting_frame_map[attempt_idx]
            try:
                frames = self._decode_with_timeout(path, start_frame)
                if frames is None:
                    raise RuntimeError("Decoder returned None")
                num_decoded = frames.shape[0] if hasattr(frames, "shape") else len(frames)
                if num_decoded != self.num_frames:
                    raise RuntimeError(
                        f"Decoder returned {num_decoded} frames, expected {self.num_frames}"
                    )
                return frames, (None, path, start_frame)
            except Exception as exc:
                last_exc = exc
                self._warn_decode_error(
                    f"[MultiMP4Dataset] decode failure worker={worker_id} attempt={attempt + 1}/"
                    f"{self.decode_max_retries + 1} idx={attempt_idx} path={path} "
                    f"start_frame={start_frame}: {type(exc).__name__}: {exc}"
                )

        raise RuntimeError(
            f"Failed to decode sample idx={idx} after {self.decode_max_retries + 1} attempts"
        ) from last_exc

    def close(self):
        if self.intrinsics_file is not None:
            self.intrinsics_file.close()


class MultiMP4DatasetMultiFrameIdxMappingWithL2Context(L2ContextMixin, MultiMP4DatasetMultiFrameIdxMapping):
    """
    Extends MultiMP4DatasetMultiFrameIdxMapping to also return a 'l2_context' key
    containing num_l2_context frames at l2_frame_rate (typically 1 Hz) that end on
    the first L1 frame.

    The l2_context frames are sampled from the same video at the same spatial crop,
    with l2_context[-1] matching images[0].

    Args:
        num_l2_context     : number of L2 context frames (default 3)
        l2_frame_rate      : frame rate for L2 context frames in Hz (default 1.0)
        All other args forwarded to MultiMP4DatasetMultiFrameIdxMapping.

    Batch output adds:
        'l2_context' : (num_l2_context, C, H, W) pixel tensor in [-1, 1],
                       ordered oldest → newest (l2_context[-1] is the same
                       frame as images[0]).
    """

    def __init__(self, num_l2_context=3, l2_frame_rate=1.0, l1_context_frames=1, **kwargs):
        super().__init__(**kwargs)
        self._init_l2_context(
            num_l2_context=num_l2_context,
            l2_frame_rate=l2_frame_rate,
            l1_context_frames=l1_context_frames,
        )
        self.index_to_starting_frame_map = self.filter_index_map_with_l2_headroom(
            self.index_to_starting_frame_map,
        )

    def get_images_and_indices(self, idx):
        """Return (l1_frames, l2_frames, metadata)."""
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f"Index {idx} out of range.")
        path, start_frame = self.index_to_starting_frame_map[idx]
        l1_indices, l2_indices = self.get_l1_and_l2_indices(start_frame, self.num_frames)

        all_indices = l1_indices + l2_indices
        all_frames = self._decode_indices(path, all_indices)

        l1_frames = all_frames[:len(l1_indices)]
        l2_frames = all_frames[len(l1_indices):]
        return l1_frames, l2_frames, (None, path, start_frame)

    def _decode_indices(self, path, indices):
        """Decode an arbitrary list of frame indices from an MP4."""
        if self.backend == "decord":
            file = VideoReader(path, ctx=decord_cpu(0))
            frames = file.get_batch(indices).asnumpy()
        elif self.backend == "torchcodec":
            file = VideoDecoder(path)
            frames = torch.stack([file[i] for i in indices])
        else:
            raise RuntimeError(f"Unknown backend {self.backend}")
        return frames

    def __getitem__(self, idx):
        map_len = len(self.index_to_starting_frame_map)
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        last_exc = None

        for attempt in range(self.decode_max_retries + 1):
            attempt_idx = (idx + attempt) % map_len
            path, start_frame = self.index_to_starting_frame_map[attempt_idx]
            try:
                l1_frames, l2_frames, _ = self.get_images_and_indices(attempt_idx)
                all_frames = concat_frame_batches(l1_frames, l2_frames)
                all_transformed = self.augmenter(
                    all_frames,
                    context=self.build_context((None, path, start_frame)),
                )

                n_l1 = self.num_frames
                images_l1 = all_transformed[:n_l1]    # (F, C, H, W)
                images_l2 = all_transformed[n_l1:]    # (num_l2_context, C, H, W)

                return {
                    "images": images_l1,
                    "l2_context": images_l2,
                    "frame_rate": self.frame_rate,
                }
            except Exception as exc:
                last_exc = exc
                self._warn_decode_error(
                    f"[MultiMP4DatasetWithL2] decode failure worker={worker_id} "
                    f"attempt={attempt + 1}/{self.decode_max_retries + 1} "
                    f"idx={attempt_idx} path={path} start_frame={start_frame}: "
                    f"{type(exc).__name__}: {exc}"
                )

        raise RuntimeError(
            f"Failed to decode sample idx={idx} after {self.decode_max_retries + 1} attempts"
        ) from last_exc

