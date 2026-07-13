import os

import h5py
import numpy as np
import torch

from data.utils import get_trajectory_from_speeds_and_yaw_rates
from data.l2_context import L2ContextMixin
from data.video_loaders import (
    MultiHDF5DatasetMultiFrameIdxMapping,
    MultiMP4DatasetMultiFrameIdxMapping,
    MultiMP4DatasetMultiFrameIdxMappingWithL2Context,
)
from data.vista_style import (
    VistaStyleNuScenesLoader,
    VistaStyleNuScenesLoaderWithL2Context,
    extract_pose_table,
    get_pose,
)
from util import instantiate_from_config


class OdometryHorizonMixin:
    @staticmethod
    def _validate_frame_rate_ratio(stored_rate, sampled_rate, rate_name):
        ratio = stored_rate / sampled_rate
        if not np.isclose(ratio, round(ratio), atol=1e-8):
            raise ValueError(
                f"stored_data_frame_rate={stored_rate} must be an integer multiple of "
                f"{rate_name}={sampled_rate}"
            )

    @staticmethod
    def _infer_num_frames_odo_for_matched_horizon(num_frames, frame_rate, odometry_frame_rate):
        inferred = (num_frames - 1) * odometry_frame_rate / frame_rate + 1
        if not np.isclose(inferred, round(inferred), atol=1e-8):
            raise ValueError(
                "Matched video/odometry horizons require an integer odometry length, got "
                f"num_frames={num_frames}, frame_rate={frame_rate}, "
                f"odometry_frame_rate={odometry_frame_rate}"
            )
        return int(round(inferred))

    @classmethod
    def _resolve_odometry_horizon(
        cls,
        num_frames,
        frame_rate,
        num_frames_odo,
        odometry_frame_rate,
        odometry_horizon,
        inferred_num_frames_odo,
    ):
        valid_horizons = {"auto", "match_video", "explicit"}
        if odometry_horizon not in valid_horizons:
            raise ValueError(
                f"Unknown odometry_horizon={odometry_horizon!r}. Expected one of {sorted(valid_horizons)}"
            )

        if odometry_horizon == "auto":
            if num_frames_odo is None or num_frames_odo == inferred_num_frames_odo:
                odometry_horizon = "match_video"
            else:
                odometry_horizon = "explicit"

        if odometry_horizon == "match_video":
            resolved_num_frames_odo = inferred_num_frames_odo
            if num_frames_odo is not None and num_frames_odo != resolved_num_frames_odo:
                raise ValueError(
                    f"num_frames_odo={num_frames_odo} does not match the video horizon; "
                    f"expected {resolved_num_frames_odo} for num_frames={num_frames}, "
                    f"frame_rate={frame_rate}, odometry_frame_rate={odometry_frame_rate}"
                )
            return resolved_num_frames_odo, odometry_horizon

        if num_frames_odo is None:
            raise ValueError("odometry_horizon='explicit' requires num_frames_odo to be provided")
        return int(num_frames_odo), odometry_horizon

    @classmethod
    def reconfigure_params_for_required_odometry_horizon(cls, params, required_odo_steps):
        if required_odo_steps is None:
            raise ValueError("`required_odo_steps` must be provided")
        required_odo_steps = int(required_odo_steps)
        if required_odo_steps <= 0:
            raise ValueError(f"`required_odo_steps` must be positive, got {required_odo_steps}.")

        if "odometry_frame_rate" not in params or params.odometry_frame_rate is None:
            params.odometry_frame_rate = params.frame_rate

        frame_rate = float(params.frame_rate)
        odometry_frame_rate = float(params.odometry_frame_rate)

        if odometry_frame_rate != frame_rate:
            required_num_frames = int(np.ceil((required_odo_steps - 1) * frame_rate / odometry_frame_rate)) + 1
            params.num_frames = max(int(params.num_frames), required_num_frames)
            params.odometry_horizon = "match_video"
            if "num_frames_odo" in params:
                del params["num_frames_odo"]
            return params

        if "num_frames_odo" in params and params.num_frames_odo is not None:
            params.num_frames_odo = max(int(params.num_frames_odo), required_odo_steps)
        else:
            params.num_frames_odo = required_odo_steps
        params.odometry_horizon = "explicit"
        return params


class PlaceholderSteeringMixin:
    """Mixin for datasets that have no real steering data.

    Injects a NaN steering tensor of the correct shape so that rollout_steering_v2
    can work with --steering_file or --no_steering without any changes to downstream
    code.  Using such a dataset without either of those flags raises an explicit error
    at rollout validation time (see validate_steering_source in rollout_steering_v2.py).
    """

    def _init_placeholder_steering(self, num_frames_odo, steering_dim=2, steering_format="speed_yawrate"):
        self._placeholder_num_frames_odo = int(num_frames_odo)
        self._placeholder_steering_dim = int(steering_dim)
        self._placeholder_steering_format = steering_format

    def _add_placeholder_steering(self, item: dict) -> dict:
        item["steering"] = torch.full(
            (self._placeholder_num_frames_odo, self._placeholder_steering_dim), float("nan")
        )
        item["steering_format"] = self._placeholder_steering_format
        item["_steering_placeholder"] = True
        return item

    @classmethod
    def reconfigure_params_for_required_odometry_horizon(cls, params, required_odo_steps):
        if required_odo_steps is None:
            raise ValueError("`required_odo_steps` must be provided")
        required_odo_steps = int(required_odo_steps)
        if required_odo_steps <= 0:
            raise ValueError(f"`required_odo_steps` must be positive, got {required_odo_steps}.")
        current = int(getattr(params, "num_frames_odo", 0) or 0)
        params.num_frames_odo = max(current, required_odo_steps)
        return params


class OdometryLoaderConti:
    steering_format = "speed_yawrate"

    def __call__(self, odo_data):
        speeds = np.array([odo_frame[0] for odo_frame in odo_data])
        yaw_rates = np.array([odo_frame[6] for odo_frame in odo_data])
        ret_odo = np.stack([speeds, yaw_rates], axis=1)
        assert ret_odo.shape[1] == 2 and ret_odo.shape[0] == len(odo_data), f"Unexpected odometry shape {ret_odo.shape}, expected ({len(odo_data)}, 2)"
        return ret_odo
    

class OdometryLoaderNuPlan:
    steering_format = "speed_yawrate"

    def __call__(self, odo_data):
        """
        odo_data: list of dicts, each dict contains IMU data for a frame
        """
        ret = np.stack([np.array([odo_frame["vx"], odo_frame["angular_rate_z"]]) for odo_frame in odo_data], axis=0)
        assert ret.shape[1] == 2 and ret.shape[0] == len(odo_data), f"Unexpected odometry shape {ret.shape}, expected ({len(odo_data)}, 2)"
        return ret


class OdometryLoaderNVIDIAPhysAI:
    steering_format = "speed_yawrate"

    def __init__(self, speed_key="vx", curvature_key="curvature"):
        self.speed_key = speed_key
        self.curvature_key = curvature_key
        
    def __call__(self, odo_data):
        """
        odo_data: list of dicts, each dict contains vx (speed), and curvature (which can be used to compute yaw rate as curvature * speed)
        
        We return the speed and yaw rate as the odometry.
        
        """
        speeds = np.array([odo_frame[self.speed_key] for odo_frame in odo_data])
        yaw_rates = np.array([odo_frame[self.curvature_key] * odo_frame[self.speed_key] for odo_frame in odo_data])
        ret_odo = np.stack([speeds, yaw_rates], axis=1)
        assert ret_odo.shape[1] == 2 and ret_odo.shape[0] == len(odo_data), f"Unexpected odometry shape {ret_odo.shape}, expected ({len(odo_data)}, 2)"
        return ret_odo


class TrajectoryLoaderNuPlanFromSpeedYawRate:
    steering_format = "trajectory_with_heading"

    def __init__(self, frame_rate, speed_key="vx", yaw_rate_key="angular_rate_z"):
        self.frame_rate = frame_rate
        self.speed_key = speed_key
        self.yaw_rate_key = yaw_rate_key

    def __call__(self, odo_data):
        """
        odo_data: list of dicts, each dict contains IMU data for a frame
        """
        traj, headings = get_trajectory_from_speeds_and_yaw_rates(
            speeds=np.array([odo_frame[self.speed_key] for odo_frame in odo_data]),
            yaw_rates=np.array([odo_frame[self.yaw_rate_key] for odo_frame in odo_data]),
            dt=1.0 / self.frame_rate,
        )

        assert traj.shape[1] == 2 and traj.shape[0] == len(odo_data), f"Unexpected odometry shape {traj.shape}, expected ({len(odo_data)}, 2)"
        steering = np.concatenate([traj, headings[:, None]], axis=-1)  # (num_frames, 3)
        return steering


class TrajectoryLoaderNVIDIAPhysAIFromSpeedCurvature:
    def __init__(self, frame_rate, speed_key="speed", curvature_key="curvature", return_headings=False):
        self.frame_rate = frame_rate
        self.speed_key = speed_key
        self.curvature_key = curvature_key
        self.return_headings = return_headings
        self.steering_format = "trajectory_with_heading" if return_headings else "trajectory"

    def __call__(self, odo_data):
        """
        odo_data: list of dicts, each dict contains vx (speed), and curvature (which can be used to compute yaw rate as curvature * speed)
        
        We then compute the trajectory by integrating the speeds and yaw rates over time.
        
        """
        speeds = np.array([odo_frame[self.speed_key] for odo_frame in odo_data])
        yaw_rates = np.array([odo_frame[self.curvature_key] * odo_frame[self.speed_key] for odo_frame in odo_data])
        traj, headings = get_trajectory_from_speeds_and_yaw_rates(
            speeds=speeds,
            yaw_rates=yaw_rates,
            dt=1.0 / self.frame_rate,
        )
        assert traj.shape[1] == 2 and traj.shape[0] == len(odo_data), f"Unexpected odometry shape {traj.shape}, expected ({len(odo_data)}, 2)"
        if self.return_headings:
            steering = np.concatenate([traj, headings[:, None]], axis=-1)  # (num_frames, 3)
        else:
            steering = traj  # (num_frames, 2)
        return steering



class MultiMP4DatasetMultiFrameIdxMappingNVIDIAPhysAI(OdometryHorizonMixin, MultiMP4DatasetMultiFrameIdxMapping):
    """
    Loads the odometry from the NVIDIAPhysAI dataset. The odometry is stored in a single HDF5 file. The HDF5 file contains a dataset for each video (named after the video id).
    
    """
    def __init__(
        self,
        size,
        mp4_paths_file,
        odometry_h5_path,
        num_frames,
        num_frames_odo=None,
        stored_data_frame_rate=5,
        frame_rate=5,
        odometry_frame_rate=None,
        odometry_horizon="auto",
        aug="resize_center",
        backend=None,
        odo_transform_config=None,
        return_video_id=False,
        subsample_interval=None,
        intrinsics_h5_path=None,
        spatial_transform_config=None,
        validate_frame_rate_sample=True,
    ):
        self.odometry_frame_rate = frame_rate if odometry_frame_rate is None else odometry_frame_rate
        self.odometry_horizon = odometry_horizon

        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=frame_rate,
            rate_name="frame_rate",
        )
        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=self.odometry_frame_rate,
            rate_name="odometry_frame_rate",
        )
        self.odo_frame_interval = int(round(stored_data_frame_rate / self.odometry_frame_rate))
        inferred_num_frames_odo = self._infer_num_frames_odo_for_matched_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            odometry_frame_rate=self.odometry_frame_rate,
        )
        self.num_frames_odo, self.odometry_horizon = self._resolve_odometry_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_frames_odo=num_frames_odo,
            odometry_frame_rate=self.odometry_frame_rate,
            odometry_horizon=odometry_horizon,
            inferred_num_frames_odo=inferred_num_frames_odo,
        )

        super().__init__(
            size=size,
            mp4_paths_file=mp4_paths_file,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            aug=aug,
            backend=backend,
            subsample_interval=subsample_interval,
            intrinsics_h5_path=intrinsics_h5_path,
            spatial_transform_config=spatial_transform_config,
            validate_frame_rate_sample=validate_frame_rate_sample,
        )

        self.odo_file = h5py.File(odometry_h5_path, "r")
        self.return_video_id = return_video_id
        
        if odo_transform_config is None:
            raise ValueError(
                "MultiMP4DatasetMultiFrameIdxMappingNVIDIAPhysAI requires explicit "
                "`odo_transform_config`. Use `data.steering_loaders.OdometryLoaderNVIDIAPhysAI` "
                "for raw [speed, yaw_rate], or `data.steering_loaders."
                "TrajectoryLoaderNVIDIAPhysAIFromSpeedCurvature` for trajectory targets."
            )
        self.odo_transform = instantiate_from_config(odo_transform_config)
        self.steering_format = getattr(self.odo_transform, "steering_format", "unknown")
        
        # check that all videos have corresponding odometry data, and that the odometry data has the same number of frames as the video
        for mp4_path in self.mp4_paths:
            video_id = self._get_video_id(mp4_path)
            if video_id not in self.odo_file:
                raise KeyError(f"Video id {video_id} not found in odometry file {odometry_h5_path}")
            odo_length = len(self.odo_file[video_id]['odometry'])
            video_length = self.mp4_lengths_in_frames[mp4_path] if self.mp4_lengths_in_frames is not None else self.get_video_length(mp4_path)
            # max difference of 1 frame is allowed to account for rounding issues when the frame rates are different
            if abs(odo_length - video_length) > 1:
                raise ValueError(f"Odometry length {odo_length} does not match video length {video_length} for video id {video_id}")

    def scan_mp4_files(self):
        self.index_to_starting_frame_map = []
        required_span = max(
            self.num_frames * self.frame_interval,
            self.num_frames_odo * self.odo_frame_interval,
        )
        for path in self.mp4_paths:
            if self.mp4_lengths_in_frames is not None and path in self.mp4_lengths_in_frames:
                video_length = self.mp4_lengths_in_frames[path]
            else:
                video_length = self.get_video_length(path)

            frame_interval = self.frame_interval if self.subsample_interval is None else self.subsample_interval * self.frame_interval
            max_frame_index = video_length - required_span - 1
            for i in range(0, max_frame_index + 1, frame_interval):
                self.index_to_starting_frame_map.append((path, i))

    def _get_odometry_indices(self, start_frame):
        return list(range(
            start_frame,
            start_frame + self.num_frames_odo * self.odo_frame_interval,
            self.odo_frame_interval,
        ))

    def _load_steering_for_video(self, video_id, start_frame):
        indices = self._get_odometry_indices(start_frame)
        odo_data = [self.odo_file[video_id]['odometry'][i] for i in indices]
        return torch.from_numpy(self.odo_transform(odo_data)).float()

    def __getitem__(self, idx):
        images, (_, path, start_frame) = self.get_images_and_indices(idx)
        images = self.apply_transforms(images, context=self.build_context((None, path, start_frame)))
        video_id = self._get_video_id(path)
        result = {
            "images": images,
            "frame_rate": self.frame_rate,
            "steering": self._load_steering_for_video(video_id, start_frame),
            "steering_format": self.steering_format,
        }
        if self.return_video_id:
            result["video_id"] = video_id
        return result


class MultiHDF5DatasetMultiFrameIdxMappingOdometry(
    OdometryHorizonMixin,
    MultiHDF5DatasetMultiFrameIdxMapping,
):
    def __init__(
        self,
        size,
        hdf5_paths_file,
        num_frames,
        num_frames_odo=None,
        frames_file_suffix="frames.h5",
        odo_files_suffix="odometry.h5",
        stored_data_frame_rate=5,
        frame_rate=5,
        odometry_frame_rate=None,
        odometry_horizon="auto",
        aug="resize_center",
        scale_min=0.15,
        scale_max=0.5,
        odo_transform_config=None,
    ):
        self.frames_file_suffix = frames_file_suffix
        self.odo_files_suffix = odo_files_suffix if odo_files_suffix is None or odo_files_suffix.lower() != "none" else None
        self.odometry_frame_rate = frame_rate if odometry_frame_rate is None else odometry_frame_rate
        self.odometry_horizon = odometry_horizon

        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=frame_rate,
            rate_name="frame_rate",
        )
        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=self.odometry_frame_rate,
            rate_name="odometry_frame_rate",
        )
        self.odo_frame_interval = int(round(stored_data_frame_rate / self.odometry_frame_rate))
        inferred_num_frames_odo = self._infer_num_frames_odo_for_matched_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            odometry_frame_rate=self.odometry_frame_rate,
        )
        self.num_frames_odo, self.odometry_horizon = self._resolve_odometry_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_frames_odo=num_frames_odo,
            odometry_frame_rate=self.odometry_frame_rate,
            odometry_horizon=odometry_horizon,
            inferred_num_frames_odo=inferred_num_frames_odo,
        )

        super().__init__(
            size=size,
            hdf5_paths_file=hdf5_paths_file,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            aug=aug,
            scale_min=scale_min,
            scale_max=scale_max,
        )

        if odo_transform_config is not None:
            self.odo_transform = instantiate_from_config(odo_transform_config)
        else:
            self.odo_transform = OdometryLoaderNuPlan()
        self.steering_format = getattr(self.odo_transform, "steering_format", "unknown")

    @staticmethod
    def frames_odo_matching_check(frames_h5, odo_h5):
        for key in frames_h5.keys():
            if "meta_data" in key:
                continue
            if key not in odo_h5:
                raise KeyError(f"Odometry key {key} not found in frames data")
            if len(odo_h5[key]) != len(frames_h5[key]):
                raise ValueError(
                    f"Odometry key {key} has different length than frames data: "
                    f"{len(odo_h5[key])} != {len(frames_h5[key])}"
                )

    def scan_h5_files_odo(self):
        self.files_odo = {}
        if self.odo_files_suffix is None:
            return

        for h5_file_frames in self.hdf5_files:
            odo_h5_path = h5_file_frames.filename.replace(self.frames_file_suffix, self.odo_files_suffix)
            self.files_odo[odo_h5_path] = h5py.File(odo_h5_path, "r")
            self.frames_odo_matching_check(h5_file_frames, self.files_odo[odo_h5_path])

    def scan_h5_files(self):
        self.index_to_starting_frame_map = []
        required_span = max(
            self.num_frames * self.frame_interval,
            self.num_frames_odo * self.odo_frame_interval,
        )
        for file in self.hdf5_files:
            for key in file.keys():
                if "meta_data" in key:
                    continue
                video_length = len(file[key])
                max_frame_index = video_length - required_span - 1
                for i in range(0, max_frame_index + 1):
                    self.index_to_starting_frame_map.append((file, key, i))
        self.scan_h5_files_odo()

    def _get_odometry_indices(self, start_frame):
        return list(range(
            start_frame,
            start_frame + self.num_frames_odo * self.odo_frame_interval,
            self.odo_frame_interval,
        ))

    def get_odometry(self, filename, key, start_idx):
        odo_filename = filename.replace(self.frames_file_suffix, self.odo_files_suffix)
        indices = self._get_odometry_indices(start_idx)
        odo = [self.files_odo[odo_filename][key][i] for i in indices]
        return torch.as_tensor(self.odo_transform(odo)).float()

    def __getitem__(self, idx):
        images, (filename, key, start_frame) = self.get_images_and_indices(idx)
        images = self.apply_transforms(images)
        if self.odo_files_suffix is not None:
            odo = self.get_odometry(filename, key, start_frame)
        else:
            odo = torch.full((self.num_frames_odo, 3), float("nan"))
        return {
            "images": images,
            "steering": odo,
            "frame_rate": self.frame_rate,
            "steering_format": self.steering_format,
        }


class MultiMP4DatasetMultiFrameIdxMappingNVIDIAPhysAIWithL2Context(
    L2ContextMixin,
    MultiMP4DatasetMultiFrameIdxMappingNVIDIAPhysAI,
):
    def __init__(self, *, num_l2_context, l2_frame_rate=1.0, l1_context_frames=1, **kwargs):
        super().__init__(**kwargs)
        self._init_l2_context(
            num_l2_context=num_l2_context,
            l2_frame_rate=l2_frame_rate,
            l1_context_frames=l1_context_frames,
            require_l2_context=True,
        )
        self.index_to_starting_frame_map = self.filter_index_map_with_l2_headroom(
            self.index_to_starting_frame_map,
        )

    def _decode_indices(self, path, indices):
        if self.backend == "decord":
            from decord import VideoReader, cpu as decord_cpu

            file = VideoReader(path, ctx=decord_cpu(0))
            frames = file.get_batch(indices).asnumpy()
        elif self.backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder

            file = VideoDecoder(path)
            frames = torch.stack([file[i] for i in indices])
        else:
            raise RuntimeError(f"Unknown backend {self.backend}")
        return frames

    def __getitem__(self, idx):
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}")

        path, start_frame = self.index_to_starting_frame_map[idx]
        l1_indices, l2_indices = self.get_l1_and_l2_indices(start_frame, self.num_frames)
        all_frames = self._decode_indices(path, l1_indices + l2_indices)
        all_transformed = self.augmenter(
            all_frames,
            context=self.build_context((None, path, start_frame)),
        )

        video_id = self._get_video_id(path)
        return {
            "images": all_transformed[:len(l1_indices)],
            "l2_context": all_transformed[len(l1_indices):],
            "frame_rate": self.frame_rate,
            "steering": self._load_steering_for_video(video_id, start_frame),
            "steering_format": self.steering_format,
        }


class VistaStyleNuScenesSteeringMixin(OdometryHorizonMixin):
    def _init_vista_style_steering(
        self,
        *,
        dbs_root,
        annotation_key,
        num_frames,
        stored_data_frame_rate,
        frame_rate,
        num_frames_odo=None,
        odometry_frame_rate=None,
        odometry_horizon="auto",
    ):
        self.dbs_root = dbs_root
        self.annotation_key = annotation_key
        self.steering_format = annotation_key
        self.odometry_frame_rate = frame_rate if odometry_frame_rate is None else odometry_frame_rate
        self.odometry_horizon = odometry_horizon
        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=frame_rate,
            rate_name="frame_rate",
        )
        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=self.odometry_frame_rate,
            rate_name="odometry_frame_rate",
        )
        self.odo_frame_interval = int(round(stored_data_frame_rate / self.odometry_frame_rate))
        inferred_num_frames_odo = self._infer_num_frames_odo_for_matched_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            odometry_frame_rate=self.odometry_frame_rate,
        )
        self.num_frames_odo, self.odometry_horizon = self._resolve_odometry_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_frames_odo=num_frames_odo,
            odometry_frame_rate=self.odometry_frame_rate,
            odometry_horizon=odometry_horizon,
            inferred_num_frames_odo=inferred_num_frames_odo,
        )
        assert self.annotation_key in ["speed_yawrate", "trajectory", "trajectory_with_heading"], (
            f"Unsupported annotation key {self.annotation_key}"
        )

    @staticmethod
    def _infer_num_frames_odo_for_matched_horizon(num_frames, frame_rate, odometry_frame_rate):
        if num_frames is None:
            raise ValueError("VistaStyleNuScenesLoaderSteering requires num_frames to resolve odometry horizon")
        return OdometryHorizonMixin._infer_num_frames_odo_for_matched_horizon(
            num_frames=num_frames,
            frame_rate=frame_rate,
            odometry_frame_rate=odometry_frame_rate,
        )

    def _get_vista_style_steering(self, sample):
        pose_table = extract_pose_table(os.path.join(self.dbs_root, sample["db_name"]))
        pose_tokens = sample["pose_tokens"]
        poses = [get_pose(pose_table, pose_token) for pose_token in pose_tokens]
        speeds = np.array([pose["vx"].values[0] for pose in poses])
        yaw_rates = np.array([pose["angular_rate_z"].values[0] for pose in poses])

        if self.annotation_key == "speed_yawrate":
            speed_yawrate = np.stack([speeds, yaw_rates], axis=-1)
            steering = speed_yawrate[::self.odo_frame_interval]
            assert len(steering) >= self.num_frames_odo, (
                f"Speed/YawRate length {len(steering)} is less than the required {self.num_frames_odo}"
            )
            steering = steering[:self.num_frames_odo]
        elif self.annotation_key == "trajectory_with_heading":
            traj, headings = get_trajectory_from_speeds_and_yaw_rates(
                speeds,
                yaw_rates,
                dt=1 / self.stored_data_frame_rate,
            )
            traj, headings = traj[::self.odo_frame_interval], headings[::self.odo_frame_interval]
            assert len(traj) >= self.num_frames_odo, (
                f"Trajectory length {len(traj)} is less than the required {self.num_frames_odo}"
            )
            steering = np.concatenate([traj[:self.num_frames_odo], headings[:self.num_frames_odo, None]], axis=-1)
        elif self.annotation_key == "trajectory":
            traj, _ = get_trajectory_from_speeds_and_yaw_rates(
                speeds,
                yaw_rates,
                dt=1 / self.stored_data_frame_rate,
            )
            traj = traj[::self.odo_frame_interval]
            assert len(traj) >= self.num_frames_odo, (
                f"Trajectory length {len(traj)} is less than the required {self.num_frames_odo}"
            )
            steering = traj[:self.num_frames_odo]
        else:
            raise ValueError(f"Unsupported annotation key {self.annotation_key}")

        return torch.from_numpy(steering).float()


class VistaStyleNuScenesLoaderSteering(VistaStyleNuScenesSteeringMixin, VistaStyleNuScenesLoader):
    def __init__(
        self,
        *,
        size,
        json_path,
        images_root,
        dbs_root,
        annotation_key,
        num_frames=None,
        num_frames_odo=None,
        stored_data_frame_rate=10,
        frame_rate=5,
        odometry_frame_rate=None,
        odometry_horizon="auto",
        aug="resize_center",
        sample_indices=None,
    ):
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate
        self.odometry_frame_rate = frame_rate if odometry_frame_rate is None else odometry_frame_rate
        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=frame_rate,
            rate_name="frame_rate",
        )
        self._validate_frame_rate_ratio(
            stored_rate=stored_data_frame_rate,
            sampled_rate=self.odometry_frame_rate,
            rate_name="odometry_frame_rate",
        )
        frame_rate_multiplier = frame_rate / stored_data_frame_rate
        super().__init__(
            size=size,
            json_path=json_path,
            images_root=images_root,
            num_frames=num_frames,
            frame_rate_multiplier=frame_rate_multiplier,
            aug=aug,
            sample_indices=sample_indices,
        )
        self._init_vista_style_steering(
            dbs_root=dbs_root,
            annotation_key=annotation_key,
            num_frames=self.num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            num_frames_odo=num_frames_odo,
            odometry_frame_rate=odometry_frame_rate,
            odometry_horizon=odometry_horizon,
        )

    def __getitem__(self, index):
        images = super().__getitem__(index)
        sample = self.data[index]
        return {
            "images": images,
            "steering": self._get_vista_style_steering(sample),
            "frame_rate": torch.tensor(self.frame_rate).float(),
            "steering_format": self.steering_format,
        }


class VistaStyleNuScenesLoaderSteeringWithL2Context(
    VistaStyleNuScenesSteeringMixin,
    VistaStyleNuScenesLoaderWithL2Context,
):
    def __init__(
        self,
        *,
        size,
        json_path,
        images_root,
        dbs_root,
        annotation_key,
        num_frames=None,
        num_frames_odo=None,
        stored_data_frame_rate=10,
        frame_rate=5,
        odometry_frame_rate=None,
        odometry_horizon="auto",
        num_l2_context,
        l2_frame_rate=1.0,
        l1_context_frames=1,
        aug="resize_center",
        sample_indices=None,
    ):
        super().__init__(
            size=size,
            json_path=json_path,
            images_root=images_root,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            num_l2_context=num_l2_context,
            l2_frame_rate=l2_frame_rate,
            l1_context_frames=l1_context_frames,
            aug=aug,
            sample_indices=sample_indices,
        )
        self._init_vista_style_steering(
            dbs_root=dbs_root,
            annotation_key=annotation_key,
            num_frames=self.num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            num_frames_odo=num_frames_odo,
            odometry_frame_rate=odometry_frame_rate,
            odometry_horizon=odometry_horizon,
        )

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        sample = self.data[index]
        batch["steering"] = self._get_vista_style_steering(sample)
        batch["steering_format"] = self.steering_format
        return batch


class MultiMP4DatasetMultiFrameIdxMappingNoSteering(PlaceholderSteeringMixin, MultiMP4DatasetMultiFrameIdxMapping):
    """MP4 dataset without real steering data.

    Returns NaN steering placeholders so that rollout_steering_v2 can work with
    --steering_file or --no_steering.  Using it without either option raises an
    explicit error at rollout time.

    Args:
        num_frames_odo: Number of steering timesteps to expose per sample.  Set
            this to at least the rollout odometry horizon, or rely on
            reconfigure_params_for_required_odometry_horizon to expand it.
        steering_dim: Feature dimension of the placeholder (default 2 for
            speed/yaw-rate).
        steering_format: The steering_format tag returned in the batch
            (default "speed_yawrate").
        All remaining args are forwarded to MultiMP4DatasetMultiFrameIdxMapping.
    """

    def __init__(
        self,
        size,
        mp4_paths_file,
        num_frames,
        num_frames_odo,
        steering_dim=2,
        steering_format="speed_yawrate",
        stored_data_frame_rate=5,
        frame_rate=5,
        aug="resize_center",
        backend=None,
        subsample_interval=None,
        intrinsics_h5_path=None,
        spatial_transform_config=None,
    ):
        self._init_placeholder_steering(num_frames_odo, steering_dim, steering_format)
        MultiMP4DatasetMultiFrameIdxMapping.__init__(
            self,
            size=size,
            mp4_paths_file=mp4_paths_file,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            aug=aug,
            backend=backend,
            subsample_interval=subsample_interval,
            intrinsics_h5_path=intrinsics_h5_path,
            spatial_transform_config=spatial_transform_config,
        )

    def __getitem__(self, idx):
        item = MultiMP4DatasetMultiFrameIdxMapping.__getitem__(self, idx)
        return self._add_placeholder_steering(item)


class MultiHDF5DatasetMultiFrameIdxMappingNoSteering(PlaceholderSteeringMixin, MultiHDF5DatasetMultiFrameIdxMapping):
    """HDF5 dataset without real steering data.

    Returns NaN steering placeholders so that rollout_steering_v2 can work with
    --steering_file or --no_steering.  Using it without either option raises an
    explicit error at rollout time.

    Args:
        num_frames_odo: Number of steering timesteps to expose per sample.
        steering_dim: Feature dimension of the placeholder (default 2).
        steering_format: steering_format tag returned in the batch (default "speed_yawrate").
        All remaining args are forwarded to MultiHDF5DatasetMultiFrameIdxMapping.
    """

    def __init__(
        self,
        size,
        hdf5_paths_file,
        num_frames,
        num_frames_odo,
        steering_dim=2,
        steering_format="speed_yawrate",
        stored_data_frame_rate=5,
        frame_rate=5,
        aug="resize_center",
        scale_min=0.15,
        scale_max=0.5,
    ):
        self._init_placeholder_steering(num_frames_odo, steering_dim, steering_format)
        MultiHDF5DatasetMultiFrameIdxMapping.__init__(
            self,
            size=size,
            hdf5_paths_file=hdf5_paths_file,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            aug=aug,
            scale_min=scale_min,
            scale_max=scale_max,
        )

    def __getitem__(self, idx):
        item = MultiHDF5DatasetMultiFrameIdxMapping.__getitem__(self, idx)
        return self._add_placeholder_steering(item)


class VistaStyleNuScenesLoaderWithL2ContextNoSteering(
    PlaceholderSteeringMixin,
    VistaStyleNuScenesLoaderWithL2Context,
):
    """VistaStyle NuScenes + L2-context dataset without real steering data.

    Returns NaN steering placeholders alongside the l2_context frames so that
    rollout_steering_v2 can work with --steering_file or --no_steering.
    Using it without either option raises an explicit error at rollout time.

    Args:
        num_frames_odo: Number of steering timesteps to expose per sample.
        steering_dim: Feature dimension of the placeholder (default 2 for
            speed/yaw-rate).
        steering_format: steering_format tag returned in the batch
            (default "speed_yawrate").
        num_l2_context, l2_frame_rate, l1_context_frames: forwarded to
            VistaStyleNuScenesLoaderWithL2Context / L2ContextMixin.
        All remaining args forwarded to VistaStyleNuScenesLoaderWithL2Context.
    """

    def __init__(
        self,
        num_frames_odo,
        steering_dim=2,
        steering_format="speed_yawrate",
        **kwargs,
    ):
        self._init_placeholder_steering(num_frames_odo, steering_dim, steering_format)
        VistaStyleNuScenesLoaderWithL2Context.__init__(self, **kwargs)

    def __getitem__(self, idx):
        item = VistaStyleNuScenesLoaderWithL2Context.__getitem__(self, idx)
        return self._add_placeholder_steering(item)


class MultiMP4DatasetMultiFrameIdxMappingWithL2ContextNoSteering(
    PlaceholderSteeringMixin,
    MultiMP4DatasetMultiFrameIdxMappingWithL2Context,
):
    """MP4 + L2-context dataset without real steering data.

    Returns NaN steering placeholders alongside the l2_context frames so that
    rollout_steering_v2 can work with --steering_file or --no_steering.
    Using it without either option raises an explicit error at rollout time.

    Args:
        num_frames_odo: Number of steering timesteps to expose per sample.
        steering_dim: Feature dimension of the placeholder (default 2).
        steering_format: steering_format tag returned in the batch (default "speed_yawrate").
        num_l2_context, l2_frame_rate, l1_context_frames: forwarded to L2ContextMixin.
        All remaining args forwarded to MultiMP4DatasetMultiFrameIdxMapping.
    """

    def __init__(
        self,
        num_frames_odo,
        steering_dim=2,
        steering_format="speed_yawrate",
        **kwargs,
    ):
        self._init_placeholder_steering(num_frames_odo, steering_dim, steering_format)
        MultiMP4DatasetMultiFrameIdxMappingWithL2Context.__init__(self, **kwargs)

    def __getitem__(self, idx):
        item = MultiMP4DatasetMultiFrameIdxMappingWithL2Context.__getitem__(self, idx)
        return self._add_placeholder_steering(item)
