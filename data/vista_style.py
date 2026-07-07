from torch.utils.data import Dataset
import json
import os, json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import sqlite3
import pandas as pd
from .utils import get_trajectory_from_speeds_and_yaw_rates
from .l2_context import L2ContextMixin
from .video_loaders import build_clip_augmenter


class VistaStyleNuScenesLoader(Dataset):
    def __init__(self, *, size, json_path, images_root, num_frames=None, frame_rate_multiplier=1, aug="resize_center", sample_indices=None):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.json_path = json_path
        self.num_frames = num_frames
        self.images_root = images_root
        self.aug = aug

        assert frame_rate_multiplier <= 1, "Frame rate multiplier should be less than or equal to 1"
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)        
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.sample_indices = sample_indices
        if sample_indices is not None:
            self.data = [self.data[i] for i in sample_indices]
        self.augmenter = build_clip_augmenter(
            aug=self.aug,
            size=self.size,
            input_format="pil",
        )
        
    def __getitem__(self, index):
        sample = self.data[index]
        frame_paths = sample['frames'][::self.frame_interval]
        if self.num_frames is not None:
            if len(frame_paths) < self.num_frames:
                print(f"Warning: Number of frames {len(frame_paths)} is less than the required {self.num_frames}")
                raise ValueError(f"Number of frames {len(frame_paths)} is less than the required {self.num_frames}")
            frame_paths = frame_paths[:self.num_frames]
        images = [cv2.imread(os.path.join(self.images_root, frame_path)) for frame_path in frame_paths]
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        return self.augmenter(images)
        
    def __len__(self):
        return len(self.data)
    
    
    
def extract_pose_table(db_path):
    """
    Connects to the nuPlan SQLite database and extracts the entire ego_pose table.
    Returns a pandas DataFrame containing the pose metadata.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM ego_pose;"
    pose_df = pd.read_sql_query(query, conn)
    conn.close()
    # Ensure timestamp is numeric and sort by timestamp
    pose_df['timestamp'] = pd.to_numeric(pose_df['timestamp'], errors='coerce')
    pose_df.sort_values("timestamp", inplace=True)
    return pose_df

def get_pose(pose_df, pose_token):
    tk = bytes.fromhex(pose_token)
    pose = pose_df[pose_df['token'] == tk]
    if pose.empty:
        raise ValueError(f"Pose with token {pose_token} not found.")
    return pose


def __getattr__(name):
    if name == "VistaStyleNuScenesLoaderSteering":
        from data.steering_loaders import VistaStyleNuScenesLoaderSteering
        return VistaStyleNuScenesLoaderSteering
    if name == "VistaStyleNuScenesLoaderSteeringWithL2Context":
        from data.steering_loaders import VistaStyleNuScenesLoaderSteeringWithL2Context
        return VistaStyleNuScenesLoaderSteeringWithL2Context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



class VistaStyleNuScenesLoaderWithL2Context(L2ContextMixin, Dataset):
    def __init__(self, *, size, json_path, images_root, num_frames=None,
                 frame_rate_multiplier=1, stored_data_frame_rate=None, frame_rate=None,
                 num_l2_context=0, l2_frame_rate=1.0, l1_context_frames=1,
                 aug=None, sample_indices=None):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.json_path = json_path
        self.num_frames = num_frames
        self.images_root = images_root
        self.aug = aug or "resize_center"

        # Compute frame_interval from stored_data_frame_rate/frame_rate when provided,
        # otherwise fall back to frame_rate_multiplier.
        if stored_data_frame_rate is not None and frame_rate is not None:
            assert stored_data_frame_rate % frame_rate == 0, \
                'stored_data_frame_rate must be divisible by frame_rate'
            self.frame_interval = int(stored_data_frame_rate / frame_rate)
            self.stored_data_frame_rate = stored_data_frame_rate
            self.frame_rate = float(frame_rate)
        else:
            assert frame_rate_multiplier <= 1, "Frame rate multiplier should be less than or equal to 1"
            assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), \
                'reciprocal of frame_rate_multiplier must be an integer'
            self.frame_interval = int(1/frame_rate_multiplier)
            # Only set if not already provided by a subclass before calling super().__init__
            if not hasattr(self, 'stored_data_frame_rate'):
                self.stored_data_frame_rate = stored_data_frame_rate
            if not hasattr(self, 'frame_rate'):
                self.frame_rate = float(frame_rate) if frame_rate is not None else None

        self._init_l2_context(
            num_l2_context=num_l2_context,
            l2_frame_rate=l2_frame_rate,
            l1_context_frames=l1_context_frames,
        )
        self.l1_start_offset = self.get_required_l1_start_offset()

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.sample_indices = sample_indices
        if sample_indices is not None:
            self.data = [self.data[i] for i in sample_indices]
        self.augmenter = build_clip_augmenter(
            aug=self.aug,
            size=self.size,
            input_format="tensor",
        )

    def _load_raw_frames(self, frame_paths):
        images = [cv2.imread(os.path.join(self.images_root, p)) for p in frame_paths]
        images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
        return torch.stack(
            [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in images],
            dim=0,
        )

    def _load_images(self, frame_paths):
        frames = self._load_raw_frames(frame_paths)
        return self.augmenter(frames)

    def __getitem__(self, index):
        sample = self.data[index]
        all_frames = sample['frames']

        l1_start = self.l1_start_offset
        l1_indices = self.get_l1_indices(l1_start, self.num_frames)
        l2_indices = self.get_l2_indices(l1_start)
        all_indices = l1_indices + l2_indices

        if max(all_indices) >= len(all_frames):
            raise ValueError(
                f"Number of available L1 frames {len(all_frames)} is insufficient for "
                f"num_frames={self.num_frames} with l1_start_offset={l1_start}."
            )

        images_all = self._load_images([all_frames[i] for i in all_indices])
        images = images_all[:len(l1_indices)]

        if self.l2_context_enabled:
            l2_context = images_all[len(l1_indices):]
            return {
                'images': images,
                'l2_context': l2_context,
                'frame_rate': torch.tensor(self.frame_rate).float(),
            }

        return images

    def __len__(self):
        return len(self.data)
