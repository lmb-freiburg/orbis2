import argparse
import json
import os
import sys
import imageio
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigTypeError
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

from data.utils import get_trajectory_from_speeds_and_yaw_rates_batch
from util import get_obj_from_str, instantiate_from_config
from models.second_stage.fm_model_v2 import _decode_selected_if_branch


logger = logging.getLogger(__name__)


def get_ckpt_epoch_step(ckpt_path):
    """Return the training epoch and global step stored in a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    return epoch, global_step


def _save_tensor_image(tensor, path):
    save_image((tensor.float().clamp(-1.0, 1.0) + 1.0) / 2.0, path)


# --- BEGIN: l2_predictor path info in save dir (safe to remove) ---
def _maybe_append_l2_predictor_debug_subdir(frames_dir, model, save_l2_debug):
    """When --save_l2_debug is set, nest frames_dir under the L2 predictor's exp folder/ckpt names."""
    if not save_l2_debug:
        return frames_dir

    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    get_path_info = getattr(condition_preprocessor, "get_l2_predictor_path_info", None)
    if not callable(get_path_info):
        return frames_dir

    l2_folder_name, l2_ckpt_name = get_path_info()
    return os.path.join(frames_dir, "l2_predictor", l2_folder_name, l2_ckpt_name)
# --- END: l2_predictor path info in save dir (safe to remove) ---


def _get_l2_debug_branch_and_scale(model):
    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None:
        return "sem", model.enc_scale_dino

    get_branch = getattr(condition_preprocessor, "get_l2_predictor_encoder_branch", None)
    get_scale = getattr(condition_preprocessor, "get_l2_predictor_latent_scale", None)
    if callable(get_branch) and callable(get_scale):
        return get_branch(), get_scale()

    return "sem", model.enc_scale_dino


@torch.no_grad()
def reconstruct_frames_in_chunks(model, frames, output_device="cpu"):
    """Reconstruct frames with bounded decode memory by processing one timestep at a time."""
    reconstructed = []
    for frame_idx in range(frames.shape[1]):
        frame = frames[:, frame_idx : frame_idx + 1]
        latent = model.encode_frames(frame)
        try:
            decoded = model.decode_frames(latent, output_device=output_device)
        except TypeError:
            decoded = model.decode_frames(latent)
            if output_device is not None:
                decoded = decoded.to(output_device)
        reconstructed.append(decoded)
    return torch.cat(reconstructed, dim=1)


@torch.no_grad()
def save_l2_debug_images(model, condition_history, frames_dir, sample_in_batch_idx, sample_idx):
    if not condition_history:
        return

    device = next(model.parameters()).device
    l2_branch, l2_scale = _get_l2_debug_branch_and_scale(model)
    debug_root = os.path.join(frames_dir, "l2_debug", f"sequence_{sample_idx:04d}")
    os.makedirs(debug_root, exist_ok=True)

    for step_idx, condition_kwargs in enumerate(condition_history):
        latent = condition_kwargs.get("z_l2_end")
        if not torch.is_tensor(latent):
            continue
        decoded = _decode_selected_if_branch(
            model.ae,
            l2_branch,
            latent[sample_in_batch_idx : sample_in_batch_idx + 1].to(device) / l2_scale,
        )
        _save_tensor_image(decoded[0], os.path.join(debug_root, f"step_{step_idx:04d}.jpg"))


def _tensor_to_serializable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _tensor_to_serializable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_to_serializable(item) for item in value]
    return value


def _save_2d_tensor_csv(tensor, csv_path, header_prefix):
    """Save a [T, D] tensor as CSV."""
    tensor = tensor.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor for debug dump, got shape {tuple(tensor.shape)}")

    num_cols = tensor.shape[1]
    header = [f"{header_prefix}_{col_idx}" for col_idx in range(num_cols)]
    np.savetxt(
        csv_path,
        tensor.numpy(),
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def _maybe_save_displacement_trajectory(condition_kwargs, debug_root, sample_in_batch_idx, output_prefix):
    """Persist the full anchor-aligned XY displacement trajectory when raw speed/yaw is available."""
    if not isinstance(condition_kwargs, dict):
        return None

    raw_speed_yaw = condition_kwargs.get("_raw_speed_yaw")
    anchor_odo_index = condition_kwargs.get("_anchor_odo_index")
    odometry_dt = condition_kwargs.get("_odometry_dt")
    if not (torch.is_tensor(raw_speed_yaw) and torch.is_tensor(anchor_odo_index) and torch.is_tensor(odometry_dt)):
        return None
    if raw_speed_yaw.shape[0] <= sample_in_batch_idx:
        return None

    raw_speed_yaw = raw_speed_yaw[sample_in_batch_idx].detach().cpu()
    anchor_odo_index = int(anchor_odo_index[sample_in_batch_idx].item())
    odometry_dt = odometry_dt[sample_in_batch_idx : sample_in_batch_idx + 1].detach().cpu().to(dtype=raw_speed_yaw.dtype)

    if raw_speed_yaw.ndim != 2 or raw_speed_yaw.shape[1] < 2:
        return None
    if not (0 <= anchor_odo_index < raw_speed_yaw.shape[0]):
        return None

    centered_traj = get_trajectory_from_speeds_and_yaw_rates_batch(
        speeds=raw_speed_yaw[anchor_odo_index:, 0].unsqueeze(0),
        yaw_rates=raw_speed_yaw[anchor_odo_index:, 1].unsqueeze(0),
        dt=odometry_dt,
    )[0, :, :2]

    _save_2d_tensor_csv(
        centered_traj,
        os.path.join(debug_root, f"{output_prefix}_displacement_trajectory.csv"),
        header_prefix=f"{output_prefix}_xy",
    )
    return {
        "shape": list(centered_traj.shape),
        "anchor_odo_index": anchor_odo_index,
        "trajectory_start_odo_index": anchor_odo_index,
        "trajectory_end_odo_index": raw_speed_yaw.shape[0] - 1,
    }


def _save_rollout_condition_history_csv(condition_history, sample_in_batch_idx, csv_path, key_path, value_prefix):
    """Save per-rollout-step conditioning tensors as a flat CSV with explicit step/point indices."""
    if not condition_history:
        return None

    rows = []
    value_dim = None
    for rollout_step_idx, step_condition_kwargs in enumerate(condition_history):
        value = step_condition_kwargs
        for key in key_path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if not torch.is_tensor(value) or value.shape[0] <= sample_in_batch_idx:
            continue

        value = value[sample_in_batch_idx].detach().cpu()
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            continue

        if value_dim is None:
            value_dim = value.shape[1]
        elif value.shape[1] != value_dim:
            raise ValueError(
                f"Inconsistent conditioning width in history for {key_path}: "
                f"{value.shape[1]} vs expected {value_dim}."
            )

        for point_idx, row in enumerate(value.tolist()):
            rows.append([float(rollout_step_idx), float(point_idx), *row])

    if not rows or value_dim is None:
        return None

    header = ["rollout_step", "condition_point_idx"] + [f"{value_prefix}_{dim_idx}" for dim_idx in range(value_dim)]
    np.savetxt(
        csv_path,
        np.asarray(rows, dtype=np.float32),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    return {
        "num_rows": len(rows),
        "value_dim": value_dim,
        "num_rollout_steps": len(condition_history),
    }


def save_condition_debug_dump(data_batch, condition_kwargs, condition_history, frames_dir, sample_in_batch_idx, sample_idx):
    """Persist raw and delegated steering state for one rollout sample."""
    debug_root = os.path.join(frames_dir, "condition_debug", f"sequence_{sample_idx:04d}")
    os.makedirs(debug_root, exist_ok=True)

    payload = {}

    if "frame_rate" in data_batch:
        frame_rate = data_batch["frame_rate"]
        if torch.is_tensor(frame_rate):
            payload["frame_rate"] = _tensor_to_serializable(frame_rate[sample_in_batch_idx])
        else:
            payload["frame_rate"] = frame_rate

    if "steering_format" in data_batch:
        payload["steering_format"] = data_batch["steering_format"]

    raw_steering = data_batch.get("steering")
    if torch.is_tensor(raw_steering):
        raw_steering = raw_steering[sample_in_batch_idx].detach().cpu()
        _save_2d_tensor_csv(
            raw_steering,
            os.path.join(debug_root, "raw_steering.csv"),
            header_prefix="raw",
        )
        payload["raw_steering_shape"] = list(raw_steering.shape)

    if "l2_context" in data_batch and torch.is_tensor(data_batch["l2_context"]):
        payload["l2_context_num_frames"] = int(data_batch["l2_context"][sample_in_batch_idx].shape[0])

    if condition_history:
        payload["condition_history"] = {}
        top_level_history_metadata = _save_rollout_condition_history_csv(
            condition_history=condition_history,
            sample_in_batch_idx=sample_in_batch_idx,
            csv_path=os.path.join(debug_root, "rollout_condition_history.csv"),
            key_path=("steering",),
            value_prefix="condition",
        )
        if top_level_history_metadata is not None:
            payload["condition_history"]["top_level"] = top_level_history_metadata

        delegated_l2_history_metadata = _save_rollout_condition_history_csv(
            condition_history=condition_history,
            sample_in_batch_idx=sample_in_batch_idx,
            csv_path=os.path.join(debug_root, "delegated_l2_rollout_condition_history.csv"),
            key_path=("_l2_condition_kwargs", "steering"),
            value_prefix="delegated_l2_condition",
        )
        if delegated_l2_history_metadata is not None:
            payload["condition_history"]["delegated_l2"] = delegated_l2_history_metadata

    if condition_kwargs:
        sample_payload = {}
        for key in ("steering", "_anchor_odo_index", "_rollout_step_odo", "_odometry_dt"):
            if key in condition_kwargs:
                value = condition_kwargs[key]
                if torch.is_tensor(value) and value.shape[0] > sample_in_batch_idx:
                    sample_payload[key] = _tensor_to_serializable(value[sample_in_batch_idx])
                else:
                    sample_payload[key] = _tensor_to_serializable(value)
        source_raw_speed_yaw = condition_kwargs.get("_source_raw_speed_yaw")
        if torch.is_tensor(source_raw_speed_yaw):
            source_raw_speed_yaw = source_raw_speed_yaw[sample_in_batch_idx].detach().cpu()
            _save_2d_tensor_csv(
                source_raw_speed_yaw,
                os.path.join(debug_root, "source_raw_steering.csv"),
                header_prefix="source_raw",
            )
            sample_payload["_source_raw_speed_yaw_shape"] = list(source_raw_speed_yaw.shape)
        transformed_raw_speed_yaw = condition_kwargs.get("_raw_speed_yaw")
        if torch.is_tensor(transformed_raw_speed_yaw):
            transformed_raw_speed_yaw = transformed_raw_speed_yaw[sample_in_batch_idx].detach().cpu()
            _save_2d_tensor_csv(
                transformed_raw_speed_yaw,
                os.path.join(debug_root, "transformed_raw_steering.csv"),
                header_prefix="transformed_raw",
            )
            sample_payload["_raw_speed_yaw_shape"] = list(transformed_raw_speed_yaw.shape)
        displacement_metadata = _maybe_save_displacement_trajectory(
            condition_kwargs,
            debug_root,
            sample_in_batch_idx,
            output_prefix="condition",
        )
        if displacement_metadata is not None:
            sample_payload["displacement_trajectory"] = displacement_metadata

        l2_condition_kwargs = condition_kwargs.get("_l2_condition_kwargs")
        if l2_condition_kwargs:
            l2_payload = {}
            for key in ("steering", "_anchor_odo_index", "_rollout_step_odo", "_odometry_dt"):
                if key not in l2_condition_kwargs:
                    continue
                value = l2_condition_kwargs[key]
                if torch.is_tensor(value) and value.shape[0] > sample_in_batch_idx:
                    l2_payload[key] = _tensor_to_serializable(value[sample_in_batch_idx])
                else:
                    l2_payload[key] = _tensor_to_serializable(value)
            l2_displacement_metadata = _maybe_save_displacement_trajectory(
                l2_condition_kwargs,
                debug_root,
                sample_in_batch_idx,
                output_prefix="delegated_l2_condition",
            )
            if l2_displacement_metadata is not None:
                l2_payload["displacement_trajectory"] = l2_displacement_metadata
            sample_payload["delegated_l2_condition_kwargs"] = l2_payload

            delegated_raw_speed_yaw = l2_condition_kwargs.get("_raw_speed_yaw")
            if torch.is_tensor(delegated_raw_speed_yaw):
                delegated_raw_speed_yaw = delegated_raw_speed_yaw[sample_in_batch_idx].detach().cpu()
                _save_2d_tensor_csv(
                    delegated_raw_speed_yaw,
                    os.path.join(debug_root, "delegated_l2_raw_speed_yaw.csv"),
                    header_prefix="delegated_l2_raw",
                )
                l2_payload["_raw_speed_yaw_shape"] = list(delegated_raw_speed_yaw.shape)

            delegated_source_raw_speed_yaw = l2_condition_kwargs.get("_source_raw_speed_yaw")
            if torch.is_tensor(delegated_source_raw_speed_yaw):
                delegated_source_raw_speed_yaw = delegated_source_raw_speed_yaw[sample_in_batch_idx].detach().cpu()
                _save_2d_tensor_csv(
                    delegated_source_raw_speed_yaw,
                    os.path.join(debug_root, "delegated_l2_source_raw_speed_yaw.csv"),
                    header_prefix="delegated_l2_source_raw",
                )
                l2_payload["_source_raw_speed_yaw_shape"] = list(delegated_source_raw_speed_yaw.shape)

        payload["condition_kwargs"] = sample_payload

    with open(os.path.join(debug_root, "condition_debug.json"), "w") as f:
        json.dump(payload, f, indent=2)


def get_steering_source_string(steering_file, no_steering=False):
    """Return a short label describing whether steering comes from data or a file."""
    if no_steering:
        return "none"
    if steering_file is None:
        return "from_data"
    return os.path.splitext(os.path.basename(steering_file))[0]


def get_steering_counterfactual_string(steering_file, speed_scale, yaw_rate_scale, no_steering=False):
    source = get_steering_source_string(steering_file, no_steering=no_steering)
    if float(speed_scale) == 1.0 and float(yaw_rate_scale) == 1.0:
        return source
    return f"{source}_speedx{float(speed_scale):g}_yawratex{float(yaw_rate_scale):g}"


def maybe_reconfigure_validation_odometry_horizon(config, model, num_condition_frames, num_gen_frames, rollout_steps):
    """Ask the validation dataset class to expose enough raw steering horizon for the rollout."""
    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None:
        return

    validation_config = config.data.params.validation
    custom_required_steps = getattr(condition_preprocessor, "get_required_rollout_odometry_steps", None)
    if callable(custom_required_steps):
        required_odo_steps = custom_required_steps(
            validation_params=validation_config.params,
            num_condition_frames=num_condition_frames,
            num_gen_frames=num_gen_frames,
            rollout_steps=rollout_steps,
        )
        if required_odo_steps is None:
            return
    else:
        max_condition_offset = condition_preprocessor.get_max_condition_odometry_offset()
        if max_condition_offset is None:
            return
        required_odo_steps = num_condition_frames + num_gen_frames + max_condition_offset - 1

    dataset_cls = get_obj_from_str(validation_config.target)
    reconfigure = getattr(dataset_cls, "reconfigure_params_for_required_odometry_horizon", None)
    if reconfigure is None:
        raise TypeError(
            f"Validation dataset {validation_config.target} must implement "
            "`reconfigure_params_for_required_odometry_horizon(...)` for odometry-conditioned rollout."
        )
    reconfigure(validation_config.params, required_odo_steps=required_odo_steps)


def get_rollout_future_frame_count(model, num_gen_frames):
    """Return the total number of future image frames produced by the rollout."""
    return int(num_gen_frames) * int(model.num_pred_frames)


def _rgb_to_cv2_color(color):
    """Convert an RGB tuple in [0, 1] to OpenCV BGR channel order."""
    return (color[2], color[1], color[0])


def _resolve_rollout_cursor_index(cursor_index, shared_trajectory, ctx_size, t, num_points):
    cursor_idx = 0
    if cursor_index is not None:
        cursor_idx = int(cursor_index.item())
    elif shared_trajectory and t >= ctx_size:
        cursor_idx = t - ctx_size
    return min(max(cursor_idx, 0), num_points - 1)


def _panel_coords_fit_trajectory(traj_xy, panel_w, panel_h, margin):
    if traj_xy.shape[0] == 0:
        return None

    forward = traj_xy[:, 0]
    lateral = traj_xy[:, 1]

    forward_min = np.min(forward)
    forward_max = np.max(forward)
    lateral_min = np.min(lateral)
    lateral_max = np.max(lateral)
    forward_span = forward_max - forward_min
    lateral_span = lateral_max - lateral_min
    usable_w = max(1, panel_w - 2 * margin)
    usable_h = max(1, panel_h - 2 * margin)

    scales = []
    if lateral_span > 1e-6:
        scales.append(usable_w / lateral_span)
    if forward_span > 1e-6:
        scales.append(usable_h / forward_span)
    scale = min(scales) if scales else 1.0

    if lateral_span > 1e-6:
        px = margin + (lateral_max - lateral) * scale
    else:
        px = np.full_like(lateral, panel_w / 2)
    if forward_span > 1e-6:
        py = margin + (forward_max - forward) * scale
    else:
        py = np.full_like(forward, panel_h / 2)
    px = np.clip(px, 0, panel_w - 1)
    py = np.clip(py, 0, panel_h - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def _transform_trajectory_to_ego_frame(traj_xy, traj_heading, cursor_idx):
    """Translate and rotate the trajectory so the current ego pose is at the origin facing +forward."""
    if traj_xy.shape[0] == 0:
        return traj_xy

    centered = traj_xy - traj_xy[cursor_idx : cursor_idx + 1]
    heading = float(traj_heading[cursor_idx])
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    rotation = np.array(
        [
            [cos_heading, sin_heading],
            [-sin_heading, cos_heading],
        ],
        dtype=np.float32,
    )
    return centered @ rotation.T


def _panel_coords_ego_frame(traj_xy, panel_w, panel_h, margin):
    if traj_xy.shape[0] == 0:
        return None

    forward = traj_xy[:, 0]
    lateral = traj_xy[:, 1]
    extent = max(
        float(np.max(np.abs(forward))),
        float(np.max(np.abs(lateral))),
        1e-3,
    )
    usable_w = max(1, panel_w - 2 * margin)
    usable_h = max(1, panel_h - 2 * margin)
    scale = min(usable_w, usable_h) / (2.0 * extent)
    center_x = panel_w / 2.0
    center_y = panel_h / 2.0

    px = center_x - lateral * scale
    py = center_y - forward * scale
    px = np.clip(px, 0, panel_w - 1)
    py = np.clip(py, 0, panel_h - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def overlay_trajectory_on_images(images, visualization, mode="trajectory"):
    """Draw a compact trajectory panel on top of each rollout frame."""
    cursor_index = None
    headings = None
    trajectory = visualization
    if isinstance(visualization, dict):
        trajectory = visualization.get("trajectory")
        cursor_index = visualization.get("cursor_index")
        headings = visualization.get("heading")

    if trajectory is None:
        return images

    if mode not in {"trajectory", "trajectory_ego"}:
        raise ValueError(f"Unsupported trajectory visualization mode: {mode}")

    shared_trajectory = trajectory.ndim == 3
    if trajectory.ndim == 3:
        trajectory = trajectory.unsqueeze(1).expand(-1, images.shape[1], -1, -1)
    elif trajectory.ndim != 4:
        raise ValueError(f"Expected trajectory with shape [B, T, N, 2] or [B, N, 2], got {tuple(trajectory.shape)}")
    if trajectory.shape[0] != images.shape[0] or trajectory.shape[1] != images.shape[1]:
        raise ValueError(
            f"Trajectory/image batch mismatch: images={tuple(images.shape)}, trajectory={tuple(trajectory.shape)}"
        )
    if cursor_index is not None:
        if not torch.is_tensor(cursor_index):
            cursor_index = torch.as_tensor(cursor_index, dtype=torch.long)
        if cursor_index.ndim == 1:
            cursor_index = cursor_index.unsqueeze(0).expand(images.shape[0], -1)
        if cursor_index.shape[0] != images.shape[0] or cursor_index.shape[1] != images.shape[1]:
            raise ValueError(
                "Cursor/image batch mismatch: "
                f"images={tuple(images.shape)}, cursor_index={tuple(cursor_index.shape)}"
            )
        cursor_index = cursor_index.to(device=trajectory.device)
    if mode == "trajectory_ego":
        if headings is None:
            raise ValueError("Ego-aligned trajectory visualization requires per-point heading data.")
        if headings.ndim == 2:
            headings = headings.unsqueeze(1).expand(-1, images.shape[1], -1)
        elif headings.ndim != 3:
            raise ValueError(f"Expected heading with shape [B, T, N] or [B, N], got {tuple(headings.shape)}")
        if headings.shape[0] != images.shape[0] or headings.shape[1] != images.shape[1]:
            raise ValueError(f"Heading/image batch mismatch: images={tuple(images.shape)}, heading={tuple(headings.shape)}")
    ctx_size = images.shape[1] - trajectory.shape[2] if shared_trajectory else 0

    height = images.shape[3]
    width = images.shape[4]

    panel_w = int(height * 0.35)
    panel_h = int(height * 0.35)
    margin = max(2, int(min(height, width) * 0.02))
    panel_x0 = margin
    panel_y0 = height - panel_h - margin

    traj_color = tuple(c * 2 - 1 for c in _rgb_to_cv2_color((0.0, 1.0, 0.0)))
    cursor_color = tuple(c * 2 - 1 for c in _rgb_to_cv2_color((1.0, 0.25, 0.0)))
    border_color = tuple(c * 2 - 1 for c in _rgb_to_cv2_color((1.0, 1.0, 1.0)))

    for b in range(images.shape[0]):
        for t in range(images.shape[1]):
            traj_bt = trajectory[b, t].detach().cpu().numpy()
            valid = ~np.isnan(traj_bt).any(axis=1)
            if not np.any(valid):
                continue

            traj_xy = traj_bt[valid, :2]
            cursor_idx = _resolve_rollout_cursor_index(
                None if cursor_index is None else cursor_index[b, t],
                shared_trajectory=shared_trajectory,
                ctx_size=ctx_size,
                t=t,
                num_points=traj_xy.shape[0],
            )

            if mode == "trajectory_ego":
                heading_bt = headings[b, t].detach().cpu().numpy()
                heading_valid = heading_bt[valid]
                ego_traj = _transform_trajectory_to_ego_frame(traj_xy, heading_valid, cursor_idx)
                traj_pts = _panel_coords_ego_frame(ego_traj, panel_w, panel_h, margin)
            else:
                traj_pts = _panel_coords_fit_trajectory(traj_xy, panel_w, panel_h, margin)

            if traj_pts is None:
                continue

            panel_t = np.full((panel_h, panel_w, 3), -1.0, dtype=np.float32)
            cv2.rectangle(panel_t, (0, 0), (panel_w - 1, panel_h - 1), border_color, 1)
            if traj_pts.shape[0] > 1:
                cv2.polylines(panel_t, [traj_pts.reshape(-1, 1, 2)], False, traj_color, 2)
            else:
                cv2.circle(panel_t, tuple(traj_pts[0]), 2, traj_color, -1)
            if mode == "trajectory_ego":
                ego_marker = np.array([panel_w / 2.0, panel_h / 2.0], dtype=np.float32).astype(np.int32)
                cv2.circle(panel_t, tuple(ego_marker), 3, cursor_color, -1)
            else:
                cv2.circle(panel_t, tuple(traj_pts[cursor_idx]), 3, cursor_color, -1)

            frame = images[b, t].permute(1, 2, 0).cpu().numpy().copy()
            frame[panel_y0:panel_y0 + panel_h, panel_x0:panel_x0 + panel_w] = panel_t
            images[b, t] = torch.from_numpy(frame).permute(2, 0, 1)

    return images


def load_steering_override(steering_file, expected_shape, dtype, device):
    """Load a raw steering override from disk and broadcast it to the batch shape."""
    if not os.path.isfile(steering_file):
        raise FileNotFoundError(f"Steering override file {steering_file} does not exist")

    if steering_file.endswith(".npy"):
        loaded = np.load(steering_file)
    elif steering_file.endswith(".csv"):
        loaded = np.loadtxt(steering_file, delimiter=",")
    else:
        raise ValueError("Steering override file must end with .npy or .csv")

    expected_b, expected_t, expected_d = expected_shape

    if loaded.ndim == 2:
        if loaded.shape[1] != expected_d:
            raise ValueError(
                f"Steering override feature dimension mismatch: got {loaded.shape[1]}, expected {expected_d}"
            )
        if loaded.shape[0] < expected_t:
            raise ValueError(
                f"Steering override has too few timesteps: got {loaded.shape[0]}, expected at least {expected_t}"
            )
        loaded = np.broadcast_to(loaded[:expected_t][None, :, :], (expected_b, expected_t, expected_d))
    elif loaded.ndim == 3:
        if loaded.shape[2] != expected_d:
            raise ValueError(
                f"Steering override feature dimension mismatch: got {loaded.shape[2]}, expected {expected_d}"
            )
        if loaded.shape[1] < expected_t:
            raise ValueError(
                f"Steering override has too few timesteps: got {loaded.shape[1]}, expected at least {expected_t}"
            )
        if loaded.shape[0] not in (1, expected_b):
            raise ValueError(
                f"Steering override batch dimension must be 1 or {expected_b}, got {loaded.shape[0]}"
            )
        loaded = loaded[:, :expected_t, :]
        if loaded.shape[0] == 1:
            loaded = np.broadcast_to(loaded, (expected_b, expected_t, expected_d))
    else:
        raise ValueError(
            f"Steering override must have shape [T, D] or [B, T, D], got {tuple(loaded.shape)}"
        )

    return torch.as_tensor(loaded, dtype=dtype, device=device)


def maybe_override_raw_steering(data_batch, steering_file):
    """Replace the batch steering tensor with values loaded from an override file."""
    if steering_file is None:
        return data_batch

    if "steering" not in data_batch:
        raise KeyError("Batch does not contain `steering`, cannot apply steering override")

    data_batch["steering"] = load_steering_override(
        steering_file=steering_file,
        expected_shape=tuple(data_batch["steering"].shape),
        dtype=data_batch["steering"].dtype,
        device=data_batch["steering"].device,
    )
    return data_batch


def maybe_drop_raw_steering(data_batch, no_steering):
    """Replace the batch steering tensor with NaNs to disable steering conditioning."""
    if not no_steering:
        return data_batch

    if "steering" not in data_batch:
        raise KeyError("Batch does not contain `steering`, cannot disable steering conditioning")

    data_batch["steering"] = torch.full_like(data_batch["steering"], torch.nan)
    return data_batch


def validate_steering_source(data_batch, steering_file, no_steering):
    """Raise if the batch carries only a placeholder steering tensor and no external source is given."""
    is_placeholder = data_batch.get("_steering_placeholder")
    if torch.is_tensor(is_placeholder):
        is_placeholder = bool(is_placeholder.any().item())
    if is_placeholder and steering_file is None and not no_steering:
        raise ValueError(
            "The validation dataset contains no steering data (placeholder only). "
            "Provide --steering_file <path> to supply steering, or pass --no_steering "
            "to run unconditionally."
        )


def move_batch_to_device(data_batch, device):
    """Move all tensor batch entries to the requested device."""
    moved = {}
    for key, value in data_batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_apply_condition_preprocessor_scales(model, speed_scale, yaw_rate_scale):
    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None:
        return

    if hasattr(condition_preprocessor, "speed_scale"):
        condition_preprocessor.speed_scale = float(speed_scale)
    if hasattr(condition_preprocessor, "yaw_rate_scale"):
        condition_preprocessor.yaw_rate_scale = float(yaw_rate_scale)


@torch.no_grad()
def generate_images(args, unknown_args):
    """Run v2 steering-conditioned rollout generation and save the resulting frames."""
    if args.seed > 0:
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown_args))
    model = instantiate_from_config(config.model)

    _ckpt_result = model.load_state_dict(torch.load(args.ckpt)["state_dict"], strict=False)
    _exempt_prefixes = tuple(getattr(model, "checkpoint_exempt_key_prefixes", ()))
    _unexpected_missing_keys = [k for k in _ckpt_result.missing_keys if not k.startswith(_exempt_prefixes)]
    assert _unexpected_missing_keys == [], _unexpected_missing_keys
    model = model.to(args.device)
    _ = model.eval()

    args.frames_dir = _maybe_append_l2_predictor_debug_subdir(args.frames_dir, model, args.save_l2_debug)
    if os.path.exists(args.frames_dir):
        print("Folder exist, new images will be saved to the same folder, delete it if you want to start from scratch")
    else:
        os.makedirs(args.frames_dir)

    if args.compile:
        def _maybe_compile(module, attr):
            net = getattr(module, attr, None)
            if net is not None:
                setattr(module, attr, torch.compile(net, mode=args.compile_mode))
                logger.info(f"Compiled {type(module).__name__}.{attr} with mode={args.compile_mode!r}")

        _maybe_compile(model, 'ema_vit' if args.evaluate_ema else 'vit')

        _l2_predictor = getattr(getattr(model, 'condition_preprocessor', None), 'l2_predictor', None)
        if _l2_predictor is not None:
            _maybe_compile(_l2_predictor, 'ema_vit')

        logger.info("First rollout step will be slow (compilation). Subsequent steps reuse the graph.")

    if args.compile and args.compile_artifacts:
        if os.path.exists(args.compile_artifacts):
            with open(args.compile_artifacts, "rb") as _f:
                torch.compiler.load_cache_artifacts(_f.read())
            logger.info(f"Loaded compile artifacts from {args.compile_artifacts!r}")
        else:
            logger.info(
                f"Compile artifacts not found at {args.compile_artifacts!r}; "
                "will save after the first batch."
            )

    maybe_apply_condition_preprocessor_scales(model, args.speed_scale, args.yaw_rate_scale)

    # Read num_frames from training config before val_config overrides it.
    # Training configs may have no validation section, or validation may be a list (not a dict),
    # in either case fall back to the first train dataset.
    try:
        _base_num_frames = OmegaConf.select(config, "data.params.validation.params.num_frames")
    except ConfigTypeError:
        _base_num_frames = None
    if _base_num_frames is None:
        _base_num_frames = config.data.params.train[0].params.num_frames
    num_condition_frames = _base_num_frames - model.num_pred_frames

    if args.val_config is not None:
        config = OmegaConf.merge(OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args))
    num_future_frames = get_rollout_future_frame_count(model, args.num_gen_frames)
    if args.save_real:
        config.data.params.validation.params.num_frames = num_condition_frames + num_future_frames
    maybe_reconfigure_validation_odometry_horizon(
        config=config,
        model=model,
        num_condition_frames=num_condition_frames,
        num_gen_frames=num_future_frames,
        rollout_steps=args.num_gen_frames,
    )
    if hasattr(config.data.params, "train"):
        del config.data.params.train

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_loader = data.val_dataloader()

    logger.info(
        f"Steering source: {get_steering_source_string(args.steering_file, no_steering=args.no_steering)}"
    )
    logger.info(f"Steering scales: speed={args.speed_scale:g}, yaw_rate={args.yaw_rate_scale:g}")
    logger.info(f"Saving generated images to {args.frames_dir}")

    sample_idx = 0
    progress_bar = tqdm(range(len(val_loader.dataset) // val_loader.batch_size))
    loader_iter = iter(val_loader)

    for _batch_idx, _ in enumerate(progress_bar):
        data_batch = next(loader_iter)
        if args.num_videos is not None and sample_idx >= args.num_videos:
            break

        data_batch = move_batch_to_device(data_batch, args.device)
        validate_steering_source(data_batch, args.steering_file, args.no_steering)
        data_batch = maybe_override_raw_steering(data_batch, args.steering_file)
        data_batch = maybe_drop_raw_steering(data_batch, args.no_steering)

        x = data_batch["images"]
        cond_x = x[:, :num_condition_frames]
        frame_rate = data_batch.get("frame_rate")
        rollout_context = {"images": cond_x}
        for key, value in data_batch.items():
            if key != "images" and key != "steering":
                rollout_context[key] = value
        condition_batch = dict(data_batch)
        condition_batch["images"] = cond_x
        condition_kwargs = model.condition_preprocessor.get_condition_kwargs_from_batch(
            condition_batch,
            split="rollout",
        )

        autocast_enabled = args.device.startswith("cuda")
        condition_history = None
        capture_condition_history = args.save_l2_debug or args.save_condition_debug
        with torch.autocast(dtype=torch.float16, device_type="cuda", enabled=autocast_enabled):
            rollout_result = model.roll_out(
                x_0=rollout_context,
                num_gen_frames=args.num_gen_frames,
                latent_input=False,
                NFE=args.num_steps,
                eta=args.eta,
                sample_with_ema=args.evaluate_ema,
                num_samples=cond_x.size(0),
                frame_rate=frame_rate,
                condition_kwargs=condition_kwargs,
                decode_device=args.decode_device,
                return_condition_history=capture_condition_history,
                num_condition_frames=cond_x.size(1),
            )
            if capture_condition_history:
                _latents, gen_frames, condition_history = rollout_result
            else:
                _latents, gen_frames = rollout_result
            real_frames_enc_dec = None

        if args.vis_mode in {"trajectory", "trajectory_ego"}:
            overlay_trajectory = model.condition_preprocessor.get_rollout_visualization_trajectory(
                condition_kwargs=model.condition_preprocessor.get_condition_kwargs_from_batch(
                    condition_batch,
                    split="rollout",
                ),
                num_condition_frames=num_condition_frames,
                num_gen_steps=args.num_gen_frames,
                num_pred_frames=model.num_pred_frames,
            )
            if overlay_trajectory is not None:
                gen_frames = overlay_trajectory_on_images(gen_frames, overlay_trajectory, mode=args.vis_mode)

        condition_debug_kwargs = condition_kwargs
        # Release rollout-time latent state before CPU-side file I/O.
        del _latents, condition_kwargs, rollout_context, condition_batch, cond_x

        for sample_in_batch_idx in range(gen_frames.shape[0]):
            subfolder_path_fake = os.path.join(args.frames_dir, "fake_images", f"sequence_{sample_idx:04d}")
            subfolder_path_gifs = os.path.join(args.frames_dir, "gen_gifs")
            if not os.path.exists(subfolder_path_fake):
                os.makedirs(subfolder_path_fake)
            if not os.path.exists(subfolder_path_gifs):
                os.makedirs(subfolder_path_gifs)

            for f in range(gen_frames.shape[1]):
                save_image(
                    (gen_frames[sample_in_batch_idx, f] + 1.0) / 2.0,
                    os.path.join(subfolder_path_fake, f"frame_{f:04d}.jpg"),
                )

            imageio.mimsave(
                os.path.join(subfolder_path_gifs, f"sequence_{sample_idx:04d}.gif"),
                [
                    np.array(Image.open(os.path.join(subfolder_path_fake, f"frame_{f:04d}.jpg")))
                    for f in range(gen_frames.shape[1])
                ],
                fps=args.frame_rate if frame_rate is not None else 7,
                loop=0,
            )

            if args.save_real:
                subfolder_path_real = os.path.join(args.frames_dir, "real_images", f"sequence_{sample_idx:04d}")
                if not os.path.exists(subfolder_path_real):
                    os.makedirs(subfolder_path_real)
                for f in range(x.shape[1]):
                    save_image(
                        (x[sample_in_batch_idx, f] + 1.0) / 2.0,
                        os.path.join(subfolder_path_real, f"frame_{f:04d}.jpg"),
                    )

            if args.save_l2_debug:
                save_l2_debug_images(model, condition_history, args.frames_dir, sample_in_batch_idx, sample_idx)
            if args.save_condition_debug:
                save_condition_debug_dump(
                    data_batch,
                    condition_debug_kwargs,
                    condition_history,
                    args.frames_dir,
                    sample_in_batch_idx,
                    sample_idx,
                )

            sample_idx += 1

        progress_bar.set_description(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.02f} GB")

        if _batch_idx == 0 and args.compile and args.compile_artifacts and not os.path.exists(args.compile_artifacts):
            _artifacts = torch.compiler.save_cache_artifacts()
            if _artifacts is not None:
                with open(args.compile_artifacts, "wb") as _f:
                    _f.write(_artifacts[0])
                logger.info(f"Saved compile artifacts to {args.compile_artifacts!r}")


def main(args, unknown_args):
    """Entrypoint that launches rollout generation with resolved CLI arguments."""
    generate_images(args, unknown_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None, help="Path to the experiment directory, where the config and checkpoints are stored")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt", help="Path to the checkpoint file, relative to exp_dir")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file, relative to exp_dir")
    parser.add_argument("--val_config", type=str, default=None, help="Path to the validation data config file")
    parser.add_argument(
        "--num_gen_frames",
        type=int,
        default=1,
        help="Number of rollout steps to generate; each step predicts `model.num_pred_frames` future frames.",
    )
    parser.add_argument("--frames_dir", type=str, default=None, help="Path of the folder for the real and fake frames, relative to exp_dir")
    parser.add_argument("--save_real", type=str2bool, default=True, help="Save real frames in the same folder as fake frames")
    parser.add_argument("--num_videos", type=int, default=None, help="Number of videos to generate")
    parser.add_argument("--save_l2_debug", type=str2bool, default=False, help="Save decoded L2 context/start/end conditioning for each rollout step")
    parser.add_argument(
        "--save_condition_debug",
        type=str2bool,
        default=False,
        help="Save raw steering and delegated condition-preprocessor state for each rollout sample",
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="none",
        choices=["none", "trajectory", "trajectory_ego"],
        help="Visualization mode",
    )
    parser.add_argument("--steering_file", type=str, default=None, help="Optional .npy or .csv file used to replace raw batch steering")
    parser.add_argument(
        "--no_steering",
        type=str2bool,
        default=False,
        help="Replace the raw steering sequence with NaNs before conditioning.",
    )
    parser.add_argument("--speed_scale", type=float, default=1.0, help="Global multiplicative factor applied to raw speed conditioning")
    parser.add_argument("--yaw_rate_scale", type=float, default=1.0, help="Global multiplicative factor applied to raw yaw-rate conditioning")
    parser.add_argument("--frame_rate", type=int, default=7, help="Frame rate for the generated GIFs")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--decode_device",
        type=str,
        default="cpu",
        help="Device used for decoded rollout frames. Use 'cpu' to reduce peak GPU memory during saving.",
    )
    parser.add_argument("--num_steps", type=int, default=30, help="Number of steps for sampling")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity for sampling")
    parser.add_argument("--evaluate_ema", "--use_ema", type=str2bool, default=True, help="If the evaluation happen with ema model")
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=False,
        help="Wrap the DiT network with torch.compile for faster inference (PyTorch 2.x).",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode. 'reduce-overhead' uses CUDA graphs; 'max-autotune' adds kernel autotuning.",
    )
    parser.add_argument(
        "--compile_artifacts",
        type=str,
        default=None,
        help=(
            "Path to torch.compiler cache artifacts (.pkl), relative to --exp_dir. "
            "If the file exists, artifacts are loaded before rollout (fast startup). "
            "If it does not exist, artifacts are saved after rollout (for future runs). "
            "Only effective when --compile is True."
        ),
    )

    args, unknown = parser.parse_known_args()

    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)
    if args.compile_artifacts:
        args.compile_artifacts = os.path.join(args.exp_dir, args.compile_artifacts)

    if args.frames_dir is None:
        epoch, global_step = get_ckpt_epoch_step(args.ckpt)
        args.frames_dir = os.path.join(
            "gen_rollout",
            os.path.basename(args.val_config).split(".")[0] if args.val_config is not None else "default_data",
            f"ep{epoch}iter{global_step}_{args.num_steps}steps",
            f"steering_{get_steering_counterfactual_string(args.steering_file, args.speed_scale, args.yaw_rate_scale, no_steering=args.no_steering)}",
            f"vis_{args.vis_mode}",
            f"seed{args.seed}",
        )
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)

    main(args, unknown)
