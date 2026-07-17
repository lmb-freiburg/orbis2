import argparse

import cv2
import numpy as np
import torch

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False


def get_rollout_future_frame_count(model, num_gen_frames):
    """Return the total number of future image frames produced by the rollout."""
    return int(num_gen_frames) * int(model.num_pred_frames)


def resolve_video_backend():
    """Pick decord as the primary video backend, falling back to the slower torchcodec if unavailable."""
    if DECORD_AVAILABLE:
        return "decord"
    if TORCHCODEC_AVAILABLE:
        return "torchcodec"
    raise ImportError("Either `decord` or `torchcodec` must be installed to read video frames.")


def get_video_fps_and_length(path, backend):
    """Return (native_fps, num_frames) for a video file."""
    if backend == "decord":
        reader = VideoReader(path, ctx=decord_cpu(0))
        return float(reader.get_avg_fps()), len(reader)
    decoder = VideoDecoder(path)
    return float(decoder.metadata.average_fps), int(decoder.metadata.num_frames)


def decode_video_frames(path, indices, backend):
    """Decode an arbitrary list of frame indices from a video file."""
    if backend == "decord":
        reader = VideoReader(path, ctx=decord_cpu(0))
        return reader.get_batch(indices).asnumpy()
    decoder = VideoDecoder(path)
    return torch.stack([decoder[i] for i in indices])


def compute_frame_interval(native_fps, target_frame_rate, label):
    """Return the integer native-frame stride for `target_frame_rate`, or raise if not exact."""
    ratio = native_fps / float(target_frame_rate)
    rounded = round(ratio)
    if abs(ratio - rounded) > 1e-6:
        raise ValueError(
            f"The video's native frame rate ({native_fps:g} Hz) must be an integer multiple of "
            f"{label} ({target_frame_rate:g} Hz), got ratio={ratio:g}."
        )
    return int(rounded)


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


def maybe_apply_condition_preprocessor_scales(model, speed_scale, yaw_rate_scale):
    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None:
        return

    if hasattr(condition_preprocessor, "speed_scale"):
        condition_preprocessor.speed_scale = float(speed_scale)
    if hasattr(condition_preprocessor, "yaw_rate_scale"):
        condition_preprocessor.yaw_rate_scale = float(yaw_rate_scale)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
