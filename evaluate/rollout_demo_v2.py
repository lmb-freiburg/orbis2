import argparse
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
from scipy.interpolate import CubicSpline
from torchvision.utils import save_image

from data.l2_context import L2ContextMixin
from data.video_loaders import ClipAugmenter, DecordFrameAdapter, ResizeCenterPolicy, TensorFrameAdapter
from util import instantiate_from_config

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


logger = logging.getLogger(__name__)

STEERING_FORMAT = "speed_yawrate"


class _L1L2FrameIndexer(L2ContextMixin):
    """Computes L1/L2 frame indices into a single source video via `L2ContextMixin`."""

    def __init__(self, frame_interval, stored_data_frame_rate, num_l2_context, l2_frame_rate, l1_context_frames):
        self.frame_interval = frame_interval
        self.stored_data_frame_rate = stored_data_frame_rate
        self._init_l2_context(
            num_l2_context=num_l2_context,
            l2_frame_rate=l2_frame_rate,
            l1_context_frames=l1_context_frames,
            require_l2_context=True,
        )


def get_rollout_future_frame_count(model, num_gen_frames):
    """Return the total number of future image frames produced by the rollout."""
    return int(num_gen_frames) * int(model.num_pred_frames)


def resolve_context_image_size(args, config):
    """Return (height, width) to resize context images to, from CLI args or the training config."""
    if args.height is not None and args.width is not None:
        return int(args.height), int(args.width)

    try:
        size = OmegaConf.select(config, "data.params.validation.params.size")
    except ConfigTypeError:
        size = None
    if size is None:
        size = config.data.params.train[0].params.size
    size = (size, size) if isinstance(size, int) else tuple(size)

    height = int(args.height) if args.height is not None else int(size[0])
    width = int(args.width) if args.width is not None else int(size[1])
    return height, width


def require_l1l2_model(model):
    """Raise a clear error if the loaded model isn't an L1-L2 hierarchical model."""
    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None or not hasattr(condition_preprocessor, "num_context_frames") \
            or not hasattr(condition_preprocessor, "l2_predictor_frame_rate"):
        raise TypeError(
            "This script requires an L1-L2 hierarchical model, i.e. a `condition_preprocessor` "
            "with a frozen `l2_predictor` (num_context_frames/l2_predictor_frame_rate). "
            f"Got {type(condition_preprocessor).__name__ if condition_preprocessor else None}."
        )


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


def resolve_l2_frame_rate(args, model):
    """Return the L2 sampling rate, defaulting to (and validating against) the frozen L2 predictor's own rate."""
    model_rate = model.condition_preprocessor.l2_predictor_frame_rate
    if args.l2_frame_rate is None:
        return float(model_rate)
    if abs(float(args.l2_frame_rate) - float(model_rate)) > 1e-6:
        raise ValueError(
            f"--l2_frame_rate={args.l2_frame_rate:g} does not match the frozen L2 predictor's "
            f"trained frame rate ({model_rate:g}); the L2 image context must be sampled at the "
            "rate the L2 predictor was trained on."
        )
    return float(args.l2_frame_rate)


def load_l1_l2_context(video_path, start_frame, l1_frame_rate, l2_frame_rate, l1_context_frames, l2_context_frames, height, width, backend, device):
    """Sample L1 (high-rate) and L2 (low-rate, further back) context windows from one video."""
    native_fps, video_length = get_video_fps_and_length(video_path, backend)
    frame_interval = compute_frame_interval(native_fps, l1_frame_rate, "--l1_frame_rate")
    compute_frame_interval(native_fps, l2_frame_rate, "--l2_frame_rate")

    indexer = _L1L2FrameIndexer(
        frame_interval=frame_interval,
        stored_data_frame_rate=native_fps,
        num_l2_context=l2_context_frames,
        l2_frame_rate=l2_frame_rate,
        l1_context_frames=l1_context_frames,
    )

    l1_span = (l1_context_frames - 1) * frame_interval + 1
    if start_frame is None:
        start_frame = video_length - l1_span
    if start_frame < 0 or start_frame + l1_span > video_length:
        raise ValueError(
            f"Video is too short for the requested L1 context: need frames [{start_frame}, "
            f"{start_frame + l1_span - 1}], but the video only has {video_length} frames."
        )

    required_offset = indexer.get_required_l1_start_offset()
    if start_frame < required_offset:
        raise ValueError(
            f"Video does not have enough lookback for L2 context: start_frame={start_frame} but "
            f"at least {required_offset} frames of history are required before it. "
            "Use a longer video, a later --start_frame, or a lower --l2_frame_rate."
        )

    l1_indices, l2_indices = indexer.get_l1_and_l2_indices(start_frame, l1_context_frames)
    all_frames = decode_video_frames(video_path, l1_indices + l2_indices, backend)

    adapter = DecordFrameAdapter() if backend == "decord" else TensorFrameAdapter()
    augmenter = ClipAugmenter(adapter, ResizeCenterPolicy((height, width)))
    all_tensor = augmenter(all_frames)  # [F, C, H, W] in [-1, 1]

    l1_tensor = all_tensor[: len(l1_indices)].unsqueeze(0).to(device)
    l2_tensor = all_tensor[len(l1_indices) :].unsqueeze(0).to(device)
    return l1_tensor, l2_tensor


def load_steering_trajectory(steering_file, min_odo_steps, dtype, device):
    """Load a raw [speed, yaw_rate] steering trajectory from a .npy/.csv file as a [1, T, 2] tensor."""
    if not os.path.isfile(steering_file):
        raise FileNotFoundError(f"Steering file {steering_file} does not exist")

    if steering_file.endswith(".npy"):
        loaded = np.load(steering_file)
    elif steering_file.endswith(".csv"):
        loaded = np.loadtxt(steering_file, delimiter=",")
    else:
        raise ValueError("Steering file must end with .npy or .csv")

    if loaded.ndim != 2 or loaded.shape[1] != 2:
        raise ValueError(
            f"Steering file must contain [T, 2] (speed, yaw_rate) rows, got shape {tuple(loaded.shape)}"
        )
    if min_odo_steps is not None and loaded.shape[0] < min_odo_steps:
        raise ValueError(
            f"Steering file has too few timesteps: got {loaded.shape[0]}, expected at least {min_odo_steps}"
        )

    return torch.as_tensor(loaded, dtype=dtype, device=device).unsqueeze(0)


def parse_spline_points(spline_points_str):
    """Parse 'x1,y1;x2,y2;...' into an [N, 2] float array (forward, lateral meters)."""
    try:
        points = np.array(
            [[float(v) for v in pair.split(",")] for pair in spline_points_str.split(";")],
            dtype=np.float64,
        )
    except ValueError as e:
        raise ValueError(
            f"--spline_points must be formatted as 'x1,y1;x2,y2;...', got {spline_points_str!r}"
        ) from e
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"--spline_points must be formatted as 'x1,y1;x2,y2;...', got {spline_points_str!r}"
        )
    if points.shape[0] < 2:
        raise ValueError(f"--spline_points must contain at least 2 waypoints, got {points.shape[0]}")
    return points


def build_steering_from_spline(spline_points_str, min_odo_steps, anchor_odo_index, dt, dtype, device):
    """Fit a natural cubic spline through inline waypoints and sample it into a
    [1, min_odo_steps, 2] (speed, yaw_rate) tensor, spacing the waypoints evenly across
    the post-anchor rollout horizon. Indices before `anchor_odo_index` are never read by
    the condition preprocessor, so they are left zero-filled.
    """
    points = parse_spline_points(spline_points_str)
    tail_len = int(min_odo_steps) - int(anchor_odo_index)
    if tail_len < 2:
        raise ValueError(
            f"Not enough rollout horizon to fit a spline: tail_len={tail_len} "
            f"(min_odo_steps={min_odo_steps}, anchor_odo_index={anchor_odo_index}), need >= 2."
        )

    waypoint_times = np.linspace(0.0, (tail_len - 1) * dt, points.shape[0])
    spline_x = CubicSpline(waypoint_times, points[:, 0], bc_type="natural")
    spline_y = CubicSpline(waypoint_times, points[:, 1], bc_type="natural")

    sample_times = np.arange(tail_len) * dt
    dx, dy = spline_x(sample_times, 1), spline_y(sample_times, 1)
    ddx, ddy = spline_x(sample_times, 2), spline_y(sample_times, 2)

    speed = np.hypot(dx, dy)
    yaw_rate = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)  # curvature * speed

    full = np.zeros((int(min_odo_steps), 2), dtype=np.float32)
    full[int(anchor_odo_index):] = np.stack([speed, yaw_rate], axis=1)
    return torch.as_tensor(full, dtype=dtype, device=device).unsqueeze(0)


def make_unconditional_steering(min_odo_steps, dtype, device):
    """Build an all-NaN [1, T, 2] steering placeholder, the framework's own 'no steering data' signal."""
    if min_odo_steps is None:
        raise ValueError(
            "Cannot run unconditionally without --steering_file: the loaded condition_preprocessor "
            "could not report a required odometry length (`get_required_rollout_odometry_steps` "
            "returned None), so there is no safe length to build a NaN placeholder at. "
            "Supply --steering_file explicitly."
        )
    return torch.full((1, int(min_odo_steps), 2), float("nan"), dtype=dtype, device=device)


def maybe_apply_condition_preprocessor_scales(model, speed_scale, yaw_rate_scale):
    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None:
        return

    if hasattr(condition_preprocessor, "speed_scale"):
        condition_preprocessor.speed_scale = float(speed_scale)
    if hasattr(condition_preprocessor, "yaw_rate_scale"):
        condition_preprocessor.yaw_rate_scale = float(yaw_rate_scale)


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


@torch.no_grad()
def generate_images(args, unknown_args):
    """Run an L1-L2 hierarchical rollout from a single input video and save the resulting frames."""
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

    require_l1l2_model(model)

    if os.path.exists(args.output_dir):
        print("Folder exists, new images will be saved to the same folder, delete it if you want to start from scratch")
    else:
        os.makedirs(args.output_dir)

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

    height, width = resolve_context_image_size(args, config)
    backend = resolve_video_backend()
    l2_frame_rate = resolve_l2_frame_rate(args, model)

    l1_context_frames = int(model.vit.num_context_frames)
    l2_context_frames = int(model.condition_preprocessor.num_context_frames)

    l1_tensor, l2_tensor = load_l1_l2_context(
        video_path=args.video,
        start_frame=args.start_frame,
        l1_frame_rate=args.l1_frame_rate,
        l2_frame_rate=l2_frame_rate,
        l1_context_frames=l1_context_frames,
        l2_context_frames=l2_context_frames,
        height=height,
        width=width,
        backend=backend,
        device=args.device,
    )

    num_future_frames = get_rollout_future_frame_count(model, args.num_gen_frames)
    frame_rate = torch.tensor(float(args.l1_frame_rate), device=args.device)
    data_batch = {"images": l1_tensor, "l2_context": l2_tensor, "frame_rate": frame_rate}

    if args.steering_file is not None and args.spline_points is not None:
        raise ValueError("--steering_file and --spline_points are mutually exclusive.")

    get_required_steps = getattr(model.condition_preprocessor, "get_required_rollout_odometry_steps", None)
    min_odo_steps = None
    if callable(get_required_steps):
        min_odo_steps = get_required_steps(
            validation_params=None,
            num_condition_frames=l1_context_frames,
            num_gen_frames=num_future_frames,
            rollout_steps=args.num_gen_frames,
        )
    if args.steering_file is not None:
        data_batch["steering"] = load_steering_trajectory(
            args.steering_file, min_odo_steps, dtype=l1_tensor.dtype, device=args.device
        )
    elif args.spline_points is not None:
        odometry_steps_per_image_frame = getattr(model.condition_preprocessor, "odometry_steps_per_image_frame", 1)
        anchor_frame_index = getattr(model.condition_preprocessor, "context_frame_anchor_index", -1)
        if anchor_frame_index < 0:
            anchor_frame_index += l1_context_frames
        anchor_odo_index = anchor_frame_index * odometry_steps_per_image_frame
        dt = 1.0 / (args.l1_frame_rate * odometry_steps_per_image_frame)
        data_batch["steering"] = build_steering_from_spline(
            args.spline_points, min_odo_steps, anchor_odo_index, dt, dtype=l1_tensor.dtype, device=args.device
        )
    else:
        data_batch["steering"] = make_unconditional_steering(min_odo_steps, dtype=l1_tensor.dtype, device=args.device)
    data_batch["steering_format"] = STEERING_FORMAT

    condition_kwargs = model.condition_preprocessor.get_condition_kwargs_from_batch(data_batch, split="rollout")

    if args.steering_file is not None:
        steering_source = args.steering_file
    elif args.spline_points is not None:
        steering_source = f"spline({args.spline_points})"
    else:
        steering_source = "none"
    logger.info(f"Steering source: {steering_source}")
    logger.info(f"Steering scales: speed={args.speed_scale:g}, yaw_rate={args.yaw_rate_scale:g}")
    logger.info(f"L1/L2 frame rates: {args.l1_frame_rate:g}/{l2_frame_rate:g} Hz")
    logger.info(f"Saving generated images to {args.output_dir}")

    autocast_enabled = args.device.startswith("cuda")
    with torch.autocast(dtype=torch.float16, device_type="cuda", enabled=autocast_enabled):
        _latents, gen_frames = model.roll_out(
            x_0={"images": l1_tensor},
            num_gen_frames=args.num_gen_frames,
            latent_input=False,
            NFE=args.num_steps,
            eta=args.eta,
            sample_with_ema=args.evaluate_ema,
            num_samples=l1_tensor.size(0),
            frame_rate=frame_rate.unsqueeze(0),
            condition_kwargs=condition_kwargs,
            decode_device=args.decode_device,
            num_condition_frames=l1_tensor.size(1),
        )

    if args.vis_mode in {"trajectory", "trajectory_ego"}:
        overlay_trajectory = model.condition_preprocessor.get_rollout_visualization_trajectory(
            condition_kwargs=model.condition_preprocessor.get_condition_kwargs_from_batch(data_batch, split="rollout"),
            num_condition_frames=l1_context_frames,
            num_gen_steps=args.num_gen_frames,
            num_pred_frames=model.num_pred_frames,
        )
        if overlay_trajectory is not None:
            gen_frames = overlay_trajectory_on_images(gen_frames, overlay_trajectory, mode=args.vis_mode)

    # Release rollout-time latent state before CPU-side file I/O.
    del _latents, condition_kwargs, l1_tensor, l2_tensor

    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(gen_frames.shape[1]):
        save_image(
            (gen_frames[0, f] + 1.0) / 2.0,
            os.path.join(frames_dir, f"frame_{f:04d}.jpg"),
        )

    imageio.mimsave(
        os.path.join(args.output_dir, "rollout.gif"),
        [
            np.array(Image.open(os.path.join(frames_dir, f"frame_{f:04d}.jpg")))
            for f in range(gen_frames.shape[1])
        ],
        fps=args.l1_frame_rate,
        loop=0,
    )

    if args.device.startswith("cuda"):
        logger.info(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.02f} GB")

    if args.compile and args.compile_artifacts and not os.path.exists(args.compile_artifacts):
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
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file to sample L1/L2 context from.")
    parser.add_argument("--l1_frame_rate", type=float, required=True, help="Frame rate (Hz) to sample L1 context/rollout frames at, and the generated GIF's fps.")
    parser.add_argument("--l2_frame_rate", type=float, default=None, help="Frame rate (Hz) to sample L2 context frames at. Defaults to the frozen L2 predictor's own trained frame rate; must match it if given explicitly.")
    parser.add_argument("--start_frame", type=int, default=None, help="Native-video frame index to start the L1 context window at. Defaults to the latest window that fits (the end of the video).")
    parser.add_argument("--height", type=int, default=None, help="Height to resize context images to. Defaults to the training config's size.")
    parser.add_argument("--width", type=int, default=None, help="Width to resize context images to. Defaults to the training config's size.")
    parser.add_argument(
        "--num_gen_frames",
        type=int,
        default=1,
        help="Number of rollout steps to generate; each step predicts `model.num_pred_frames` future frames.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated frames and GIF to.")
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="none",
        choices=["none", "trajectory", "trajectory_ego"],
        help="Visualization mode",
    )
    parser.add_argument("--steering_file", type=str, default=None, help="Optional .npy or .csv trajectory file (columns: speed, yaw_rate), already at the expected odometry rate, used as steering input.")
    parser.add_argument(
        "--spline_points",
        type=str,
        default=None,
        help="Inline waypoints 'x1,y1;x2,y2;...' (forward,lateral meters, local ego frame) defining a path "
             "via a natural cubic spline; converted to speed/yaw_rate and spread evenly across the rollout "
             "duration. Mutually exclusive with --steering_file.",
    )
    parser.add_argument("--speed_scale", type=float, default=1.0, help="Global multiplicative factor applied to raw speed conditioning")
    parser.add_argument("--yaw_rate_scale", type=float, default=1.0, help="Global multiplicative factor applied to raw yaw-rate conditioning")

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

    main(args, unknown)
