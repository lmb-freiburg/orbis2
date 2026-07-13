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
from torchvision.utils import save_image

from data.video_loaders import ClipAugmenter, PILFrameAdapter, ResizeCenterPolicy
from util import instantiate_from_config


logger = logging.getLogger(__name__)

STEERING_FORMAT = "speed_yawrate"


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


def load_context_images(image_paths, height, width, device):
    """Load and preprocess raw image files into a [1, F, C, H, W] tensor in [-1, 1]."""
    frames = [Image.open(path).convert("RGB") for path in image_paths]
    augmenter = ClipAugmenter(PILFrameAdapter(), ResizeCenterPolicy((height, width)))
    return augmenter(frames).unsqueeze(0).to(device)


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
    """Run a v2 steering-conditioned rollout from raw context images and save the resulting frames."""
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
    num_condition_frames = len(args.images)
    num_future_frames = get_rollout_future_frame_count(model, args.num_gen_frames)

    cond_x = load_context_images(args.images, height, width, args.device)
    frame_rate = torch.tensor(float(args.frame_rate), device=args.device)
    data_batch = {"images": cond_x, "frame_rate": frame_rate}

    if args.steering_file is not None:
        get_required_steps = getattr(model.condition_preprocessor, "get_required_rollout_odometry_steps", None)
        min_odo_steps = None
        if callable(get_required_steps):
            min_odo_steps = get_required_steps(
                validation_params=None,
                num_condition_frames=num_condition_frames,
                num_gen_frames=num_future_frames,
                rollout_steps=args.num_gen_frames,
            )
        data_batch["steering"] = load_steering_trajectory(
            args.steering_file, min_odo_steps, dtype=cond_x.dtype, device=args.device
        )
        data_batch["steering_format"] = STEERING_FORMAT

    condition_kwargs = model.condition_preprocessor.get_condition_kwargs_from_batch(data_batch, split="rollout")

    logger.info(f"Steering source: {'none' if args.steering_file is None else args.steering_file}")
    logger.info(f"Steering scales: speed={args.speed_scale:g}, yaw_rate={args.yaw_rate_scale:g}")
    logger.info(f"Saving generated images to {args.output_dir}")

    autocast_enabled = args.device.startswith("cuda")
    with torch.autocast(dtype=torch.float16, device_type="cuda", enabled=autocast_enabled):
        _latents, gen_frames = model.roll_out(
            x_0={"images": cond_x},
            num_gen_frames=args.num_gen_frames,
            latent_input=False,
            NFE=args.num_steps,
            eta=args.eta,
            sample_with_ema=args.evaluate_ema,
            num_samples=cond_x.size(0),
            frame_rate=frame_rate,
            condition_kwargs=condition_kwargs,
            decode_device=args.decode_device,
            num_condition_frames=cond_x.size(1),
        )

    if args.vis_mode in {"trajectory", "trajectory_ego"}:
        overlay_trajectory = model.condition_preprocessor.get_rollout_visualization_trajectory(
            condition_kwargs=model.condition_preprocessor.get_condition_kwargs_from_batch(data_batch, split="rollout"),
            num_condition_frames=num_condition_frames,
            num_gen_steps=args.num_gen_frames,
            num_pred_frames=model.num_pred_frames,
        )
        if overlay_trajectory is not None:
            gen_frames = overlay_trajectory_on_images(gen_frames, overlay_trajectory, mode=args.vis_mode)

    # Release rollout-time latent state before CPU-side file I/O.
    del _latents, condition_kwargs, cond_x

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
        fps=args.frame_rate,
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
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the context image files, in temporal order (oldest first).",
    )
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
    parser.add_argument("--steering_file", type=str, default=None, help="Optional .npy or .csv trajectory file (columns: speed, yaw_rate) used as steering input. If omitted, the rollout is unconditional.")
    parser.add_argument("--speed_scale", type=float, default=1.0, help="Global multiplicative factor applied to raw speed conditioning")
    parser.add_argument("--yaw_rate_scale", type=float, default=1.0, help="Global multiplicative factor applied to raw yaw-rate conditioning")
    parser.add_argument("--frame_rate", type=float, default=7, help="Frame rate (Hz) of the input context images, used for steering alignment and as the generated GIF's fps.")

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
