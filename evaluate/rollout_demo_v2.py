import argparse
import os
import sys
import imageio
import logging
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.utils import save_image

from data.l2_context import L2ContextMixin
from data.video_loaders import ClipAugmenter, DecordFrameAdapter, ResizeCenterPolicy, TensorFrameAdapter
from evaluate.utils import (
    compute_frame_interval,
    decode_video_frames,
    get_rollout_future_frame_count,
    get_video_fps_and_length,
    maybe_apply_condition_preprocessor_scales,
    overlay_trajectory_on_images,
    resolve_video_backend,
    str2bool,
)

from util import instantiate_from_config

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


def resolve_context_image_size(config):
    """Return (height, width) to resize context images to, from the inference config."""
    size = OmegaConf.select(config, "data.params.validation.params.size")
    if size is None:
        raise ValueError("Config is missing `data.params.validation.params.size`.")
    size = (size, size) if isinstance(size, int) else tuple(size)
    return int(size[0]), int(size[1])


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


def resolve_l2_frame_rate(model):
    """Return the L2 sampling rate: the frozen L2 predictor's own trained rate."""
    return float(model.condition_preprocessor.l2_predictor_frame_rate)


def resolve_l1_frame_rate(config):
    """Return the L1 (context/rollout) frame rate (Hz) declared in the inference config."""
    frame_rate = OmegaConf.select(config, "data.params.validation.params.frame_rate")
    if frame_rate is None:
        raise ValueError("Config is missing `data.params.validation.params.frame_rate`.")
    return float(frame_rate)


def load_l1_l2_context(video_path, start_frame, l1_frame_rate, l2_frame_rate, l1_context_frames, l2_context_frames, height, width, backend, device):
    """Sample L1 (high-rate) and L2 (low-rate, further back) context windows from one video."""
    native_fps, video_length = get_video_fps_and_length(video_path, backend)
    frame_interval = compute_frame_interval(native_fps, l1_frame_rate, "the L1 frame rate (data.params.validation.params.frame_rate)")
    compute_frame_interval(native_fps, l2_frame_rate, "the L2 predictor's trained frame rate")

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
            "Use a longer video or a later --start_frame."
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


def load_trajectory_points(trajectory_file):
    """Load a raw [x, y] trajectory (forward, lateral meters) from a .npy/.csv file, as
    a [T, 2] numpy array. T need not match any particular rate: the caller resamples
    (by arc length) to however many odometry steps the current rollout needs."""
    if not os.path.isfile(trajectory_file):
        raise FileNotFoundError(f"Trajectory file {trajectory_file} does not exist")

    if trajectory_file.endswith(".npy"):
        loaded = np.load(trajectory_file)
    elif trajectory_file.endswith(".csv"):
        loaded = np.loadtxt(trajectory_file, delimiter=",")
    else:
        raise ValueError("Trajectory file must end with .npy or .csv")

    if loaded.ndim != 2 or loaded.shape[1] != 2:
        raise ValueError(
            f"Trajectory file must contain [T, 2] (x, y) rows, got shape {tuple(loaded.shape)}"
        )
    if loaded.shape[0] < 2:
        raise ValueError(f"Trajectory file must contain at least 2 points, got {loaded.shape[0]}")

    return loaded


def resample_trajectory_by_arclength(points, n_samples):
    """Resample a [T, 2] path to `n_samples` points evenly spaced by arc length,
    preserving its shape while changing its temporal resolution to match however many
    odometry steps the current rollout configuration needs -- so the whole drawn/provided
    path is spread across the whole requested rollout duration, rather than being
    truncated (too many input points) or rejected outright (too few)."""
    pts = np.asarray(points, dtype=float)
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len == 0:
        return np.repeat(pts[:1], n_samples, axis=0)

    targets = np.linspace(0, total_len, n_samples)
    x = np.interp(targets, cum_len, pts[:, 0])
    y = np.interp(targets, cum_len, pts[:, 1])
    return np.stack([x, y], axis=1)


def trajectory_to_speed_yawrate(traj_xy, dt):
    """Differentiate an [T, 2] (x, y) trajectory (forward, lateral meters) into a
    [T - 1, 2] (speed, yaw_rate) array at the same odometry rate, matching the convention
    of `data.utils.get_trajectory_from_speeds_and_yaw_rates_batch`: `speed[i]` is the
    magnitude of segment i (point i -> point i+1), and `yaw_rate[i]` is the turn applied
    *after* segment i so that segment i+1 points in the right direction (yaw_rate[i] only
    affects `headings_for_translation[i+1]` onward, never segment i itself). The absolute
    orientation of the input file's coordinate frame does not matter: reconstruction always
    treats segment 0's direction as the local +x axis, exactly like `get_trajectory_from_speeds_and_yaw_rates`.
    """
    deltas = np.diff(traj_xy, axis=0)  # [T - 1, 2]
    speed = np.hypot(deltas[:, 0], deltas[:, 1]) / dt
    heading = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))  # [T - 1]

    yaw_rate = np.zeros_like(heading)
    yaw_rate[:-1] = np.diff(heading) / dt  # yaw_rate[-1] left at 0: no further segment to define it

    return np.stack([speed, yaw_rate], axis=1).astype(np.float32)


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


def maybe_apply_l2_nfe(model, l2_nfe):
    """Override the frozen L2 predictor's sampler steps (its `l2_pred_NFE`).

    L1 and L2 are sampled at different step counts: --l1_nfe is L1's NFE, while
    L2's comes from the config's `l2_pred_NFE`. Setting the attribute post-init is
    enough -- it is read at sample time, not at construction.
    """
    if l2_nfe is None:
        return

    condition_preprocessor = getattr(model, "condition_preprocessor", None)
    if condition_preprocessor is None or not hasattr(condition_preprocessor, "l2_pred_NFE"):
        raise TypeError(
            "--l2_nfe was given but the loaded condition_preprocessor has no "
            "`l2_pred_NFE` (so it has no separately-sampled L2 predictor)."
        )
    condition_preprocessor.l2_pred_NFE = int(l2_nfe)


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
    maybe_apply_l2_nfe(model, args.l2_nfe)

    height, width = resolve_context_image_size(config)
    backend = resolve_video_backend()
    l1_frame_rate = resolve_l1_frame_rate(config)
    l2_frame_rate = resolve_l2_frame_rate(model)

    l1_context_frames = int(model.vit.num_context_frames)
    l2_context_frames = int(model.condition_preprocessor.num_context_frames)

    l1_tensor, l2_tensor = load_l1_l2_context(
        video_path=args.video,
        start_frame=args.start_frame,
        l1_frame_rate=l1_frame_rate,
        l2_frame_rate=l2_frame_rate,
        l1_context_frames=l1_context_frames,
        l2_context_frames=l2_context_frames,
        height=height,
        width=width,
        backend=backend,
        device=args.device,
    )

    num_future_frames = get_rollout_future_frame_count(model, args.num_rollout_steps)
    frame_rate = torch.tensor(l1_frame_rate, device=args.device)
    data_batch = {"images": l1_tensor, "l2_context": l2_tensor, "frame_rate": frame_rate}

    if args.steering_file is not None and args.trajectory_file is not None:
        raise ValueError("--steering_file and --trajectory_file are mutually exclusive.")

    get_required_steps = getattr(model.condition_preprocessor, "get_required_rollout_odometry_steps", None)
    min_odo_steps = None
    if callable(get_required_steps):
        min_odo_steps = get_required_steps(
            validation_params=None,
            num_condition_frames=l1_context_frames,
            num_gen_frames=num_future_frames,
            rollout_steps=args.num_rollout_steps,
        )
    if args.steering_file is not None:
        data_batch["steering"] = load_steering_trajectory(
            args.steering_file, min_odo_steps, dtype=l1_tensor.dtype, device=args.device
        )
    elif args.trajectory_file is not None:
        if min_odo_steps is None:
            raise ValueError(
                "--trajectory_file requires the model's condition_preprocessor to report a "
                "required odometry length (get_required_rollout_odometry_steps returned None)."
            )
        odometry_steps_per_image_frame = getattr(model.condition_preprocessor, "odometry_steps_per_image_frame", 1)
        dt = 1.0 / (l1_frame_rate * odometry_steps_per_image_frame)
        traj_xy = load_trajectory_points(args.trajectory_file)
        traj_xy = resample_trajectory_by_arclength(traj_xy, min_odo_steps + 1)
        speed_yawrate = trajectory_to_speed_yawrate(traj_xy, dt)
        data_batch["steering"] = torch.as_tensor(
            speed_yawrate, dtype=l1_tensor.dtype, device=args.device
        ).unsqueeze(0)
    else:
        data_batch["steering"] = make_unconditional_steering(min_odo_steps, dtype=l1_tensor.dtype, device=args.device)
    data_batch["steering_format"] = STEERING_FORMAT

    condition_kwargs = model.condition_preprocessor.get_condition_kwargs_from_batch(data_batch, split="rollout")

    if args.steering_file is not None:
        steering_source = args.steering_file
    elif args.trajectory_file is not None:
        steering_source = f"trajectory({args.trajectory_file})"
    else:
        steering_source = "none"
    logger.info(f"Steering source: {steering_source}")
    logger.info(f"Steering scales: speed={args.speed_scale:g}, yaw_rate={args.yaw_rate_scale:g}")
    logger.info(f"L1/L2 frame rates: {l1_frame_rate:g}/{l2_frame_rate:g} Hz")
    logger.info(f"Saving generated images to {args.output_dir}")

    autocast_enabled = args.device.startswith("cuda")
    with torch.autocast(dtype=torch.float16, device_type="cuda", enabled=autocast_enabled):
        _latents, gen_frames = model.roll_out(
            x_0={"images": l1_tensor},
            num_gen_frames=args.num_rollout_steps,
            latent_input=False,
            NFE=args.l1_nfe,
            eta=0.0,
            sample_with_ema=args.evaluate_ema,
            num_samples=l1_tensor.size(0),
            frame_rate=frame_rate.reshape(1).repeat(l1_tensor.size(0)),
            condition_kwargs=condition_kwargs,
            decode_device=args.decode_device,
            num_condition_frames=l1_tensor.size(1),
        )

    if args.vis_mode in {"trajectory", "trajectory_ego"}:
        overlay_trajectory = model.condition_preprocessor.get_rollout_visualization_trajectory(
            condition_kwargs=model.condition_preprocessor.get_condition_kwargs_from_batch(data_batch, split="rollout"),
            num_condition_frames=l1_context_frames,
            num_gen_steps=args.num_rollout_steps,
            num_pred_frames=model.num_pred_frames,
        )
        if overlay_trajectory is not None:
            gen_frames = overlay_trajectory_on_images(gen_frames, overlay_trajectory, mode=args.vis_mode)

    # Release rollout-time latent state before CPU-side file I/O.
    del _latents, condition_kwargs, l1_tensor, l2_tensor

    # Save each rollout in the minibatch to its own sequence folder. The layout
    # (fake_images/sequence_XXXX/frame_XXXX.jpg) matches what the demo app reads.
    num_out = gen_frames.shape[0]
    num_frames = gen_frames.shape[1]
    for b in range(num_out):
        seq_dir = os.path.join(args.output_dir, "fake_images", f"sequence_{b:04d}")
        os.makedirs(seq_dir, exist_ok=True)
        for f in range(num_frames):
            save_image(
                (gen_frames[b, f] + 1.0) / 2.0,
                os.path.join(seq_dir, f"frame_{f:04d}.jpg"),
            )

        imageio.mimsave(
            os.path.join(args.output_dir, f"rollout_{b:04d}.gif"),
            [
                np.array(Image.open(os.path.join(seq_dir, f"frame_{f:04d}.jpg")))
                for f in range(num_frames)
            ],
            fps=l1_frame_rate,
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=os.environ.get("ORBIS2_MODELS_DIR"),
        help="Path to the experiment directory, where the config and checkpoints are stored. "
             "Defaults to the ORBIS2_MODELS_DIR environment variable.",
    )
    parser.add_argument("--ckpt", type=str, default="L1/checkpoints/last.ckpt", help="Path to the checkpoint file, relative to exp_dir")
    parser.add_argument("--config", type=str, default="L1/config.yaml", help="Path to the config file, relative to exp_dir")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file to sample L1/L2 context from.")
    parser.add_argument("--start_frame", type=int, default=None, help="Native-video frame index to start the L1 context window at. Defaults to the latest window that fits (the end of the video).")
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        default=10,
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
        "--trajectory_file",
        type=str,
        default=None,
        help="Optional .npy or .csv file of raw [x, y] trajectory points (forward, lateral meters, "
             "local ego frame), already at the expected odometry rate; converted to speed/yaw_rate via "
             "finite differences and used as steering input. Mutually exclusive with --steering_file.",
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
    parser.add_argument("--l1_nfe", type=int, default=30, help="Number of sampler steps (NFE) for the L1 detail predictor")
    parser.add_argument(
        "--l2_nfe",
        type=int,
        default=None,
        help="Number of sampler steps (NFE) for the frozen L2 abstract predictor. "
             "Overrides the config's `l2_pred_NFE`; defaults to whatever the config sets.",
    )
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

    if args.exp_dir is None:
        raise ValueError(
            "--exp_dir was not given and the ORBIS2_MODELS_DIR environment variable is not set."
        )

    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)
    if args.compile_artifacts:
        args.compile_artifacts = os.path.join(args.exp_dir, args.compile_artifacts)

    main(args, unknown)
