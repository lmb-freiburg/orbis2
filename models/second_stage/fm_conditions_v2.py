import logging
import os

import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
from einops import rearrange
from omegaconf import OmegaConf

from data.utils import get_trajectory_from_speeds_and_yaw_rates_batch
from util import instantiate_from_config


def _requires_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag


def _text_on_image(images, texts):
    for i in range(images.shape[0]):
        image = images[i]
        text = texts[i]
        image = PIL.Image.fromarray((((image + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        draw = PIL.ImageDraw.Draw(image)
        draw.text((10, 10), text, fill="red", font_size=15)
        image = (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0) * 2 - 1
        images[i] = image
    return images


class ConditionPreprocessor:
    """
    Default no-op condition preprocessor.

    Concrete implementations can extract batch conditions, normalize sampling inputs,
    and update autoregressive rollout state without changing predictor/objective/sampler
    control flow.
    """

    def __init__(self, predictor_module):
        self.module = predictor_module

    def get_condition_kwargs_from_batch(self, batch, split):
        return {}

    def prepare_condition_kwargs(self, condition_kwargs=None, batch_size=None, device=None, split=None):
        del batch_size, split
        if condition_kwargs is None:
            return {}

        prepared = {}
        for key, value in condition_kwargs.items():
            if torch.is_tensor(value) and device is not None:
                prepared[key] = value.to(device)
            else:
                prepared[key] = value
        return prepared

    def get_model_condition_kwargs(self, condition_kwargs):
        if not condition_kwargs:
            return {}
        return {
            key: value
            for key, value in condition_kwargs.items()
            if not key.startswith("_")
        }

    def slice_condition_kwargs(self, condition_kwargs, item):
        if not condition_kwargs:
            return {}

        sliced = {}
        for key, value in condition_kwargs.items():
            if torch.is_tensor(value):
                sliced[key] = value[item]
            else:
                sliced[key] = value
        return sliced

    def update_rollout_condition_kwargs(self, condition_kwargs, prediction, context, step_idx):
        del prediction, context, step_idx
        return condition_kwargs

    def get_rollout_visualization_state(self, condition_kwargs):
        del condition_kwargs
        return None

    def get_visualization_trajectory(self, visualization_state):
        del visualization_state
        return None

    def update_rollout_visualization_state(self, visualization_state, step_idx):
        del step_idx
        return visualization_state

    def get_rollout_visualization_trajectory(
        self,
        condition_kwargs,
        num_gen_steps,
        num_pred_frames=1,
        num_condition_frames=None,
    ):
        del condition_kwargs, num_gen_steps, num_pred_frames, num_condition_frames
        return None

    def annotate_logged_images(self, images, batch, num_images):
        del batch, num_images
        return images

    def to_device(self, device):
        pass

    def get_max_condition_odometry_offset(self):
        """
        Return the largest future raw-odometry offset needed to construct one
        conditioning tensor from the current anchor position.

        Preprocessors that do not depend on extra raw odometry should return
        ``None``.
        """
        return None

    def get_required_rollout_odometry_steps(
        self,
        *,
        validation_params,
        num_condition_frames,
        num_gen_frames,
        rollout_steps,
    ):
        del validation_params, num_condition_frames, num_gen_frames, rollout_steps
        return None


def _validate_speed_yaw_scale(value, name):
    scale = float(value)
    if not np.isfinite(scale):
        raise ValueError(f"`{name}` must be finite, got {value!r}.")
    return scale


def _apply_speed_yaw_scales(steering, speed_scale=1.0, yaw_rate_scale=1.0):
    if not torch.is_tensor(steering):
        raise TypeError(f"`steering` must be a tensor, got {type(steering).__name__}.")
    scaled = steering.clone()
    scaled[..., 0] = scaled[..., 0] * float(speed_scale)
    scaled[..., 1] = scaled[..., 1] * float(yaw_rate_scale)
    return scaled


class SpeedYawConditionPreprocessorBase(ConditionPreprocessor):
    """
    Shared steering-conditioning utilities for preprocessors that consume raw
    `[speed, yaw_rate]` sequences and advance through them during rollout.
    """

    def __init__(
        self,
        predictor_module,
        odometry_steps_per_image_frame,
        context_frame_anchor_index=-1,
        steering_format="speed_yawrate",
        speed_scale=1.0,
        yaw_rate_scale=1.0,
        steering_drop=0.0,
    ):
        super().__init__(predictor_module)
        self.steering_format = str(steering_format)
        if self.steering_format != "speed_yawrate":
            raise ValueError(
                f"{type(self).__name__} only supports `steering_format='speed_yawrate'`, "
                f"got {self.steering_format!r}."
            )

        self.odometry_steps_per_image_frame = int(odometry_steps_per_image_frame)
        if self.odometry_steps_per_image_frame <= 0:
            raise ValueError(
                "`odometry_steps_per_image_frame` must be a positive integer, "
                f"got {self.odometry_steps_per_image_frame}."
            )
        self.context_frame_anchor_index = int(context_frame_anchor_index)
        self.speed_scale = _validate_speed_yaw_scale(speed_scale, "speed_scale")
        self.yaw_rate_scale = _validate_speed_yaw_scale(yaw_rate_scale, "yaw_rate_scale")
        self.steering_drop = self._validate_steering_drop(steering_drop)

    def _validate_steering_drop(self, steering_drop):
        drop_prob = float(steering_drop)
        if not np.isfinite(drop_prob):
            raise ValueError(f"`steering_drop` must be finite, got {steering_drop!r}.")
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"`steering_drop` must be in [0, 1], got {drop_prob}.")
        return drop_prob

    def _validate_offsets(self, offsets, name):
        if not offsets:
            raise ValueError(f"`{name}` must contain at least one positive odometry offset.")
        normalized_offsets = [int(offset) for offset in offsets]
        if any(offset <= 0 for offset in normalized_offsets):
            raise ValueError(f"`{name}` must be positive integers, got {normalized_offsets}.")
        if normalized_offsets != sorted(normalized_offsets):
            raise ValueError(f"`{name}` must be sorted in ascending order, got {normalized_offsets}.")
        return normalized_offsets

    def get_max_condition_odometry_offset(self):
        offsets = None
        if hasattr(self, "goal_offsets"):
            offsets = self.goal_offsets
        elif hasattr(self, "steering_offsets"):
            offsets = self.steering_offsets
        if not offsets:
            return None
        return int(max(offsets))

    def get_required_rollout_odometry_steps(
        self,
        *,
        validation_params,
        num_condition_frames,
        num_gen_frames,
        rollout_steps,
    ):
        del validation_params, num_gen_frames
        max_condition_offset = self.get_max_condition_odometry_offset()
        if max_condition_offset is None:
            return None

        rollout_steps = int(rollout_steps)
        if rollout_steps <= 0:
            raise ValueError(f"`rollout_steps` must be positive, got {rollout_steps}.")

        anchor_frame_index = self._resolve_context_frame_anchor_index(int(num_condition_frames))
        initial_anchor_odo_index = anchor_frame_index * int(self.odometry_steps_per_image_frame)
        rollout_step_odo = int(self.odometry_steps_per_image_frame) * int(self.module.num_pred_frames)
        last_anchor_odo_index = initial_anchor_odo_index + (rollout_steps - 1) * rollout_step_odo
        return last_anchor_odo_index + int(max_condition_offset) + 1

    def _normalize_batch_steering_format(self, steering_format):
        if steering_format is None:
            return None
        if isinstance(steering_format, str):
            return steering_format
        if isinstance(steering_format, (list, tuple)):
            unique_formats = {str(item) for item in steering_format}
            if len(unique_formats) != 1:
                raise ValueError(f"Batch contains mixed steering formats: {sorted(unique_formats)}.")
            return next(iter(unique_formats))
        return str(steering_format)

    def _validate_batch_steering_format(self, batch):
        batch_format = self._normalize_batch_steering_format(batch.get("steering_format"))
        if batch_format is None:
            return
        if batch_format != self.steering_format:
            raise ValueError(
                f"{type(self).__name__} expected batch steering format "
                f"{self.steering_format!r}, got {batch_format!r}. Check the dataset "
                "`odo_transform_config` or `annotation_key`."
            )

    def _validate_steering(self, steering):
        """Validate and return raw steering with expected shape `[B, T, 2]`."""
        if steering is None:
            raise ValueError(f"`steering` is required for {type(self).__name__}.")
        if steering.ndim != 3:
            raise ValueError(
                f"`steering` must have shape [B, T, 2] for {type(self).__name__}, "
                f"got {tuple(steering.shape)}."
            )
        if steering.shape[2] != 2:
            raise ValueError(
                f"{type(self).__name__} expects raw steering features "
                f"[speed, yaw_rate], got shape {tuple(steering.shape)}."
            )
        return steering

    def _resolve_odometry_dt(self, batch, batch_size, device, required):
        frame_rate = batch.get("frame_rate")
        if frame_rate is None:
            if required:
                raise ValueError(
                    f"`frame_rate` is required to compute physical steering trajectories for "
                    f"{type(self).__name__}."
                )
            return None

        frame_rate = torch.as_tensor(frame_rate, device=device, dtype=torch.float32)
        if frame_rate.ndim == 0:
            frame_rate = frame_rate.expand(batch_size)
        elif frame_rate.ndim == 1 and frame_rate.shape[0] == 1 and batch_size != 1:
            frame_rate = frame_rate.expand(batch_size)
        elif frame_rate.ndim != 1 or frame_rate.shape[0] != batch_size:
            raise ValueError(
                "`frame_rate` must be broadcastable to shape [B], got "
                f"{tuple(frame_rate.shape)} for batch_size={batch_size}."
            )

        if torch.any(frame_rate <= 0):
            raise ValueError(f"`frame_rate` must be positive, got {frame_rate}.")

        return 1.0 / (frame_rate * float(self.odometry_steps_per_image_frame))

    def _get_context_num_frames(self, batch, split):
        """Return the number of conditioning image frames available for the current split."""
        if not isinstance(batch, dict) or "images" not in batch:
            raise ValueError(f"`images` are required in the batch to compute {type(self).__name__} conditions.")
        images = batch["images"]
        if images.ndim != 5:
            raise ValueError(f"`images` must have shape [B, F, C, H, W], got {tuple(images.shape)}.")

        if split == "rollout":
            return images.shape[1]
        if images.shape[1] < self.module.num_pred_frames:
            raise ValueError(
                f"Need at least {self.module.num_pred_frames} image frames, got {images.shape[1]}."
            )
        return images.shape[1] - self.module.num_pred_frames

    def _resolve_context_frame_anchor_index(self, context_num_frames):
        """Resolve the configured anchor index against the current context length."""
        anchor_index = self.context_frame_anchor_index
        if anchor_index < 0:
            anchor_index = context_num_frames + anchor_index
        if not 0 <= anchor_index < context_num_frames:
            raise IndexError(
                f"Resolved context_frame_anchor_index={anchor_index} is out of bounds for "
                f"context_num_frames={context_num_frames}."
            )
        return anchor_index

    def _build_rollout_state(self, steering, context_num_frames):
        steering = self._validate_steering(steering)
        batch_size = steering.shape[0]
        device = steering.device
        steps_per_image_frame = torch.full(
            (batch_size,),
            self.odometry_steps_per_image_frame,
            device=device,
            dtype=torch.long,
        )
        context_frame_anchor_index = self._resolve_context_frame_anchor_index(context_num_frames)
        anchor_odo_index = context_frame_anchor_index * steps_per_image_frame
        rollout_step_odo = steps_per_image_frame * self.module.num_pred_frames
        return {
            "_raw_speed_yaw": steering,
            "_anchor_odo_index": anchor_odo_index,
            "_rollout_step_odo": rollout_step_odo,
        }

    def _transform_raw_steering(self, steering):
        steering = self._validate_steering(steering)
        if self.speed_scale == 1.0 and self.yaw_rate_scale == 1.0:
            return steering
        return _apply_speed_yaw_scales(
            steering,
            speed_scale=self.speed_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )

    def _build_condition_kwargs(self, steering, context_num_frames, odometry_dt=None):
        condition_kwargs = self._build_rollout_state(steering, context_num_frames)
        if odometry_dt is not None:
            condition_kwargs["_odometry_dt"] = odometry_dt.to(
                device=steering.device,
                dtype=steering.dtype,
        )
        condition_kwargs["steering"] = self._compute_condition_steering(
            steering,
            condition_kwargs["_anchor_odo_index"],
            odometry_dt=condition_kwargs.get("_odometry_dt"),
        )
        return condition_kwargs

    def _maybe_apply_steering_dropout(self, steering, split):
        if split != "train" or self.steering_drop == 0.0:
            return steering

        drop_mask = torch.rand(steering.shape[0], device=steering.device) < self.steering_drop
        if not drop_mask.any():
            return steering

        dropped = steering.clone()
        dropped[drop_mask] = torch.nan
        return dropped

    def _compute_condition_steering(self, steering, anchor_odo_index, odometry_dt=None):
        raise NotImplementedError

    def _requires_odometry_dt_for_conditioning(self):
        return False

    def _make_no_steering_condition(self, batch_size, device):
        """Return a fully-NaN steering condition tensor for batches with no steering data.

        The NaN values cause LinearSteeringEmbedder to fall back to its no_value_embeddings,
        making missing-steering batches equivalent to fully-dropped steering (steering_drop=1.0).
        Returns None if the subclass does not support unconditional batches.
        """
        return None

    def get_condition_kwargs_from_batch(self, batch, split):
        if not isinstance(batch, dict) or "steering" not in batch:
            if isinstance(batch, dict) and "images" in batch:
                images = batch["images"]
                no_steering = self._make_no_steering_condition(images.shape[0], images.device)
                if no_steering is not None:
                    return {"steering": no_steering}
            return {}
        self._validate_batch_steering_format(batch)
        source_steering = self._validate_steering(batch["steering"])
        steering = self._transform_raw_steering(source_steering)
        context_num_frames = self._get_context_num_frames(batch, split)
        odometry_dt = self._resolve_odometry_dt(
            batch,
            batch_size=steering.shape[0],
            device=steering.device,
            required=self._requires_odometry_dt_for_conditioning(),
        )
        condition_kwargs = self._build_condition_kwargs(steering, context_num_frames, odometry_dt=odometry_dt)
        condition_kwargs["steering"] = self._maybe_apply_steering_dropout(condition_kwargs["steering"], split)
        condition_kwargs["_source_raw_speed_yaw"] = source_steering
        return condition_kwargs

    def update_rollout_condition_kwargs(self, condition_kwargs, prediction, context, step_idx):
        del prediction, context, step_idx
        if not condition_kwargs:
            return {}
        updated = dict(condition_kwargs)
        updated["_anchor_odo_index"] = condition_kwargs["_anchor_odo_index"] + condition_kwargs["_rollout_step_odo"]
        updated["steering"] = self._compute_condition_steering(
            condition_kwargs["_raw_speed_yaw"],
            updated["_anchor_odo_index"],
            odometry_dt=updated.get("_odometry_dt"),
        )
        return updated

    def get_rollout_visualization_state(self, condition_kwargs):
        if not condition_kwargs:
            return None
        return dict(condition_kwargs)

    def get_visualization_trajectory(self, visualization_state):
        if not visualization_state:
            return None
        return visualization_state.get("steering")

    def update_rollout_visualization_state(self, visualization_state, step_idx):
        if not visualization_state:
            return None
        return self.update_rollout_condition_kwargs(
            visualization_state,
            prediction=None,
            context=None,
            step_idx=step_idx,
        )

    def get_rollout_visualization_trajectory(
        self,
        condition_kwargs,
        num_gen_steps,
        num_pred_frames=1,
        num_condition_frames=None,
    ):
        """Return shared trajectory geometry plus explicit per-frame cursor indices."""
        if not condition_kwargs:
            return None

        odometry_dt = condition_kwargs.get("_odometry_dt")
        if odometry_dt is None:
            return None

        rollout_steps = max(1, int(num_gen_steps))
        rendered_frames_per_step = max(1, int(num_pred_frames))
        num_context_frames = max(0, int(num_condition_frames or 0))

        steering = self._validate_steering(condition_kwargs["_raw_speed_yaw"])
        anchor_odo_index = condition_kwargs["_anchor_odo_index"]
        rollout_step_odo = condition_kwargs["_rollout_step_odo"]
        trajectories = []

        for batch_idx in range(steering.shape[0]):
            anchor_idx = int(anchor_odo_index[batch_idx].item())
            step_odo = int(rollout_step_odo[batch_idx].item())
            end_idx = anchor_idx + rollout_steps * step_odo
            if end_idx >= steering.shape[1]:
                raise ValueError(
                    f"Need steering horizon of at least {end_idx + 1} odometry steps for visualization, "
                    f"got {steering.shape[1]}. Anchor={anchor_idx}, rollout_step_odo={step_odo}, "
                    f"num_gen_steps={num_gen_steps}, num_pred_frames={num_pred_frames}."
                )

            trajectory = get_trajectory_from_speeds_and_yaw_rates_batch(
                speeds=steering[batch_idx : batch_idx + 1, anchor_idx : end_idx + 1, 0],
                yaw_rates=steering[batch_idx : batch_idx + 1, anchor_idx : end_idx + 1, 1],
                dt=odometry_dt[batch_idx : batch_idx + 1],
            )
            trajectories.append(trajectory)

        trajectory = torch.cat(trajectories, dim=0).to(device=steering.device)
        step_indices = torch.arange(
            1,
            rollout_steps + 1,
            device=steering.device,
            dtype=torch.long,
        )
        cursor_index = (step_indices * rollout_step_odo[0]).repeat_interleave(rendered_frames_per_step)
        if num_context_frames > 0:
            context_cursor_index = torch.zeros(
                num_context_frames,
                device=steering.device,
                dtype=torch.long,
            )
            cursor_index = torch.cat([context_cursor_index, cursor_index], dim=0)
        cursor_index = cursor_index.unsqueeze(0).expand(steering.shape[0], -1)
        return {
            "trajectory": trajectory[:, :, :2],
            "heading": trajectory[:, :, 2],
            "cursor_index": cursor_index,
        }

    def annotate_logged_images(self, images, batch, num_images):
        if not isinstance(batch, dict) or "steering" not in batch:
            return images

        condition_batch = {
            "images": batch["images"][:num_images],
            "steering": batch["steering"][:num_images],
        }
        if "frame_rate" in batch:
            frame_rate = batch["frame_rate"]
            condition_batch["frame_rate"] = frame_rate[:num_images] if torch.is_tensor(frame_rate) else frame_rate
        if "steering_format" in batch:
            condition_batch["steering_format"] = batch["steering_format"]

        condition_kwargs = self.get_condition_kwargs_from_batch(
            condition_batch,
            split="log_images",
        )
        steering = condition_kwargs["steering"]
        steering_strings = []
        for sample in steering.cpu().numpy():
            step_strings = ["[" + ", ".join(f"{value:.1f}" for value in step) + "]" for step in sample]
            steering_strings.append(" ".join(step_strings))
        images[:, -1] = _text_on_image(images[:, -1], steering_strings)
        return images


class SpeedYawMovingGoalConditionPreprocessor(SpeedYawConditionPreprocessorBase):
    """
    Compute future anchored goal displacements from raw speed/yaw-rate steering using an
    explicit anchor definition.

    Assumptions:
    - `batch["steering"]` has shape [B, T, 2] with features [speed, yaw_rate]
    - odometry/image frame-rate ratio is fixed and integer
    - the anchor is a configured image-frame index within the context window
    """

    def __init__(
        self,
        predictor_module,
        goal_offsets,
        odometry_steps_per_image_frame,
        context_frame_anchor_index=-1,
        steering_format="speed_yawrate",
        speed_scale=1.0,
        yaw_rate_scale=1.0,
        **kwargs,
    ):
        """Configure moving-goal conditioning from raw `[speed, yaw_rate]` sequences."""
        super().__init__(
            predictor_module=predictor_module,
            odometry_steps_per_image_frame=odometry_steps_per_image_frame,
            context_frame_anchor_index=context_frame_anchor_index,
            steering_format=steering_format,
            speed_scale=speed_scale,
            yaw_rate_scale=yaw_rate_scale,
            **kwargs
        )
        self.goal_offsets = self._validate_offsets(goal_offsets, "goal_offsets")

    def _make_no_steering_condition(self, batch_size, device):
        return torch.full((batch_size, len(self.goal_offsets), 2), float("nan"), device=device)

    def _requires_odometry_dt_for_conditioning(self):
        return True

    def _compute_condition_steering(self, steering, anchor_odo_index, odometry_dt=None):
        """Convert raw speed/yaw windows into future goal points anchored at odometry indices."""
        steering = self._validate_steering(steering)
        batch_size, steering_len, _dims = steering.shape
        device = steering.device
        if odometry_dt is None:
            raise ValueError(
                "SpeedYawMovingGoalConditionPreprocessor requires `odometry_dt` to compute goal-point "
                "conditioning."
            )
        odometry_dt = torch.as_tensor(odometry_dt, device=device, dtype=steering.dtype)
        if odometry_dt.ndim == 0:
            odometry_dt = odometry_dt.expand(batch_size)
        elif odometry_dt.ndim != 1 or odometry_dt.shape[0] != batch_size:
            raise ValueError(
                "`odometry_dt` must be broadcastable to shape [B], got "
                f"{tuple(odometry_dt.shape)} for batch_size={batch_size}."
            )
        max_offset = self.goal_offsets[-1]
        goals = []

        for batch_idx in range(batch_size):
            current_anchor = int(anchor_odo_index[batch_idx].item())
            goal_end_idx = current_anchor + max_offset
            if current_anchor < 0:
                raise ValueError(f"Anchor index must be non-negative, got {current_anchor}.")
            if goal_end_idx >= steering_len:
                raise ValueError(
                    f"Need steering horizon of at least {goal_end_idx + 1} odometry steps, "
                    f"got {steering_len}. Anchor={current_anchor}, max_offset={max_offset}."
                )

            centered_traj = get_trajectory_from_speeds_and_yaw_rates_batch(
                speeds=steering[batch_idx : batch_idx + 1, current_anchor : goal_end_idx + 1, 0],
                yaw_rates=steering[batch_idx : batch_idx + 1, current_anchor : goal_end_idx + 1, 1],
                dt=odometry_dt[batch_idx : batch_idx + 1],
            )
            goals.append(centered_traj[:, self.goal_offsets, :2])

        return torch.cat(goals, dim=0).to(device=device)


class SpeedYawDirectConditionPreprocessor(SpeedYawConditionPreprocessorBase):
    """
    Slice future raw speed/yaw-rate commands directly from the odometry sequence.

    This shares the same rollout alignment logic as moving-goal conditioning, but
    returns raw `[speed, yaw_rate]` values instead of integrating them into XY goals.
    """

    def __init__(
        self,
        predictor_module,
        steering_offsets,
        odometry_steps_per_image_frame,
        context_frame_anchor_index=-1,
        steering_format="speed_yawrate",
        speed_scale=1.0,
        yaw_rate_scale=1.0,
    ):
        super().__init__(
            predictor_module=predictor_module,
            odometry_steps_per_image_frame=odometry_steps_per_image_frame,
            context_frame_anchor_index=context_frame_anchor_index,
            steering_format=steering_format,
            speed_scale=speed_scale,
            yaw_rate_scale=yaw_rate_scale,
        )
        self.steering_offsets = self._validate_offsets(steering_offsets, "steering_offsets")

    def _make_no_steering_condition(self, batch_size, device):
        return torch.full((batch_size, len(self.steering_offsets), 2), float("nan"), device=device)

    def _compute_condition_steering(self, steering, anchor_odo_index, odometry_dt=None):
        del odometry_dt
        steering = self._validate_steering(steering)
        batch_size, steering_len, _dims = steering.shape
        commands = []
        max_offset = self.steering_offsets[-1]
        steering_offsets = torch.as_tensor(
            self.steering_offsets,
            device=steering.device,
            dtype=torch.long,
        )

        for batch_idx in range(batch_size):
            current_anchor = int(anchor_odo_index[batch_idx].item())
            end_idx = current_anchor + max_offset
            if current_anchor < 0:
                raise ValueError(f"Anchor index must be non-negative, got {current_anchor}.")
            if end_idx >= steering_len:
                raise ValueError(
                    f"Need steering horizon of at least {end_idx + 1} odometry steps, "
                    f"got {steering_len}. Anchor={current_anchor}, max_offset={max_offset}."
                )
            commands.append(steering[batch_idx : batch_idx + 1, current_anchor + steering_offsets])

        return torch.cat(commands, dim=0).to(device=steering.device)


class L2EndpointConditionPreprocessor(ConditionPreprocessor):
    """
    Inference-only L2 endpoint conditioning for the CleanCtx PredL2 v1 model.

    The preprocessor owns the frozen L2 predictor and supplies only
    `z_l2_start` / `z_l2_end` to the DiT. Private keys are retained as rollout
    state and filtered before model calls by ConditionPreprocessor.
    """

    def __init__(
        self,
        predictor_module,
        l2_predictor_config,
        l2_predictor_frame_rate=1.0,
        l2_context_latent=False,
        l2_pred_NFE=10,
        num_context_frames=1,
        use_z_l2_start=True,
    ):
        super().__init__(predictor_module)
        self.l2_predictor_frame_rate = float(l2_predictor_frame_rate)
        self.l2_context_latent = bool(l2_context_latent)
        self.l2_pred_NFE = int(l2_pred_NFE)
        self.num_context_frames = int(num_context_frames)
        self.use_z_l2_start = bool(use_z_l2_start)
        self.l2_predictor_config = l2_predictor_config
        self.l2_predictor = self._build_l2_predictor(l2_predictor_config)
        self.l2_predictor_encoder_branch = self._get_l2_predictor_encoder_branch(
            predictor=self.l2_predictor,
            predictor_config=l2_predictor_config,
        )

    # --- BEGIN: rollout debug path info (safe to remove) ---
    def get_l2_predictor_path_info(self):
        """Return (exp_folder_name, ckpt_name) describing where the L2 predictor was loaded from."""
        folder = os.path.expandvars(self.l2_predictor_config.folder)
        ckpt_path = self.l2_predictor_config.ckpt_path if self.l2_predictor_config.ckpt_path else "checkpoints/last.ckpt"
        return os.path.basename(os.path.normpath(folder)), os.path.basename(ckpt_path)
    # --- END: rollout debug path info (safe to remove) ---

    @property
    def ae(self):
        return self.module.ae

    @property
    def enc_scale(self):
        return self.module.enc_scale

    @property
    def enc_scale_dino(self):
        return self.module.enc_scale_dino

    def _build_l2_predictor(self, cfg):
        folder = os.path.expandvars(cfg.folder)
        ckpt_path = cfg.ckpt_path if cfg.ckpt_path else "checkpoints/last.ckpt"
        logging.info(f"Loading L2 checkpoint from {os.path.join(folder, ckpt_path)}")
        model_cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
        predictor = instantiate_from_config(model_cfg.model)
        state_dict = torch.load(
            os.path.join(folder, ckpt_path),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]
        predictor.load_state_dict(state_dict, strict=False)
        predictor.eval()
        _requires_grad(predictor, False)
        return predictor

    def _normalize_l2_predictor_encoder_branch(self, encoder_branch):
        if isinstance(encoder_branch, int):
            if encoder_branch == 0:
                return "rec"
            if encoder_branch == 1:
                return "sem"

        branch = str(encoder_branch).strip().lower()
        aliases = {
            "rec": "rec",
            "x0": "rec",
            "h": "rec",
            "sem": "sem",
            "x1": "sem",
            "h2": "sem",
        }
        if branch not in aliases:
            raise ValueError(
                f"Unsupported delegated L2 encoder_branch={encoder_branch!r}. "
                "Expected one of: rec, x0, h, sem, x1, h2."
            )
        return aliases[branch]

    def _get_l2_predictor_encoder_branch(self, predictor, predictor_config):
        first_stage = getattr(predictor, "first_stage", None)
        if first_stage is not None and hasattr(first_stage, "encoder_branch"):
            return self._normalize_l2_predictor_encoder_branch(first_stage.encoder_branch)

        if hasattr(predictor, "encoder_branch"):
            return self._normalize_l2_predictor_encoder_branch(predictor.encoder_branch)

        folder = os.path.expandvars(predictor_config.folder)
        model_cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
        first_stage_cfg = model_cfg.model.params.get("first_stage_handler_config")
        if first_stage_cfg and "params" in first_stage_cfg and "encoder_branch" in first_stage_cfg.params:
            return self._normalize_l2_predictor_encoder_branch(first_stage_cfg.params.encoder_branch)

        if "encoder_branch" in model_cfg.model.params:
            return self._normalize_l2_predictor_encoder_branch(model_cfg.model.params.encoder_branch)

        raise ValueError(
            "Could not determine the delegated L2 predictor encoder branch from the loaded model "
            "or its config. Expected a rec/sem branch selection."
        )

    def to_device(self, device):
        self.l2_predictor.to(device)

    def get_l2_predictor_encoder_branch(self):
        return self.l2_predictor_encoder_branch

    def get_l2_predictor_latent_scale(self):
        if self.l2_predictor_encoder_branch == "rec":
            return self.enc_scale
        return self.enc_scale_dino

    def _reject_training_split(self, split):
        if split in {"train", "val"}:
            raise RuntimeError(
                "L2EndpointConditionPreprocessor is inference-only. "
                "It supports sample/log_images/rollout conditioning, not train/val."
            )

    def _validate_guidance_scale(self, condition_kwargs):
        if not condition_kwargs or "l2_guidance_scale" not in condition_kwargs:
            return
        guidance_scale = condition_kwargs["l2_guidance_scale"]
        if torch.is_tensor(guidance_scale):
            guidance_scale = float(guidance_scale.detach().cpu().item())
        else:
            guidance_scale = float(guidance_scale)
        if abs(guidance_scale - 1.0) > 1e-6:
            raise RuntimeError(
                "L2EndpointConditionPreprocessor only supports "
                "l2_guidance_scale == 1.0."
            )

    @torch.no_grad()
    def _encode_l2_predictor_frames(self, images):
        if images.ndim == 5:
            b, f, _c, _h, _w = images.size()
            images = rearrange(images, "b f c h w -> (b f) c h w")
        else:
            b, _c, _h, _w = images.size()
            f = 1

        continuous = self.ae.encode(images)["continuous"]
        if not isinstance(continuous, tuple) or len(continuous) != 2:
            raise ValueError(
                "L2EndpointConditionPreprocessor expects tokenizer.encode(...)[\"continuous\"] "
                f"to return a 2-tuple, got {type(continuous)}."
            )
        branch_index = 0 if self.l2_predictor_encoder_branch == "rec" else 1
        latent_scale = self.get_l2_predictor_latent_scale()
        latents = continuous[branch_index] * latent_scale
        return rearrange(latents, "(b f) c h w -> b f c h w", b=b, f=f)

    @torch.no_grad()
    def _encode_l2_context(self, l2_context):
        if self.l2_context_latent:
            return l2_context * self.get_l2_predictor_latent_scale()
        return self._encode_l2_predictor_frames(l2_context)

    @torch.no_grad()
    def _predict_l2_end(self, l2_context_latent):
        batch_size = l2_context_latent.shape[0]
        device = l2_context_latent.device
        if next(self.l2_predictor.parameters()).device != device:
            self.l2_predictor.to(device)
        predictor = self.l2_predictor
        frame_rate = torch.full(
            (batch_size,),
            self.l2_predictor_frame_rate,
            device=device,
        )
        pred = predictor.sample(
            images=l2_context_latent,
            latent=True,
            eta=0.0,
            NFE=self.l2_pred_NFE,
            sample_with_ema=True,
            num_samples=batch_size,
            frame_rate=frame_rate,
        )
        return pred[:, -1]

    @torch.no_grad()
    def _sample_l2_next_latent(self, l2_context_latent):
        return self._predict_l2_end(l2_context_latent)

    def _get_z_l2_start_from_context(self, l2_context_latent, z_l2_end):
        if self.use_z_l2_start:
            if l2_context_latent is None:
                raise ValueError(
                    "L2EndpointConditionPreprocessor requires z_l2_start or "
                    "l2_context/_l2_context_latent when use_z_l2_start=True."
                )
            return l2_context_latent[:, -1]
        return torch.zeros_like(z_l2_end)

    def get_condition_kwargs_from_batch(self, batch, split):
        self._reject_training_split(split)
        if not isinstance(batch, dict) or "l2_context" not in batch:
            raise ValueError(
                "L2EndpointConditionPreprocessor requires batch['l2_context'] "
                f"for split={split!r}."
            )
        return {"l2_context": batch["l2_context"]}

    def _move_public_condition_kwargs(self, condition_kwargs, device):
        prepared = {}
        for key, value in condition_kwargs.items():
            if key == "l2_guidance_scale":
                continue
            if torch.is_tensor(value) and device is not None:
                prepared[key] = value.to(device)
            else:
                prepared[key] = value
        return prepared

    @torch.no_grad()
    def prepare_condition_kwargs(self, condition_kwargs=None, batch_size=None, device=None, split=None):
        self._reject_training_split(split)
        condition_kwargs = condition_kwargs or {}
        self._validate_guidance_scale(condition_kwargs)
        prepared = self._move_public_condition_kwargs(condition_kwargs, device)

        has_start = "z_l2_start" in prepared
        has_end = "z_l2_end" in prepared
        prepared["_z_l2_start_fixed"] = bool(prepared.get("_z_l2_start_fixed", has_start))
        prepared["_z_l2_end_fixed"] = bool(prepared.get("_z_l2_end_fixed", has_end))

        l2_context_latent = prepared.get("_l2_context_latent")
        if l2_context_latent is None and "l2_context" in prepared:
            l2_context_latent = self._encode_l2_context(prepared.pop("l2_context"))
            prepared["_l2_context_latent"] = l2_context_latent
        elif l2_context_latent is not None:
            prepared["_l2_context_latent"] = l2_context_latent

        if "z_l2_end" not in prepared:
            if l2_context_latent is None:
                raise ValueError(
                    "L2EndpointConditionPreprocessor requires z_l2_end or "
                    "l2_context/_l2_context_latent to predict it."
                )
            prepared["_l2_next_latent"] = self._sample_l2_next_latent(l2_context_latent)
            prepared["z_l2_end"] = prepared["_l2_next_latent"]
        elif "_l2_next_latent" not in prepared and not prepared["_z_l2_end_fixed"]:
            prepared["_l2_next_latent"] = prepared["z_l2_end"]

        if "z_l2_start" not in prepared:
            prepared["z_l2_start"] = self._get_z_l2_start_from_context(
                l2_context_latent,
                prepared["z_l2_end"],
            )

        if batch_size is not None:
            for key in ("z_l2_start", "z_l2_end"):
                if prepared[key].shape[0] != batch_size:
                    raise ValueError(
                        f"{key} batch size {prepared[key].shape[0]} does not match "
                        f"expected batch_size={batch_size}."
                    )

        return prepared

    @torch.no_grad()
    def update_rollout_condition_kwargs(self, condition_kwargs, prediction, context, step_idx):
        del prediction, context, step_idx
        if not condition_kwargs:
            return {}

        updated = dict(condition_kwargs)
        l2_context_latent = updated.get("_l2_context_latent")
        start_fixed = bool(updated.get("_z_l2_start_fixed", False))
        end_fixed = bool(updated.get("_z_l2_end_fixed", False))
        if l2_context_latent is None or (start_fixed and end_fixed):
            return updated

        l2_next_latent = updated.get("_l2_next_latent")
        if l2_next_latent is None:
            if end_fixed:
                l2_next_latent = updated["z_l2_end"]
            else:
                raise ValueError(
                    "L2EndpointConditionPreprocessor requires cached `_l2_next_latent` "
                    "or explicit `z_l2_end` during rollout updates."
                )
        l2_next_latent = l2_next_latent.to(l2_context_latent.device)
        updated["_l2_context_latent"] = torch.cat(
            [l2_context_latent[:, 1:], l2_next_latent.unsqueeze(1)],
            dim=1,
        )
        if not start_fixed:
            updated["z_l2_start"] = self._get_z_l2_start_from_context(
                updated["_l2_context_latent"],
                updated["z_l2_end"],
            )
        if not end_fixed:
            updated["_l2_next_latent"] = self._sample_l2_next_latent(updated["_l2_context_latent"])
            updated["z_l2_end"] = updated["_l2_next_latent"]
        return updated


class L2SteeringEndpointConditionPreprocessor(L2EndpointConditionPreprocessor):
    """
    Steering-aware variant of L2EndpointConditionPreprocessor.

    Raw speed/yaw-rate steering is converted into the conditioning expected by
    the frozen L2 predictor. That steering state remains private to the L2
    branch and is updated in lockstep with the autoregressive L2-context slide.
    """

    def __init__(
        self,
        predictor_module,
        l2_predictor_config,
        l2_predictor_frame_rate=1.0,
        l2_context_latent=False,
        l2_pred_NFE=10,
        num_context_frames=1,
        use_z_l2_start=True,
        speed_scale=1.0,
        yaw_rate_scale=1.0,
    ):
        super().__init__(
            predictor_module=predictor_module,
            l2_predictor_config=l2_predictor_config,
            l2_predictor_frame_rate=l2_predictor_frame_rate,
            l2_context_latent=l2_context_latent,
            l2_pred_NFE=l2_pred_NFE,
            num_context_frames=num_context_frames,
            use_z_l2_start=use_z_l2_start,
        )
        self.speed_scale = _validate_speed_yaw_scale(speed_scale, "speed_scale")
        self.yaw_rate_scale = _validate_speed_yaw_scale(yaw_rate_scale, "yaw_rate_scale")
        self.l2_condition_preprocessor = self._get_l2_condition_preprocessor()
        if hasattr(self.l2_condition_preprocessor, "goal_offsets"):
            self.goal_offsets = [int(offset) for offset in self.l2_condition_preprocessor.goal_offsets]
        if hasattr(self.l2_condition_preprocessor, "steering_offsets"):
            self.steering_offsets = [int(offset) for offset in self.l2_condition_preprocessor.steering_offsets]
        self._validate_l2_condition_preprocessor()

    def _get_l2_condition_preprocessor(self):
        preprocessor = getattr(self.l2_predictor, "condition_preprocessor", None)
        if preprocessor is None:
            raise TypeError(
                "L2SteeringEndpointConditionPreprocessor requires the frozen L2 predictor "
                "to expose `condition_preprocessor`."
            )
        return preprocessor

    def _validate_l2_condition_preprocessor(self):
        required_methods = (
            "get_condition_kwargs_from_batch",
            "prepare_condition_kwargs",
            "update_rollout_condition_kwargs",
        )
        missing = [
            method_name
            for method_name in required_methods
            if not callable(getattr(self.l2_condition_preprocessor, method_name, None))
        ]
        if missing:
            raise TypeError(
                "Frozen L2 condition preprocessor is incompatible with steering delegation; "
                f"missing methods: {missing}."
            )

    def get_max_condition_odometry_offset(self):
        if hasattr(self.l2_condition_preprocessor, "get_max_condition_odometry_offset"):
            return self.l2_condition_preprocessor.get_max_condition_odometry_offset()
        return None

    def get_required_rollout_odometry_steps(
        self,
        *,
        validation_params,
        num_condition_frames,
        num_gen_frames,
        rollout_steps,
    ):
        del num_condition_frames
        delegated_required_steps = getattr(self.l2_condition_preprocessor, "get_required_rollout_odometry_steps", None)
        if callable(delegated_required_steps):
            delegated_num_pred_frames = int(getattr(self.l2_predictor, "num_pred_frames", 1))
            return delegated_required_steps(
                validation_params=validation_params,
                num_condition_frames=int(self.num_context_frames),
                num_gen_frames=int(rollout_steps) * delegated_num_pred_frames,
                rollout_steps=rollout_steps,
            )

        del num_gen_frames
        max_condition_offset = self.get_max_condition_odometry_offset()
        if max_condition_offset is None:
            return None

        source_odometry_rate = float(
            getattr(validation_params, "odometry_frame_rate", getattr(validation_params, "frame_rate"))
        )
        target_odometry_rate = float(self.l2_predictor_frame_rate)
        ratio = source_odometry_rate / target_odometry_rate
        rounded_ratio = round(ratio)
        if abs(ratio - rounded_ratio) > 1e-8:
            raise ValueError(
                "Delegated L2 rollout requires the parent odometry rate to be an integer multiple of "
                f"the frozen L2 rate, got odometry_frame_rate={source_odometry_rate} and "
                f"l2_predictor_frame_rate={target_odometry_rate}."
            )

        required_l2_steps = self.num_context_frames + int(rollout_steps) + int(max_condition_offset) - 1
        return int((required_l2_steps - 1) * rounded_ratio + 1)

    def _maybe_resample_raw_steering_for_l2(self, steering, batch):
        if not torch.is_tensor(steering):
            raise TypeError(f"`steering` must be a tensor, got {type(steering).__name__}.")
        if "frame_rate" not in batch:
            return steering

        source_frame_rate = torch.as_tensor(
            batch["frame_rate"],
            device=steering.device,
            dtype=torch.float32,
        )
        if source_frame_rate.ndim == 0:
            source_frame_rate = source_frame_rate.expand(steering.shape[0])
        elif source_frame_rate.ndim == 1 and source_frame_rate.shape[0] == 1 and steering.shape[0] != 1:
            source_frame_rate = source_frame_rate.expand(steering.shape[0])
        elif source_frame_rate.ndim != 1 or source_frame_rate.shape[0] != steering.shape[0]:
            raise ValueError(
                "`frame_rate` must be broadcastable to shape [B] for delegated L2 steering, got "
                f"{tuple(source_frame_rate.shape)} for batch_size={steering.shape[0]}."
            )

        target_frame_rate = torch.full_like(source_frame_rate, float(self.l2_predictor_frame_rate))
        if torch.any(target_frame_rate <= 0):
            raise ValueError(f"`l2_predictor_frame_rate` must be positive, got {self.l2_predictor_frame_rate}.")

        if torch.allclose(source_frame_rate, target_frame_rate):
            return steering

        ratios = source_frame_rate / target_frame_rate
        rounded_ratios = torch.round(ratios)
        if not torch.allclose(ratios, rounded_ratios, atol=1e-8, rtol=0.0):
            raise ValueError(
                "Delegated L2 steering requires the parent frame rate to be an integer multiple of "
                f"the frozen L2 frame rate. Got source_frame_rate={source_frame_rate.tolist()} and "
                f"l2_predictor_frame_rate={self.l2_predictor_frame_rate}."
            )

        resampled = []
        for batch_idx in range(steering.shape[0]):
            step = int(rounded_ratios[batch_idx].item())
            if step <= 0:
                raise ValueError(
                    f"Delegated L2 steering subsample step must be positive, got {step}."
                )
            resampled.append(steering[batch_idx : batch_idx + 1, ::step])
        return torch.cat(resampled, dim=0)

    def _build_l2_rollout_batch(self, batch):
        if "l2_context" not in batch:
            raise ValueError(
                "L2SteeringEndpointConditionPreprocessor requires batch['l2_context'] "
                "to build delegated L2 conditions."
            )
        if "steering" not in batch:
            raise ValueError(
                "L2SteeringEndpointConditionPreprocessor requires batch['steering'] "
                "to build delegated L2 conditions."
            )
        source_steering = batch["steering"]
        scaled_steering = _apply_speed_yaw_scales(
            source_steering,
            speed_scale=self.speed_scale,
            yaw_rate_scale=self.yaw_rate_scale,
        )
        l2_batch = {
            "images": batch["l2_context"],
            # Keep steering at raw odometry resolution and let the frozen L2
            # preprocessor interpret it via its own odometry alignment config.
            "steering": scaled_steering,
        }
        if "steering_format" in batch:
            l2_batch["steering_format"] = batch["steering_format"]
        if "frame_rate" in batch:
            frame_rate = torch.as_tensor(
                batch["frame_rate"],
                device=batch["steering"].device,
                dtype=torch.float32,
            )
            l2_batch["frame_rate"] = torch.full_like(
                frame_rate,
                float(self.l2_predictor_frame_rate),
                dtype=torch.float32,
            )
        else:
            batch_size = batch["steering"].shape[0]
            l2_batch["frame_rate"] = torch.full(
                (batch_size,),
                float(self.l2_predictor_frame_rate),
                device=batch["steering"].device,
                dtype=torch.float32,
            )
        return l2_batch

    def _prepare_l2_predictor_condition_kwargs(self, l2_condition_kwargs, device):
        return self.l2_condition_preprocessor.prepare_condition_kwargs(
            l2_condition_kwargs,
            batch_size=None,
            device=device,
            split="rollout",
        )

    def _update_l2_predictor_condition_kwargs(self, l2_condition_kwargs):
        return self.l2_condition_preprocessor.update_rollout_condition_kwargs(
            l2_condition_kwargs,
            prediction=None,
            context=None,
            step_idx=0,
        )

    @torch.no_grad()
    def _predict_l2_end(self, l2_context_latent, l2_condition_kwargs=None):
        batch_size = l2_context_latent.shape[0]
        device = l2_context_latent.device
        if next(self.l2_predictor.parameters()).device != device:
            self.l2_predictor.to(device)
        predictor = self.l2_predictor
        frame_rate = torch.full(
            (batch_size,),
            self.l2_predictor_frame_rate,
            device=device,
        )
        pred = predictor.sample(
            images=l2_context_latent,
            latent=True,
            eta=0.0,
            NFE=self.l2_pred_NFE,
            sample_with_ema=True,
            num_samples=batch_size,
            frame_rate=frame_rate,
            condition_kwargs=l2_condition_kwargs,
        )
        return pred[:, -1]

    @torch.no_grad()
    def _sample_l2_next_latent(self, l2_context_latent, l2_condition_kwargs=None):
        return self._predict_l2_end(l2_context_latent, l2_condition_kwargs=l2_condition_kwargs)

    def get_condition_kwargs_from_batch(self, batch, split):
        self._reject_training_split(split)
        if not isinstance(batch, dict) or "l2_context" not in batch:
            raise ValueError(
                "L2SteeringEndpointConditionPreprocessor requires batch['l2_context'] "
                f"for split={split!r}."
            )
        if "steering" not in batch:
            raise ValueError(
                "L2SteeringEndpointConditionPreprocessor requires batch['steering'] "
                f"for split={split!r}."
            )
        l2_context = batch["l2_context"]
        if l2_context.ndim != 5:
            raise ValueError(f"`l2_context` must have shape [B, F, C, H, W], got {tuple(l2_context.shape)}.")
        source_l2_steering = self._maybe_resample_raw_steering_for_l2(batch["steering"], batch)
        l2_condition_kwargs = self.l2_condition_preprocessor.get_condition_kwargs_from_batch(
            self._build_l2_rollout_batch(batch),
            split="rollout",
        )
        l2_condition_kwargs["_source_raw_speed_yaw"] = source_l2_steering
        return {
            "l2_context": l2_context,
            "_l2_condition_kwargs": l2_condition_kwargs,
        }

    @torch.no_grad()
    def prepare_condition_kwargs(self, condition_kwargs=None, batch_size=None, device=None, split=None):
        self._reject_training_split(split)
        condition_kwargs = condition_kwargs or {}
        self._validate_guidance_scale(condition_kwargs)
        prepared = self._move_public_condition_kwargs(condition_kwargs, device)

        has_start = "z_l2_start" in prepared
        has_end = "z_l2_end" in prepared
        prepared["_z_l2_start_fixed"] = bool(prepared.get("_z_l2_start_fixed", has_start))
        prepared["_z_l2_end_fixed"] = bool(prepared.get("_z_l2_end_fixed", has_end))

        l2_context_latent = prepared.get("_l2_context_latent")
        if l2_context_latent is None and "l2_context" in prepared:
            l2_context_latent = self._encode_l2_context(prepared.pop("l2_context"))
            prepared["_l2_context_latent"] = l2_context_latent
        elif l2_context_latent is not None:
            prepared["_l2_context_latent"] = l2_context_latent

        l2_condition_kwargs = self._prepare_l2_predictor_condition_kwargs(
            prepared.get("_l2_condition_kwargs"),
            device=device,
        )
        prepared["_l2_condition_kwargs"] = l2_condition_kwargs

        if "z_l2_end" not in prepared:
            if l2_context_latent is None:
                raise ValueError(
                    "L2SteeringEndpointConditionPreprocessor requires z_l2_end or "
                    "l2_context/_l2_context_latent to predict it."
                )
            prepared["_l2_next_latent"] = self._sample_l2_next_latent(
                l2_context_latent,
                l2_condition_kwargs=l2_condition_kwargs,
            )
            prepared["z_l2_end"] = prepared["_l2_next_latent"]
        elif "_l2_next_latent" not in prepared and not prepared["_z_l2_end_fixed"]:
            prepared["_l2_next_latent"] = prepared["z_l2_end"]

        if "z_l2_start" not in prepared:
            prepared["z_l2_start"] = self._get_z_l2_start_from_context(
                l2_context_latent,
                prepared["z_l2_end"],
            )

        if batch_size is not None:
            for key in ("z_l2_start", "z_l2_end"):
                if prepared[key].shape[0] != batch_size:
                    raise ValueError(
                        f"{key} batch size {prepared[key].shape[0]} does not match "
                        f"expected batch_size={batch_size}."
                    )

        return prepared

    @torch.no_grad()
    def update_rollout_condition_kwargs(self, condition_kwargs, prediction, context, step_idx):
        del prediction, context, step_idx
        if not condition_kwargs:
            return {}

        updated = dict(condition_kwargs)
        l2_context_latent = updated.get("_l2_context_latent")
        start_fixed = bool(updated.get("_z_l2_start_fixed", False))
        end_fixed = bool(updated.get("_z_l2_end_fixed", False))
        if l2_context_latent is None:
            return updated

        if not (start_fixed and end_fixed):
            l2_next_latent = updated.get("_l2_next_latent")
            if l2_next_latent is None:
                if end_fixed:
                    l2_next_latent = updated["z_l2_end"]
                else:
                    raise ValueError(
                        "L2SteeringEndpointConditionPreprocessor requires cached `_l2_next_latent` "
                        "or explicit `z_l2_end` during rollout updates."
                    )
            l2_next_latent = l2_next_latent.to(l2_context_latent.device)
            updated["_l2_context_latent"] = torch.cat(
                [l2_context_latent[:, 1:], l2_next_latent.unsqueeze(1)],
                dim=1,
            )

        if "_l2_condition_kwargs" in updated:
            updated["_l2_condition_kwargs"] = self._update_l2_predictor_condition_kwargs(
                updated["_l2_condition_kwargs"],
            )

        if not start_fixed:
            updated["z_l2_start"] = self._get_z_l2_start_from_context(
                updated["_l2_context_latent"],
                updated["z_l2_end"],
            )

        if not end_fixed:
            updated["_l2_next_latent"] = self._sample_l2_next_latent(
                updated["_l2_context_latent"],
                l2_condition_kwargs=updated.get("_l2_condition_kwargs"),
            )
            updated["z_l2_end"] = updated["_l2_next_latent"]
        return updated

    def get_rollout_visualization_state(self, condition_kwargs):
        if not condition_kwargs:
            return None
        l2_condition_kwargs = condition_kwargs.get("_l2_condition_kwargs")
        if not l2_condition_kwargs:
            return None
        return self.l2_condition_preprocessor.get_rollout_visualization_state(l2_condition_kwargs)

    def get_visualization_trajectory(self, visualization_state):
        if visualization_state is None:
            return None
        return self.l2_condition_preprocessor.get_visualization_trajectory(visualization_state)

    def update_rollout_visualization_state(self, visualization_state, step_idx):
        if visualization_state is None:
            return None
        return self.l2_condition_preprocessor.update_rollout_visualization_state(
            visualization_state,
            step_idx=step_idx,
        )

    def get_rollout_visualization_trajectory(
        self,
        condition_kwargs,
        num_gen_steps,
        num_pred_frames=1,
        num_condition_frames=None,
    ):
        if not condition_kwargs:
            return None
        l2_condition_kwargs = condition_kwargs.get("_l2_condition_kwargs")
        if not l2_condition_kwargs:
            return None
        return self.l2_condition_preprocessor.get_rollout_visualization_trajectory(
            l2_condition_kwargs,
            num_gen_steps=num_gen_steps,
            num_pred_frames=num_pred_frames,
            num_condition_frames=num_condition_frames,
        )
