import math
import os
from collections import OrderedDict
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from util import get_obj_from_str, instantiate_from_config
from .fm_conditions_v2 import (
    ConditionPreprocessor,
    L2EndpointConditionPreprocessor,
    SpeedYawDirectConditionPreprocessor,
    SpeedYawMovingGoalConditionPreprocessor,
)
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def init_ema_model(model):
    ema_model = deepcopy(model)
    requires_grad(ema_model, False)
    update_ema(ema_model, model, decay=0)
    ema_model.eval()
    return ema_model


def _extract_tokenizer_decode_samples(decoded):
    if isinstance(decoded, tuple):
        return decoded[0]
    return decoded


def _decode_selected_if_branch(tokenizer, branch, latents):
    if branch == "rec":
        decoder = getattr(tokenizer, "decoder", None)
        post_quant_conv = getattr(tokenizer, "post_quant_conv", None)
    else:
        decoder = getattr(tokenizer, "decoder_sem", None)
        post_quant_conv = getattr(tokenizer, "post_quant_conv_sem", None)

    if (
        decoder is not None
        and post_quant_conv is not None
        and getattr(post_quant_conv, "in_channels", None) == latents.size(1)
    ):
        return _extract_tokenizer_decode_samples(decoder(post_quant_conv(latents)))

    rec_channels = tokenizer.quant_conv.out_channels
    sem_channels = tokenizer.quant_conv2.out_channels
    if branch == "rec":
        quant = (
            latents,
            torch.zeros(
                latents.size(0),
                sem_channels,
                latents.size(2),
                latents.size(3),
                device=latents.device,
                dtype=latents.dtype,
            ),
        )
    else:
        quant = (
            torch.zeros(
                latents.size(0),
                rec_channels,
                latents.size(2),
                latents.size(3),
                device=latents.device,
                dtype=latents.dtype,
            ),
            latents,
        )
    return _extract_tokenizer_decode_samples(tokenizer.decode(quant))


class FirstStageHandler:
    def __init__(self, predictor_module):
        self.module = predictor_module

    @property
    def ae(self):
        return self.module.ae

    @property
    def vit(self):
        return self.module.vit

    @property
    def enc_scale(self):
        return self.module.enc_scale

    @property
    def use_precomputed_training_inputs(self):
        return self.module.use_precomputed_training_inputs

    @torch.no_grad()
    def encode_frames(self, images):
        if self.use_precomputed_training_inputs and images.shape[-2:] == self.vit.input_size:
            return images * self.enc_scale

        if images.ndim == 5:
            b, f, _e, _h, _w = images.size()
            images = rearrange(images, "b f e h w -> (b f) e h w")
        else:
            b, _e, _h, _w = images.size()
            f = 1

        x = self.ae.encode(images)["continuous"]
        x = x * self.enc_scale
        return rearrange(x, "(b f) e h w -> b f e h w", b=b, f=f)

    @torch.no_grad()
    def decode_frames(self, x, output_device=None):
        samples = []
        for idx in range(x.shape[1]):
            frame = x[:, idx] / self.enc_scale
            frame = self.ae.post_quant_conv(frame)
            frame = self.ae.decoder(frame)
            if output_device is not None:
                frame = frame.to(output_device)
            samples.append(frame.unsqueeze(1))
        return torch.cat(samples, dim=1)


class IFSelectBranchFirstStageHandler(FirstStageHandler):
    """
    First-stage handler that selects a single IF tokenizer branch, mirroring
    `ModelIFSelectBranch` from the v1 codepath.
    """

    def __init__(self, predictor_module, encoder_branch="rec"):
        super().__init__(predictor_module)
        self.encoder_branch = self._normalize_encoder_branch(encoder_branch)
        expected_channels = self._selected_branch_channels()
        actual_channels = getattr(self.vit, "in_channels", None)
        if actual_channels is not None and actual_channels != expected_channels:
            raise ValueError(
                f"generator_config.params.in_channels={actual_channels} does not match "
                f"the selected {self.encoder_branch} branch channels ({expected_channels})."
            )

    @property
    def enc_scale_dino(self):
        return self.module.enc_scale_dino

    def _normalize_encoder_branch(self, encoder_branch):
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
                f"Unsupported encoder_branch={encoder_branch!r}. "
                "Expected one of: rec, x0, h, sem, x1, h2."
            )
        return aliases[branch]

    def _selected_branch_index(self):
        return 0 if self.encoder_branch == "rec" else 1

    def _selected_branch_scale(self):
        return self.enc_scale if self.encoder_branch == "rec" else self.enc_scale_dino

    def _selected_branch_channels(self):
        if self.encoder_branch == "rec":
            return self.ae.quant_conv.out_channels
        return self.ae.quant_conv2.out_channels

    @torch.no_grad()
    def encode_frames(self, images):
        if self.use_precomputed_training_inputs and images.shape[-2:] == self.vit.input_size:
            return images * self._selected_branch_scale()

        if images.ndim == 5:
            b, f, _e, _h, _w = images.size()
            images = rearrange(images, "b f e h w -> (b f) e h w")
        else:
            b, _e, _h, _w = images.size()
            f = 1

        continuous = self.ae.encode(images)["continuous"]
        if not isinstance(continuous, tuple) or len(continuous) != 2:
            raise ValueError(
                "IFSelectBranchFirstStageHandler expects tokenizer.encode(...)[\"continuous\"] "
                f"to return a 2-tuple, got {type(continuous)}."
            )

        x = continuous[self._selected_branch_index()] * self._selected_branch_scale()
        expected_channels = self._selected_branch_channels()
        if x.size(1) != expected_channels:
            raise ValueError(
                f"Selected {self.encoder_branch} branch has {x.size(1)} channels, "
                f"expected {expected_channels}."
            )

        return rearrange(x, "(b f) e h w -> b f e h w", b=b, f=f)

    @torch.no_grad()
    def decode_frames(self, x, output_device=None):
        b, f, c, _h, _w = x.size()

        expected_channels = self._selected_branch_channels()
        if c != expected_channels:
            raise ValueError(
                f"decode_frames expected {expected_channels} channels for "
                f"encoder_branch={self.encoder_branch}, got {c}."
            )

        if output_device is None:
            x = rearrange(x, "b f c h w -> (b f) c h w")
            selected = x / self._selected_branch_scale()
            samples = _decode_selected_if_branch(self.ae, self.encoder_branch, selected)
            return rearrange(samples, "(b f) c h w -> b f c h w", b=b, f=f)

        samples = []
        for idx in range(f):
            selected = x[:, idx] / self._selected_branch_scale()
            frame = _decode_selected_if_branch(self.ae, self.encoder_branch, selected)
            samples.append(frame.to(output_device).unsqueeze(1))
        return torch.cat(samples, dim=1)


class FlowMatchingObjective:
    def __init__(self, predictor_module, sigma_min=1e-5):
        self.module = predictor_module
        self.sigma_min = sigma_min

    def alpha(self, t):
        return 1.0 - t

    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)

    def A(self, t):
        return 1.0

    def B(self, t):
        return -(1.0 - self.sigma_min)

    def sample_t(self, x):
        return torch.rand((x.shape[0],), device=x.device)

    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        if t.dim() == 2:
            shape = [x.shape[0], x.shape[1]] + [1] * (x.dim() - 2)
        else:
            shape = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*shape) * x + self.sigma(t).view(*shape) * noise
        return x_t, noise

    def get_supervision_target(self, target, noise, t):
        return self.A(t) * target + self.B(t) * noise

    def compute_loss(self, pred, target):
        return (pred.float() - target.float()) ** 2

    def prepare_training_inputs(self, x):
        raise NotImplementedError

    def compute_step(self, x, frame_rate, condition_kwargs=None):
        model_input, t, supervision_target, prediction_slice = self.prepare_training_inputs(x)
        model_condition_kwargs = self.module.condition_preprocessor.get_model_condition_kwargs(
            condition_kwargs
        )
        pred = self.module.vit(model_input, t, frame_rate=frame_rate, **model_condition_kwargs)
        if prediction_slice is not None:
            pred = pred[:, prediction_slice]
        return self.compute_loss(pred, supervision_target)


class FlowMatchingObjectiveTeacherForcing(FlowMatchingObjective):
    def __init__(
        self,
        predictor_module,
        sigma_min=1e-5,
        ctx_noise_aug_ratio=0.1,
        ctx_noise_aug_prob=0.5,
        drop_ctx_rate=0.2,
    ):
        super().__init__(predictor_module, sigma_min=sigma_min)
        self.ctx_noise_aug_ratio = ctx_noise_aug_ratio
        self.ctx_noise_aug_prob = ctx_noise_aug_prob
        self.drop_ctx_rate = drop_ctx_rate

    def _split_sequence(self, x):
        context = x[:, :-self.module.num_pred_frames]
        target = x[:, -self.module.num_pred_frames:]
        return context, target

    def _maybe_drop_context(self, context, target):
        if context.size(1) == 0:
            return context
        if torch.rand(1, device=target.device) < self.drop_ctx_rate:
            return context[:, :0]
        return context

    def _maybe_noise_context(self, context, t):
        if context.size(1) == 0:
            return context
        if torch.rand(1, device=context.device) >= self.ctx_noise_aug_prob:
            return context

        mask = t >= self.ctx_noise_aug_ratio
        if not mask.any():
            return context

        context = context.clone()
        aug_noise = torch.randn_like(context)
        context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio
        return context

    def _prepare_context(self, context, target, t):
        if not self.module.training:
            return context

        context = self._maybe_drop_context(context, target)
        if context.size(1) == 0:
            return context
        return self._maybe_noise_context(context, t)

    def prepare_training_inputs(self, x):
        context, target = self._split_sequence(x)
        t = self.sample_t(target)
        context = self._prepare_context(context, target, t)
        target_t, noise = self.add_noise(target, t)

        model_input = torch.cat([context, target_t], dim=1) if context.size(1) > 0 else target_t

        supervision_target = self.get_supervision_target(target, noise, t)
        prediction_slice = slice(-self.module.num_pred_frames, None)
        model_t = self.module.sampler._build_model_t(context, target_t, t)
        return model_input, model_t, supervision_target, prediction_slice


class FlowMatchingSampler:
    """
    Solver-agnostic base for v2 inference sampling.

    The context frames are kept clean in `model_input`; only the target frame
    block is initialized with noise and integrated during sampling. Concrete
    subclasses implement `_step` to define the per-iteration integration rule
    (e.g. Euler-Maruyama, Heun).
    """

    def __init__(
        self,
        predictor_module,
        timescale=1.0,
        integration_t_eps=0.0,
        timestep_conditioning="global",
    ):
        self.module = predictor_module
        self.timescale = timescale
        self.integration_t_eps = float(integration_t_eps)
        self.timestep_conditioning = self._normalize_timestep_conditioning(timestep_conditioning)

    def _normalize_timestep_conditioning(self, timestep_conditioning):
        mode = str(timestep_conditioning).strip().lower()
        if mode not in {"global", "per_frame"}:
            raise ValueError(
                "timestep_conditioning must be one of {'global', 'per_frame'}, "
                f"got {timestep_conditioning!r}."
            )
        return mode

    def _get_net(self, sample_with_ema):
        return self.module.ema_vit if sample_with_ema else self.module.vit

    def _prepare_context(self, images, latent):
        if images is None:
            return None
        if latent:
            return images.clone()
        return self.module.encode_frames(images)

    def _default_frame_rate(self, num_samples, device):
        return torch.full_like(torch.ones((num_samples,)), 5, device=device)

    def _get_input_hw(self):
        if isinstance(self.module.vit.input_size, (list, tuple, ListConfig)):
            return self.module.vit.input_size[0], self.module.vit.input_size[1]
        return self.module.vit.input_size, self.module.vit.input_size

    def _build_model_inputs(self, context, target_t, t):
        if context is not None and context.size(1) > 0:
            model_input = torch.cat([context, target_t], dim=1)
        else:
            model_input = target_t
        return model_input

    def _build_model_t(self, context, target_t, t_scalar):
        if self.timestep_conditioning == "global":
            return t_scalar

        target_t_full = t_scalar.unsqueeze(1).expand(-1, target_t.size(1))
        if context is None or context.size(1) == 0:
            return target_t_full
        context_t = torch.zeros(
            t_scalar.size(0),
            context.size(1),
            device=t_scalar.device,
            dtype=t_scalar.dtype,
        )
        return torch.cat([context_t, target_t_full], dim=1)

    def _extract_target_prediction(self, pred):
        return pred[:, -self.module.num_pred_frames :]

    def _snapshot_condition_kwargs(self, condition_kwargs):
        if not condition_kwargs:
            return {}
        snapshot = {}
        for key, value in condition_kwargs.items():
            if torch.is_tensor(value):
                snapshot[key] = value.detach().cpu().clone()
            else:
                snapshot[key] = value
        return snapshot

    def _prepare_sampling_state(
        self, images, latent, sample_with_ema, num_samples, frame_rate, condition_kwargs
    ):
        net = self._get_net(sample_with_ema)
        device = next(net.parameters()).device
        context = self._prepare_context(images, latent)
        condition_kwargs = self.module.condition_preprocessor.prepare_condition_kwargs(
            condition_kwargs,
            batch_size=num_samples,
            device=device,
            split="sample",
        )
        model_condition_kwargs = self.module.condition_preprocessor.get_model_condition_kwargs(
            condition_kwargs
        )

        if frame_rate is None:
            frame_rate = self._default_frame_rate(num_samples, device)

        input_h, input_w = self._get_input_hw()
        target_t = torch.randn(
            num_samples,
            self.module.num_pred_frames,
            self.module.vit.in_channels,
            input_h,
            input_w,
            device=device,
        )
        return net, device, context, model_condition_kwargs, frame_rate, target_t

    def _build_t_steps(self, NFE, device):
        if not 0.0 <= self.integration_t_eps < 0.5:
            raise ValueError(
                "integration_t_eps must be in [0, 0.5), "
                f"got {self.integration_t_eps}."
            )
        return torch.linspace(
            1.0 - self.integration_t_eps,
            self.integration_t_eps,
            NFE + 1,
            device=device,
        )

    def _eval_velocity(self, net, context, target_t, t_scalar, frame_rate, model_condition_kwargs):
        model_input = self._build_model_inputs(context, target_t, t_scalar)
        model_t = self._build_model_t(context, target_t, t_scalar)
        pred = net(model_input, t=model_t * self.timescale, frame_rate=frame_rate, **model_condition_kwargs)
        # Clone: under torch.compile(mode="reduce-overhead") (CUDA graphs), `pred`
        # is a view into a static output buffer that gets overwritten by the next
        # call to `net` — multi-eval-per-step solvers (e.g. Heun) call `net` again
        # before consuming this result, so it must be an independent copy.
        return self._extract_target_prediction(pred).clone()

    def _validate_sample_kwargs(self, eta, NFE):
        """No-op hook; solvers override to enforce solver-specific constraints."""

    def _step(self, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs, eta):
        raise NotImplementedError(f"{self.__class__.__name__} must implement `_step`.")

    @torch.no_grad()
    def sample(
        self,
        images=None,
        latent=False,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        condition_kwargs=None,
        return_sample=False,
    ):
        self._validate_sample_kwargs(eta, NFE)
        net, device, context, model_condition_kwargs, frame_rate, target_t = self._prepare_sampling_state(
            images, latent, sample_with_ema, num_samples, frame_rate, condition_kwargs
        )
        t_steps = self._build_t_steps(NFE, device)
        for i in range(NFE):
            target_t = self._step(
                net, context, target_t, t_steps[i], t_steps[i + 1], frame_rate, model_condition_kwargs, eta
            )

        if return_sample:
            return target_t, self.module.decode_frames(target_t.clone())
        return target_t

    def _update_rollout_context(self, context, prediction):
        latest = prediction[:, -self.module.num_pred_frames:]
        if self.module.num_pred_frames > context.size(1):
            return latest
        return torch.cat([context[:, self.module.num_pred_frames :], latest], dim=1)

    @torch.no_grad()
    def roll_out(
        self,
        x_0,
        num_gen_frames=25,
        latent_input=True,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        condition_kwargs=None,
        decode_device=None,
        return_condition_history=False,
    ):
        context = x_0.clone() if latent_input else self.module.encode_frames(x_0)
        all_latents = context.clone()
        condition_kwargs = self.module.condition_preprocessor.prepare_condition_kwargs(
            condition_kwargs,
            batch_size=context.size(0),
            device=context.device,
            split="rollout",
        )
        condition_history = []

        for _idx in tqdm(range(num_gen_frames)):
            if return_condition_history:
                condition_history.append(self._snapshot_condition_kwargs(condition_kwargs))
            prediction = self.sample(
                images=context,
                latent=True,
                eta=eta,
                NFE=NFE,
                sample_with_ema=sample_with_ema,
                num_samples=num_samples,
                frame_rate=frame_rate,
                condition_kwargs=condition_kwargs,
            )
            all_latents = torch.cat([all_latents, prediction[:, -self.module.num_pred_frames :]], dim=1)
            if _idx < num_gen_frames - 1:
                condition_kwargs = self.module.condition_preprocessor.update_rollout_condition_kwargs(
                    condition_kwargs,
                    prediction=prediction,
                    context=context,
                    step_idx=_idx,
                )
            context = self._update_rollout_context(context, prediction)

        result = (all_latents, self.module.decode_frames(all_latents, output_device=decode_device))
        if return_condition_history:
            return result + (condition_history,)
        return result


def _euler_maruyama_update(eval_velocity, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs, eta):
    t_scalar = t_i.repeat(target_t.shape[0])
    neg_v = eval_velocity(net, context, target_t, t_scalar, frame_rate, model_condition_kwargs)
    dt = t_i - t_ip1
    dw = torch.randn(target_t.size(), device=target_t.device) * torch.sqrt(dt)
    diffusion = dt
    return target_t + neg_v * dt + eta * torch.sqrt(2 * diffusion) * dw


def _heun_update(eval_velocity, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs):
    t_i_scalar = t_i.repeat(target_t.shape[0])
    t_ip1_scalar = t_ip1.repeat(target_t.shape[0])
    dt = t_i - t_ip1
    v1 = eval_velocity(net, context, target_t, t_i_scalar, frame_rate, model_condition_kwargs)
    x_pred = target_t + v1 * dt
    v2 = eval_velocity(net, context, x_pred, t_ip1_scalar, frame_rate, model_condition_kwargs)
    return target_t + 0.5 * (v1 + v2) * dt


class FlowMatchingSamplerEuler(FlowMatchingSampler):
    """Euler-Maruyama ODE/SDE solve: one network evaluation per step."""

    def _step(self, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs, eta):
        return _euler_maruyama_update(
            self._eval_velocity, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs, eta
        )


class FlowMatchingSamplerHeun(FlowMatchingSampler):
    """
    Deterministic 2nd-order (Heun / improved-Euler) predictor-corrector solver.
    Two network evaluations per step; not compatible with the stochastic eta term.
    """

    def _validate_sample_kwargs(self, eta, NFE):
        if eta != 0.0:
            raise ValueError(
                "FlowMatchingSamplerHeun is deterministic-only (Heun's 2nd-order "
                f"correction cannot be combined with the stochastic SDE term); got eta={eta}. "
                "Set eta=0.0."
            )

    def _step(self, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs, eta):
        del eta  # validated to be 0.0 in _validate_sample_kwargs
        return _heun_update(
            self._eval_velocity, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs
        )


def _heun_schedule_step(eval_velocity, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs, eta):
    del eta  # Heun steps are deterministic-only; validated in _validate_sample_kwargs
    return _heun_update(eval_velocity, net, context, target_t, t_i, t_ip1, frame_rate, model_condition_kwargs)


class FlowMatchingSamplerHeunPlusEuler(FlowMatchingSampler):
    """
    Runs an explicit, ordered schedule of integration steps, each assigned a
    solver (heun or euler) and a step size (fraction of the
    [integration_t_eps, 1 - integration_t_eps] range). Step sizes must sum to
    exactly that range's width.
    """

    _SOLVER_UPDATE_FNS = {
        "heun": _heun_schedule_step,
        "euler": _euler_maruyama_update,
    }

    def __init__(
        self,
        predictor_module,
        timescale=1.0,
        integration_t_eps=0.0,
        timestep_conditioning="global",
        step_schedule=None,
    ):
        super().__init__(
            predictor_module,
            timescale=timescale,
            integration_t_eps=integration_t_eps,
            timestep_conditioning=timestep_conditioning,
        )
        self.step_schedule = self._expand_and_validate_schedule(step_schedule)
        self._has_heun_steps = any(solver == "heun" for solver, _ in self.step_schedule)

    def _expand_and_validate_schedule(self, step_schedule):
        if not step_schedule:
            raise ValueError("step_schedule must be a non-empty list of blocks.")

        expanded = []
        for block_idx, block in enumerate(step_schedule):
            block = dict(block)  # tolerate OmegaConf DictConfig entries, same pattern as sample_t_grids
            solver = str(block.get("solver", "")).strip().lower()
            if solver not in self._SOLVER_UPDATE_FNS:
                raise ValueError(
                    f"step_schedule[{block_idx}].solver={solver!r} must be one of "
                    f"{sorted(self._SOLVER_UPDATE_FNS)}."
                )
            num_steps = int(block.get("num_steps", 1))
            if num_steps < 1:
                raise ValueError(f"step_schedule[{block_idx}].num_steps must be >= 1, got {num_steps}.")
            step_size = float(block["step_size"])
            if step_size <= 0.0:
                raise ValueError(f"step_schedule[{block_idx}].step_size must be > 0, got {step_size}.")
            expanded.extend([(solver, step_size)] * num_steps)

        total = sum(step_size for _, step_size in expanded)
        expected = 1.0 - 2 * self.integration_t_eps
        if abs(total - expected) > 1e-6:
            raise ValueError(
                f"step_schedule step sizes sum to {total}, expected {expected} "
                f"(1 - 2 * integration_t_eps={self.integration_t_eps})."
            )
        return expanded

    def _validate_sample_kwargs(self, eta, NFE):
        del NFE  # step count is fully determined by step_schedule, not NFE
        if self._has_heun_steps and eta != 0.0:
            raise ValueError(
                "step_schedule contains Heun steps, which are deterministic-only and cannot "
                f"be combined with the stochastic SDE term; got eta={eta}. Set eta=0.0."
            )

    def _build_schedule_t_steps(self, device):
        t = 1.0 - self.integration_t_eps
        t_values = [t]
        for _, step_size in self.step_schedule:
            t -= step_size
            t_values.append(t)
        return torch.tensor(t_values, device=device)

    @torch.no_grad()
    def sample(
        self,
        images=None,
        latent=False,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        condition_kwargs=None,
        return_sample=False,
    ):
        self._validate_sample_kwargs(eta, NFE)  # NFE accepted only for call-site compatibility
        net, device, context, model_condition_kwargs, frame_rate, target_t = self._prepare_sampling_state(
            images, latent, sample_with_ema, num_samples, frame_rate, condition_kwargs
        )
        t_steps = self._build_schedule_t_steps(device)
        for i, (solver, _step_size) in enumerate(self.step_schedule):
            target_t = self._SOLVER_UPDATE_FNS[solver](
                self._eval_velocity, net, context, target_t, t_steps[i], t_steps[i + 1],
                frame_rate, model_condition_kwargs, eta,
            )

        if return_sample:
            return target_t, self.module.decode_frames(target_t.clone())
        return target_t


class PredictorModule(pl.LightningModule):
    def __init__(
        self,
        *,
        tokenizer_config,
        generator_config,
        objective_config=None,
        sampler_config=None,
        condition_preprocessor_config=None,
        first_stage_handler_config=None,
        adjust_lr_to_batch_size=False,
        num_pred_frames=1,
        warmup_steps=5000,
        min_lr_multiplier=0.1,
        enc_scale=4,
        enc_scale_dino=3.45062,
        use_precomputed_training_inputs=False,
        init_weights_path=None,
        allow_different_resolution_checkpoint=False,
    ):
        super().__init__()

        self.num_pred_frames = num_pred_frames
        self.enc_scale = enc_scale
        self.enc_scale_dino = enc_scale_dino
        self.allow_different_resolution_checkpoint = allow_different_resolution_checkpoint
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.use_precomputed_training_inputs = use_precomputed_training_inputs

        self.vit = self.build_generator(generator_config)
        self.ae = self.build_tokenizer(tokenizer_config)
        self.ema_vit = init_ema_model(self.vit)

        self.first_stage = self.build_first_stage(first_stage_handler_config)
        self.condition_preprocessor = self.build_condition_preprocessor(condition_preprocessor_config)
        self.objective = self.build_objective(objective_config)
        self.sampler = self.build_sampler(sampler_config)

        if init_weights_path is not None:
            self.init_weights_from_checkpoint(init_weights_path)

    def setup(self, stage=None):
        """Exclude 'unused' parameters from DDP gradient reduction."""
        super().setup(stage)
        # EMA
        if hasattr(self, "ema_vit") and self.ema_vit is not None:
            self.ema_vit.requires_grad_(False)
        self.ae.requires_grad_(False)

    def init_weights_from_checkpoint(self, init_weights_path):
        checkpoint_path = os.path.expandvars(init_weights_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Initial weights {init_weights_path} does not exist.")

        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = self._prepare_second_stage_checkpoint(state_dict, checkpoint_path)
        outcome = self.load_state_dict(state_dict, strict=False)
        assert outcome.missing_keys == [], outcome.missing_keys
        print(f"Loaded model from {init_weights_path}")

    def build_tokenizer(self, tokenizer_config):
        tokenizer_folder = os.path.expandvars(tokenizer_config.folder)
        ckpt_path = tokenizer_config.ckpt_path if tokenizer_config.ckpt_path else "checkpoints/last.ckpt"

        tokenizer_config = OmegaConf.load(os.path.join(tokenizer_folder, "config.yaml"))
        model_cfg = OmegaConf.to_container(tokenizer_config.model, resolve=False)
        for key in ("loss_config", "entropy_loss_weight_scheduler_config"):
            model_cfg.get("params", {}).pop(key, None)
        model_cfg.get("params", {})["distill_model_type"] = None
        encoder_params = model_cfg.get("params", {}).get("encoder_config", {}).get("params", {})
        if isinstance(encoder_params, dict):
            encoder_params["use_pretrained_weights"] = False
        model = instantiate_from_config(OmegaConf.create(model_cfg))

        checkpoint_path = os.path.join(tokenizer_folder, ckpt_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
        checkpoint = self._prepare_tokenizer_checkpoint(model, checkpoint, tokenizer_config.model, checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model

    def build_generator(self, generator_config):
        return instantiate_from_config(generator_config)

    def _build_helper_from_config(self, config, default_target):
        config = config or {"target": default_target, "params": {}}
        params = dict(config.get("params", {}))
        params["predictor_module"] = self
        helper_cls = get_obj_from_str(config["target"])
        return helper_cls(**params)

    def build_objective(self, objective_config):
        return self._build_helper_from_config(
            objective_config,
            "models.second_stage.fm_model_v2.FlowMatchingObjectiveTeacherForcing",
        )

    def build_sampler(self, sampler_config):
        return self._build_helper_from_config(
            sampler_config,
            "models.second_stage.fm_model_v2.FlowMatchingSamplerEuler",
        )

    def build_condition_preprocessor(self, condition_preprocessor_config):
        return self._build_helper_from_config(
            condition_preprocessor_config,
            "models.second_stage.fm_conditions_v2.ConditionPreprocessor",
        )

    def build_first_stage(self, first_stage_handler_config):
        return self._build_helper_from_config(
            first_stage_handler_config,
            "models.second_stage.fm_model_v2.FirstStageHandler",
        )

    def _require_or_allow_resolution_mismatch(self, *, name, checkpoint_shape, model_shape, source):
        if checkpoint_shape == model_shape:
            return
        if not self.allow_different_resolution_checkpoint:
            raise ValueError(
                f"{name} shape mismatch between checkpoint and current model: "
                f"checkpoint={checkpoint_shape}, model={model_shape}. "
                f"Source: {source}. "
                "Set allow_different_resolution_checkpoint=True to adapt or skip "
                "resolution-dependent positional embeddings."
            )

    def _prepare_tokenizer_checkpoint(self, model, checkpoint, tokenizer_model_config, checkpoint_path):
        state_dict = checkpoint.copy()
        target = tokenizer_model_config.get("target", "")
        encoder_config = tokenizer_model_config.params.get("encoder_config")
        encoder_target = encoder_config.get("target", "") if encoder_config else ""

        if target != "models.first_stage.vqgan.VQModel" or encoder_target != "networks.tokenizer.pretrained_models.Encoder":
            return state_dict

        model_pos_embed = getattr(getattr(model.encoder, "encoder", None), "pos_embed", None)
        checkpoint_key = "encoder.encoder.pos_embed"
        checkpoint_pos_embed = state_dict.get(checkpoint_key)

        if model_pos_embed is None or checkpoint_pos_embed is None:
            return state_dict

        self._require_or_allow_resolution_mismatch(
            name=f"Tokenizer positional embedding ({checkpoint_key})",
            checkpoint_shape=tuple(checkpoint_pos_embed.shape),
            model_shape=tuple(model_pos_embed.shape),
            source=checkpoint_path,
        )

        if checkpoint_pos_embed.shape != model_pos_embed.shape:
            num_prefix_tokens = getattr(model.encoder.encoder, "num_prefix_tokens", 1)
            target_grid = model.encoder.encoder.patch_embed.grid_size
            state_dict[checkpoint_key] = resample_abs_pos_embed(
                checkpoint_pos_embed,
                new_size=target_grid,
                num_prefix_tokens=num_prefix_tokens,
            )

        return state_dict

    def _prepare_second_stage_checkpoint(self, checkpoint, checkpoint_path):
        state_dict = checkpoint.copy()
        for key in ("vit.pos_embed", "ema_vit.pos_embed"):
            checkpoint_pos_embed = state_dict.get(key)
            model_pos_embed = self.state_dict().get(key)
            if checkpoint_pos_embed is None or model_pos_embed is None:
                continue

            self._require_or_allow_resolution_mismatch(
                name=f"Second-stage positional embedding ({key})",
                checkpoint_shape=tuple(checkpoint_pos_embed.shape),
                model_shape=tuple(model_pos_embed.shape),
                source=checkpoint_path,
            )

            if checkpoint_pos_embed.shape != model_pos_embed.shape:
                del state_dict[key]

        return state_dict

    def get_warmup_scheduler(self, optimizer, warmup_steps=1, min_lr_multiplier=1.0):
        batches = len(self.trainer.datamodule.train_dataloader())
        steps_per_epoch = batches // self.trainer.accumulate_grad_batches
        total_steps = self.trainer.max_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (min(step, total_steps) - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (1 - min_lr_multiplier) * cosine_decay + min_lr_multiplier

        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        params = [p for p in self.vit.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_input(self, batch, k):
        if isinstance(batch, dict):
            x = batch[k]
            frame_rate = batch["frame_rate"]
        else:
            x = batch
            frame_rate = None
        assert len(x.shape) == 5 or self.use_precomputed_training_inputs, "When using images, input must be 5D tensor"
        return x, frame_rate

    @torch.no_grad()
    def encode_frames(self, images):
        return self.first_stage.encode_frames(images)

    @torch.no_grad()
    def decode_frames(self, x, output_device=None):
        return self.first_stage.decode_frames(x, output_device=output_device)

    def compute_prediction_loss(self, pred, target):
        return self.objective.compute_loss(pred, target)

    def log_training_losses(self, loss):
        self.log("train/loss", loss.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def log_validation_losses(self, loss):
        self.log("val/loss", loss.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def log_adaln_mean_abs(self, split, condition_kwargs=None):
        stats_getter = getattr(self.vit, "get_last_adaln_mean_abs", None)
        if not callable(stats_getter):
            return

        stats = stats_getter()
        if not stats:
            return

        aggregated = {}
        for name, value in stats.items():
            _, stat_name = name.split("/", 1)
            aggregated.setdefault(stat_name, []).append(value)

        group_masks = self._get_adaln_stat_group_masks(condition_kwargs)
        for stat_name, values in aggregated.items():
            values_tensor = torch.stack(values)
            for group_name, mask in group_masks.items():
                group_values = values_tensor if mask is None else values_tensor[:, mask]
                self._log_adaln_stat_values(split, group_name, stat_name, group_values)

    def _get_adaln_stat_group_masks(self, condition_kwargs):
        if not condition_kwargs or "steering" not in condition_kwargs:
            return {"all": None}

        steering = condition_kwargs["steering"]
        if not torch.is_tensor(steering):
            return {"all": None}

        present_mask = ~torch.isnan(steering).flatten(1).all(dim=1)
        return {
            "present": present_mask,
            "missing": ~present_mask,
        }

    def _log_adaln_stat_values(self, split, group_name, stat_name, values):
        # Always emit exactly 2 log calls (mean + std) regardless of group size so that
        # sync_dist=True allreduces are consistent across ranks even when different ranks
        # receive batches with different steering availability (e.g. mixed-dataset training).
        values = values.reshape(-1)
        nan = torch.tensor(float("nan"), device=values.device)
        mean_val = values.mean() if values.numel() > 0 else nan
        std_val = values.std(unbiased=False) if values.numel() >= 2 else nan

        self.log(
            f"{split}/adaln/{group_name}/{stat_name}/mean",
            mean_val,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{split}/adaln/{group_name}/{stat_name}/std",
            std_val,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def log_condition_embeddings(self, split, condition_kwargs=None):
        getter = getattr(self.vit, 'get_last_condition_embeddings', None)
        if not callable(getter):
            return
        embeddings = getter()
        if not embeddings:
            return
        group_masks = self._get_adaln_stat_group_masks(condition_kwargs)
        for emb_name, embedding in embeddings.items():
            per_sample_mean_abs = embedding.abs().mean(dim=-1)
            per_sample_norm = embedding.norm(dim=-1)
            for stat_name, values in (("mean_abs", per_sample_mean_abs), ("norm", per_sample_norm)):
                for group_name, mask in group_masks.items():
                    group_values = values if mask is None else values[mask]
                    group_values = group_values.reshape(-1)
                    # Always emit exactly 2 log calls per group so sync_dist=True allreduces
                    # are consistent across ranks regardless of per-rank batch composition.
                    nan = torch.tensor(float("nan"), device=group_values.device)
                    mean_val = group_values.mean() if group_values.numel() > 0 else nan
                    std_val = group_values.std(unbiased=False) if group_values.numel() >= 2 else nan
                    self.log(
                        f"{split}/cond_emb/{emb_name}/{group_name}/{stat_name}/mean",
                        mean_val,
                        prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                    )
                    self.log(
                        f"{split}/cond_emb/{emb_name}/{group_name}/{stat_name}/std",
                        std_val,
                        prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                    )

    def log_adaln_gradients(self):
        getter = getattr(self.vit, 'get_adaln_grad_stats', None)
        if not callable(getter):
            return
        stats = getter()
        for name, g in stats.items():
            g_flat = g.float().flatten()
            self.log(
                f"train/adaln_grad/{name}/norm",
                g_flat.norm(),
                prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True,
            )
            self.log(
                f"train/adaln_grad/{name}/mean_abs",
                g_flat.abs().mean(),
                prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True,
            )

    def on_after_backward(self):
        self.log_adaln_gradients()

    def _shared_step(self, batch, split):
        images, frame_rate = self.get_input(batch, "images")
        x = self.encode_frames(images)
        condition_kwargs = self.condition_preprocessor.get_condition_kwargs_from_batch(batch, split=split)
        loss = self.objective.compute_step(x, frame_rate, condition_kwargs=condition_kwargs)

        if split == "train":
            self.log_training_losses(loss)
            self.log_adaln_mean_abs("train", condition_kwargs)
            self.log_condition_embeddings("train", condition_kwargs)
            return loss.mean()

        self.log_validation_losses(loss)
        self.log_adaln_mean_abs("val", condition_kwargs)
        self.log_condition_embeddings("val", condition_kwargs)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def _sync_ema_stream(self):
        """Block until the async EMA update finishes before reading ema_vit."""
        stream = getattr(self, "_ema_stream", None)
        if stream is not None:
            stream.synchronize()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA asynchronously on a separate CUDA stream."""
        if not hasattr(self, "_ema_stream"):
            self._ema_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        if self._ema_stream is not None:
            with torch.cuda.stream(self._ema_stream):
                update_ema(self.ema_vit, self.vit)
        else:
            update_ema(self.ema_vit, self.vit)

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     update_ema(self.ema_vit, self.vit)

    @torch.no_grad()
    def sample(
        self,
        images=None,
        latent=False,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        condition_kwargs=None,
        return_sample=False,
    ):
        if sample_with_ema:
            self._sync_ema_stream()
            
        return self.sampler.sample(
            images=images,
            latent=latent,
            eta=eta,
            NFE=NFE,
            sample_with_ema=sample_with_ema,
            num_samples=num_samples,
            frame_rate=frame_rate,
            condition_kwargs=condition_kwargs,
            return_sample=return_sample,
        )

    def _validate_rollout_context(self, x_0):
        if not isinstance(x_0, dict):
            raise TypeError(
                f"{self.__class__.__name__}.roll_out expects x_0 to be a dict, "
                f"got {type(x_0).__name__}."
            )
        if "images" not in x_0:
            raise KeyError(f"{self.__class__.__name__}.roll_out requires x_0['images'].")

    def _move_rollout_context_to_device(self, x_0, device):
        moved = {}
        for key, value in x_0.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _get_rollout_condition_frames(self, images, num_condition_frames=None):
        if num_condition_frames is None:
            num_condition_frames = getattr(self.vit, "num_context_frames", None)
        if num_condition_frames is None:
            num_condition_frames = images.size(1) - self.num_pred_frames
        num_condition_frames = int(num_condition_frames)
        if num_condition_frames <= 0:
            raise ValueError(
                f"rollout batch has {images.size(1)} frames, but "
                f"num_pred_frames={self.num_pred_frames}; expected at least one "
                "conditioning frame."
            )
        if num_condition_frames > images.size(1):
            raise ValueError(
                f"num_condition_frames={num_condition_frames} exceeds rollout "
                f"batch length {images.size(1)}."
            )
        return num_condition_frames

    def roll_out(
        self,
        x_0,
        num_gen_frames=25,
        latent_input=True,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        condition_kwargs=None,
        decode_device=None,
        return_condition_history=False,
        num_condition_frames=None,
    ):
        device = next(self.parameters()).device
        self._validate_rollout_context(x_0)
        x_0 = self._move_rollout_context_to_device(x_0, device)
        images = x_0["images"]
        num_condition_frames = self._get_rollout_condition_frames(
            images,
            num_condition_frames=num_condition_frames,
        )
        cond_x = images[:, :num_condition_frames]
        rollout_context = dict(x_0)
        rollout_context["images"] = cond_x

        if torch.is_tensor(frame_rate):
            frame_rate = frame_rate.to(device)
        elif frame_rate is None and torch.is_tensor(rollout_context.get("frame_rate")):
            frame_rate = rollout_context["frame_rate"]

        if condition_kwargs is None:
            condition_kwargs = self.condition_preprocessor.get_condition_kwargs_from_batch(
                rollout_context,
                split="rollout",
            )

        return self.sampler.roll_out(
            cond_x,
            num_gen_frames=num_gen_frames,
            latent_input=latent_input,
            eta=eta,
            NFE=NFE,
            sample_with_ema=sample_with_ema,
            num_samples=num_samples,
            frame_rate=frame_rate,
            condition_kwargs=condition_kwargs,
            decode_device=decode_device,
            return_condition_history=return_condition_history,
        )

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = {}
        images, frame_rate = self.get_input(batch, "images")
        N = min(5, images.size(0))
        images = images[:N]
        condition_kwargs = self.condition_preprocessor.get_condition_kwargs_from_batch(batch, split="log_images")
        condition_kwargs = self.condition_preprocessor.slice_condition_kwargs(condition_kwargs, slice(0, N))

        if self.use_precomputed_training_inputs and images.shape[-2:] == self.vit.input_size:
            images = self.decode_frames(images)

        images = self.condition_preprocessor.annotate_logged_images(images, batch=batch, num_images=N)
        frame_rate = frame_rate[:N] if frame_rate is not None else None
        num_frames = images.size(1)

        visual = [images[:, frame_idx] for frame_idx in range(num_frames)]
        visual_ema = [images[:, frame_idx] for frame_idx in range(num_frames)]

        context = images[:, :-self.num_pred_frames] if num_frames - self.num_pred_frames > 0 else None

        samples = self.sample(
            context,
            eta=0.0,
            NFE=30,
            sample_with_ema=False,
            num_samples=N,
            frame_rate=frame_rate,
            condition_kwargs=condition_kwargs,
            return_sample=True,
        )[1]
        for frame_idx in range(samples.size(1)):
            visual.append(samples[:, frame_idx])

        samples_ema = self.sample(
            context,
            eta=0.0,
            NFE=30,
            sample_with_ema=True,
            num_samples=N,
            frame_rate=frame_rate,
            condition_kwargs=condition_kwargs,
            return_sample=True,
        )[1]
        for frame_idx in range(samples_ema.size(1)):
            visual_ema.append(samples_ema[:, frame_idx])

        sampled = vutils.make_grid(torch.cat(torch.chunk(torch.cat(visual, dim=0), 4, dim=0), dim=0), nrow=N, padding=2, normalize=False)
        sampled_ema = vutils.make_grid(torch.cat(torch.chunk(torch.cat(visual_ema, dim=0), 4, dim=0), dim=0), nrow=N, padding=2, normalize=False)

        log["sampled"] = sampled
        log["ema_sampled"] = sampled_ema
        self.vit.train()
        return log


class L1L2PredL2PredictorModule(PredictorModule):
    """
    PredictorModule specialization for L1 rollout conditioned by an L2 endpoint
    predictor.

    The base PredictorModule keeps a tensor-level rollout API with explicit
    condition_kwargs. This subclass treats rollout as a batch-level L1/L2
    operation: extract the L1 context, derive the L2 condition through the
    configured condition preprocessor, then delegate to the generic sampler.
    """

    pass
