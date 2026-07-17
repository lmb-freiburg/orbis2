import warnings

import torch
from omegaconf import ListConfig
from tqdm import tqdm


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

    If `step_schedule` is omitted (the default), NFE drives the schedule
    instead: NFE-1 uniform Heun steps followed by 1 uniform Euler step, sized
    from NFE and integration_t_eps like every other sampler in this file.
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
        self.step_schedule = (
            None if step_schedule is None else self._expand_and_validate_schedule(step_schedule)
        )

    def _expand_and_validate_schedule(self, step_schedule):
        if not step_schedule:
            raise ValueError("step_schedule, if provided, must be a non-empty list of blocks.")

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

    def _resolve_schedule(self, NFE):
        if self.step_schedule is not None:
            return self.step_schedule
        if NFE < 1:
            raise ValueError(f"NFE must be >= 1, got {NFE}.")
        step_size = (1.0 - 2 * self.integration_t_eps) / NFE
        return [("heun", step_size)] * (NFE - 1) + [("euler", step_size)]

    def _validate_sample_kwargs(self, eta, NFE):
        schedule = self._resolve_schedule(NFE)
        # When an explicit step_schedule is configured, NFE is only accepted
        # for call-site compatibility (roll_out/PredictorModule/CLI scripts
        # always pass some NFE value); warn rather than raise on mismatch
        # since callers like log_images/roll_out pass a fixed NFE default
        # with no real intent behind the specific number. When step_schedule
        # is unset, NFE genuinely drives the schedule (see _resolve_schedule).
        if self.step_schedule is not None and NFE != len(self.step_schedule):
            warnings.warn(
                f"{self.__class__.__name__} ignores NFE={NFE}; it always runs its "
                f"configured step_schedule ({len(self.step_schedule)} steps).",
                stacklevel=3,
            )
        has_heun_steps = any(solver == "heun" for solver, _ in schedule)
        if has_heun_steps and eta != 0.0:
            raise ValueError(
                "The active step schedule contains Heun steps, which are deterministic-only "
                f"and cannot be combined with the stochastic SDE term; got eta={eta}. Set eta=0.0."
            )

    def _build_schedule_t_steps(self, schedule, device):
        t = 1.0 - self.integration_t_eps
        t_values = [t]
        for _, step_size in schedule:
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
        self._validate_sample_kwargs(eta, NFE)
        schedule = self._resolve_schedule(NFE)
        net, device, context, model_condition_kwargs, frame_rate, target_t = self._prepare_sampling_state(
            images, latent, sample_with_ema, num_samples, frame_rate, condition_kwargs
        )
        t_steps = self._build_schedule_t_steps(schedule, device)
        for i, (solver, _step_size) in enumerate(schedule):
            target_t = self._SOLVER_UPDATE_FNS[solver](
                self._eval_velocity, net, context, target_t, t_steps[i], t_steps[i + 1],
                frame_rate, model_condition_kwargs, eta,
            )

        if return_sample:
            return target_t, self.module.decode_frames(target_t.clone())
        return target_t
