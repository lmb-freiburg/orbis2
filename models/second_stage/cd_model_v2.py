import os
from copy import deepcopy

import torch

from .fm_model_v2 import (
    FlowMatchingObjective,
    FlowMatchingSamplerTeacherForcing,
    PredictorModule,
    requires_grad,
    update_ema,
)

_DEFAULT_SAMPLE_GRIDS = {
    1: [0.99],
    2: [0.99, 0.8333],
    4: [0.99, 0.9375, 0.8333, 0.625],
}


def _endpoint(x, t, v):
    """G(x, t) = x + t * v -- endpoint (x0) prediction."""
    shape = [t.shape[0]] + [1] * (x.dim() - 1)
    return x + t.view(*shape) * v


class ConsistencyDistillationObjective(FlowMatchingObjective):
    """
    Causal consistency distillation loss for a teacher-forced flow-matching
    predictor: distills a frozen `teacher_vit` into a few-step student `vit`,
    whose EMA (`ema_vit`) is the deliverable.
    """

    def __init__(
        self,
        predictor_module,
        sigma_min=1e-5,
        cd_num_timesteps=48,
        cd_t_min=0.01,
        cd_t_max=0.99,
        cd_grid="uniform",
        cd_loss="l2",
        cd_pseudo_huber_c=0.00054,
    ):
        super().__init__(predictor_module, sigma_min=sigma_min)
        self.cd_loss = cd_loss
        self.cd_pseudo_huber_c = cd_pseudo_huber_c

        if cd_grid == "uniform":
            grid = torch.linspace(cd_t_min, cd_t_max, cd_num_timesteps)
        else:
            grid = torch.as_tensor(list(cd_grid), dtype=torch.float32)
            if grid.numel() != cd_num_timesteps:
                raise ValueError(
                    f"cd_grid has {grid.numel()} entries, expected "
                    f"cd_num_timesteps={cd_num_timesteps}."
                )
        predictor_module.register_buffer("cd_t_grid", grid, persistent=False)

    def _split_sequence(self, x):
        context = x[:, : -self.module.num_pred_frames]
        target = x[:, -self.module.num_pred_frames :]
        return context, target

    def _forward_v(self, net, context, target_t, t, frame_rate, model_condition_kwargs):
        sampler = self.module.sampler
        model_input = sampler._build_model_inputs(context, target_t, t)
        model_t = sampler._build_model_t(context, target_t, t)
        pred = net(model_input, t=model_t * sampler.timescale, frame_rate=frame_rate, **model_condition_kwargs)
        return sampler._extract_target_prediction(pred)

    def compute_loss(self, pred, target):
        diff = pred.float() - target.float()
        if self.cd_loss == "pseudo_huber":
            c = self.cd_pseudo_huber_c * (diff[0].numel() ** 0.5)
            return (diff.flatten(1).pow(2).sum(dim=1) + c * c).sqrt() - c
        return diff.pow(2)

    def compute_step(self, x, frame_rate, condition_kwargs=None):
        if getattr(self.module, "teacher_vit", None) is None:
            raise RuntimeError(
                "ConsistencyDistillationObjective requires the predictor module to "
                "have a `teacher_vit` (set teacher_ckpt_path on "
                "ConsistencyDistillPredictorModule)."
            )

        context, target = self._split_sequence(x)
        batch_size = target.shape[0]
        model_condition_kwargs = self.module.condition_preprocessor.get_model_condition_kwargs(condition_kwargs)

        grid = self.module.cd_t_grid
        n = torch.randint(1, grid.numel(), (batch_size,), device=x.device)
        t, t_prev = grid[n], grid[n - 1]

        noise = torch.randn_like(target)
        x_t, _ = self.add_noise(target, t, noise=noise)

        with torch.no_grad():
            v_teacher = self._forward_v(self.module.teacher_vit, context, x_t, t, frame_rate, model_condition_kwargs)
            step_shape = [t.shape[0]] + [1] * (x_t.dim() - 1)
            x_prev = x_t + (t - t_prev).view(*step_shape) * v_teacher
            v_target = self._forward_v(self.module.ema_vit, context, x_prev, t_prev, frame_rate, model_condition_kwargs)
            tgt = _endpoint(x_prev, t_prev, v_target)

        v_student = self._forward_v(self.module.vit, context, x_t, t, frame_rate, model_condition_kwargs)
        pred = _endpoint(x_t, t, v_student)

        return self.compute_loss(pred, tgt)


class ConsistencyDistillationSampler(FlowMatchingSamplerTeacherForcing):
    """
    Few-step consistency sampler: repeatedly renoises the current endpoint
    estimate and re-predicts, over a short descending grid of timesteps,
    instead of integrating a many-step Euler ODE.
    """

    def __init__(
        self,
        predictor_module,
        timescale=1.0,
        integration_t_eps=0.0,
        timestep_conditioning="global",
        sample_fresh_noise=True,
        sample_t_grids=None,
    ):
        super().__init__(
            predictor_module,
            timescale=timescale,
            integration_t_eps=integration_t_eps,
            timestep_conditioning=timestep_conditioning,
        )
        self.sample_fresh_noise = sample_fresh_noise
        self.sample_t_grids = dict(_DEFAULT_SAMPLE_GRIDS)
        if sample_t_grids is not None:
            self.sample_t_grids.update({int(k): list(v) for k, v in dict(sample_t_grids).items()})

    def _forward_v(self, net, context, target_t, t, frame_rate, model_condition_kwargs):
        model_input = self._build_model_inputs(context, target_t, t)
        model_t = self._build_model_t(context, target_t, t)
        pred = net(model_input, t=model_t * self.timescale, frame_rate=frame_rate, **model_condition_kwargs)
        return self._extract_target_prediction(pred)

    def _fewstep_grid(self, NFE):
        if NFE in self.sample_t_grids:
            return [float(v) for v in self.sample_t_grids[NFE]]
        t_max = float(self.module.cd_t_grid[-1])
        t_min = float(self.module.cd_t_grid[0])
        return torch.linspace(t_max, t_min, NFE + 1)[:-1].tolist()

    @torch.no_grad()
    def sample(
        self,
        images=None,
        latent=False,
        eta=0.0,
        NFE=2,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        condition_kwargs=None,
        return_sample=False,
    ):
        del eta  # accepted only so PredictorModule.sample/roll_out call sites keep working
        net = self._get_net(sample_with_ema)
        device = next(net.parameters()).device
        context = self._prepare_context(images, latent)

        condition_kwargs = self.module.condition_preprocessor.prepare_condition_kwargs(
            condition_kwargs,
            batch_size=num_samples,
            device=device,
            split="sample",
        )
        model_condition_kwargs = self.module.condition_preprocessor.get_model_condition_kwargs(condition_kwargs)

        if frame_rate is None:
            frame_rate = self._default_frame_rate(num_samples, device)

        input_h, input_w = self._get_input_hw()
        grid = self._fewstep_grid(NFE)

        eps0 = torch.randn(
            num_samples,
            self.module.num_pred_frames,
            self.module.vit.in_channels,
            input_h,
            input_w,
            device=device,
        )
        x = eps0
        x0_hat = None
        for k, t_k in enumerate(grid):
            t = torch.full((num_samples,), t_k, device=device)
            if k > 0:
                eps_k = torch.randn_like(x) if self.sample_fresh_noise else eps0
                x = (1.0 - t_k) * x0_hat + t_k * eps_k
            v = self._forward_v(net, context, x, t, frame_rate, model_condition_kwargs)
            x0_hat = _endpoint(x, t, v)

        if return_sample:
            return x0_hat, self.module.decode_frames(x0_hat.clone())
        return x0_hat


class ConsistencyDistillPredictorModule(PredictorModule):
    """
    PredictorModule specialization that distills a frozen teacher (loaded from
    `teacher_ckpt_path`) into a few-step student via consistency distillation.
    The deliverable is `ema_vit` (evaluate with sample_with_ema=True, NFE 1/2/4).
    """

    # `teacher_vit.*` is intentionally absent from saved checkpoints (see
    # on_save_checkpoint below) since the teacher is always reloaded fresh from
    # teacher_ckpt_path at __init__ time. Eval scripts that assert on
    # load_state_dict(...).missing_keys should exempt these prefixes.
    checkpoint_exempt_key_prefixes = ("teacher_vit.",)

    def __init__(
        self,
        *,
        teacher_ckpt_path,
        cd_ema_mu=0.999,
        cd_weight_decay=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cd_ema_mu = cd_ema_mu
        self.cd_weight_decay = cd_weight_decay
        self.strict_loading = False

        checkpoint_path = os.path.expandvars(teacher_ckpt_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"teacher_ckpt_path {checkpoint_path} does not exist.")

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        except (TypeError, RuntimeError):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]

        ema_state_dict = {
            key[len("ema_vit.") :]: value.clone()
            for key, value in state_dict.items()
            if key.startswith("ema_vit.")
        }
        del state_dict
        if not ema_state_dict:
            raise ValueError(f"No ema_vit.* keys found in teacher checkpoint {checkpoint_path}.")

        self.vit.load_state_dict(ema_state_dict, strict=True)
        self.ema_vit.load_state_dict(ema_state_dict, strict=True)
        self.teacher_vit = deepcopy(self.vit)
        requires_grad(self.teacher_vit, False)
        self.teacher_vit.eval()

    def setup(self, stage=None):
        super().setup(stage)
        if hasattr(self, "teacher_vit") and self.teacher_vit is not None:
            self.teacher_vit.requires_grad_(False)

    def configure_optimizers(self):
        params = [p for p in self.vit.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.cd_weight_decay)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not hasattr(self, "_ema_stream"):
            self._ema_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        if self._ema_stream is not None:
            with torch.cuda.stream(self._ema_stream):
                update_ema(self.ema_vit, self.vit, decay=self.cd_ema_mu)
        else:
            update_ema(self.ema_vit, self.vit, decay=self.cd_ema_mu)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        for key in [k for k in checkpoint["state_dict"] if k.startswith("teacher_vit.")]:
            del checkpoint["state_dict"][key]
