import inspect

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import ListConfig
from timm.models.vision_transformer import PatchEmbed

# Returns True when executing inside a torch.compile region (PyTorch >= 2.1).
# Guarding side-effect dict writes with this prevents graph breaks during compiled inference.
_is_compiling = getattr(torch.compiler, 'is_compiling', lambda: False)

from modules.dit import (
    FinalLayer,
    FrequencyEncoder,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_norm_layer,
)
from util import get_obj_from_str, instantiate_from_config


def _signature_accepts_kwargs(cls):
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(cls.__init__).parameters.values())


def _accepted_init_param_names(cls):
    names = set()
    for base_cls in inspect.getmro(cls):
        if base_cls is object:
            continue
        signature = inspect.signature(base_cls.__init__)
        for name, param in signature.parameters.items():
            if name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            names.add(name)
    return names


def _embed_timesteps_with(t, t_embedder):
    if t.ndim == 1:
        return t_embedder(t)
    if t.ndim == 2:
        b, f = t.shape
        c = t_embedder(rearrange(t, "b f -> (b f)"))
        return rearrange(c, "(b f) d -> b f d", b=b, f=f)
    raise ValueError(f"Unsupported timestep shape: {t.shape}")


class GlobalAdditiveConditioner(nn.Module):
    """
    Sum optional timestep embeddings and extra global condition embeddings.

    The DiT module continues to own `t_embedder` for checkpoint compatibility. Extra
    condition embedders are instantiated here and consume keyworded condition tensors.
    """

    def __init__(self, hidden_size, condition_configs=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.condition_embedders = nn.ModuleDict()

        condition_configs = condition_configs or {}
        for condition_name, embedder_config in condition_configs.items():
            self.condition_embedders[condition_name] = instantiate_from_config(embedder_config)

    def _reduce_global_condition_embedding(self, condition_name, embedding):
        if embedding.ndim == 2:
            return embedding
        if embedding.ndim == 3 and embedding.shape[1] == 1:
            return embedding[:, 0]
        raise ValueError(
            f"GlobalAdditiveConditioner only supports global condition embeddings with shape [B, H] "
            f"or [B, 1, H]. Condition `{condition_name}` produced {tuple(embedding.shape)}."
        )

    def forward(self, *, t=None, t_embedder=None, **condition_kwargs):
        assert (t is None) == (t_embedder is None)
        conditioning = 0 if t is None else _embed_timesteps_with(t, t_embedder)
        if not _is_compiling():
            self._last_condition_embeddings = {}
        for condition_name, embedder in self.condition_embedders.items():
            condition_embedding = embedder(condition_kwargs[condition_name])
            condition_embedding = self._reduce_global_condition_embedding(condition_name, condition_embedding)
            if not _is_compiling():
                self._last_condition_embeddings[condition_name] = condition_embedding.detach()
            conditioning = conditioning + condition_embedding
        return conditioning


class SequenceGlobalAdditiveConditioner(GlobalAdditiveConditioner):
    """
    Create a single global conditioning vector from sequence-structured conditions.

    Each condition embedder may return embeddings with shape [B, K, H]. 
    Optionally, a positional embedding can be added to each of the K embeddings before aggregation.
    This class aggregates across K before adding the result to the timestep embedding.
    """

    def __init__(self, hidden_size, condition_configs=None, aggregation="sum", add_positional_embedding=False, sequence_length=None):
        super().__init__(hidden_size=hidden_size, condition_configs=condition_configs)
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"Unsupported aggregation `{aggregation}`. Expected one of: sum, mean.")
        self.aggregation = aggregation
        self.add_positional_embedding = add_positional_embedding
        self.sequence_length = sequence_length
        if add_positional_embedding:
            if sequence_length is None: raise ValueError("sequence_length must be specified if add_positional_embedding is True.")
            # One learnable positional embedding table per condition_name.
            # Assumes self.condition_configs exists and is iterable/dict-like from parent.
            self.positional_embeddings = nn.ParameterDict()
            for condition_name in self.condition_embedders.keys():
                pe = nn.Parameter(torch.zeros(sequence_length, hidden_size))
                nn.init.normal_(pe, mean=0.0, std=0.02)
                self.positional_embeddings[condition_name] = pe
        else:
            self.positional_embeddings = None

    def _reduce_global_condition_embedding(self, condition_name, embedding):        
        if embedding.ndim == 2:
            return embedding
        if embedding.ndim != 3:
            raise ValueError(
                f"SequenceGlobalAdditiveConditioner expects condition `{condition_name}` to produce "
                f"shape [B, H] or [B, K, H], got {tuple(embedding.shape)}."
            )
        
        # Enforce positional embeddings for true sequences (K > 1)
        k = embedding.shape[1]
        if k > 1 and not self.add_positional_embedding:
            raise ValueError(
                f"Condition `{condition_name}` produced sequence embedding with K={k}, but "
                f"`add_positional_embedding=False`. Positional embedding must be enabled when K > 1."
            )

        if self.add_positional_embedding:
            embedding = self._add_positional_embeddings(condition_name, embedding)

        if self.aggregation == "sum":
            return embedding.sum(dim=1)
        return embedding.mean(dim=1)
    
    def _add_positional_embeddings(self, condition_name, embedding):
        """
        embedding: [B, K, H]
        returns:   [B, K, H] with learnable positional embeddings added
        """
        if embedding.ndim != 3:
            raise ValueError(
                f"_add_positional_embeddings expects [B, K, H] for `{condition_name}`, "
                f"got {tuple(embedding.shape)}."
            )

        b, k, h = embedding.shape

        if h != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch for condition `{condition_name}`: "
                f"embedding has H={h}, expected hidden_size={self.hidden_size}."
            )

        if self.positional_embeddings is None:
            raise RuntimeError(
                "Positional embedding is not initialized. "
                "Set add_positional_embedding=True in constructor."
            )

        if k > self.sequence_length:
            raise ValueError(
                f"Condition `{condition_name}` has sequence length K={k}, "
                f"but configured sequence_length={self.sequence_length}. "
                f"Increase sequence_length or truncate inputs."
            )

        # [K, H] -> [1, K, H], broadcast over batch
        pos = self.positional_embeddings[condition_name][:k].unsqueeze(0)
        pos = pos.to(device=embedding.device, dtype=embedding.dtype)
        return embedding + pos


class DiT(nn.Module):
    """
    Generic DiT backbone with configurable block construction and execution layout.

    The model does not distinguish target and context inputs. Forward accepts only a
    generic sequence of frames `x` with shape [B, F, C, H, W].
    """

    def __init__(
        self,
        input_size=16,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_num_frames=6,
        dropout=0.0,
        frequency_range=(2, 15),
        learn_sigma=False,
        norm_layer="layer_norm",
        mlp_block="mlp",
        block_config=None,
        conditioner_config=None,
        frame_embedding_alignment="suffix",
        ctx_noise_aug_ratio=0.0,
        ctx_noise_aug_prob=0.0,
        drop_ctx_rate=0.0,
        log_adaln_mean_abs=False,
        log_adaln_grad=False,
        adaln_gate_init_std=0.0,
        train_steering_adaln_only=False,
    ):
        super().__init__()
        self._validate_legacy_context_args(
            ctx_noise_aug_ratio=ctx_noise_aug_ratio,
            ctx_noise_aug_prob=ctx_noise_aug_prob,
            drop_ctx_rate=drop_ctx_rate,
        )

        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)

        self.input_size = input_size if isinstance(input_size, (list, tuple, ListConfig)) else [input_size, input_size]
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.max_num_frames = max_num_frames
        self.norm_layer = norm_layer
        self.mlp_block = mlp_block
        self.log_adaln_mean_abs = bool(log_adaln_mean_abs)
        self.log_adaln_grad = bool(log_adaln_grad)
        self.adaln_gate_init_std = float(adaln_gate_init_std)
        self._last_adaln_mean_abs = {}
        self.frame_embedding_alignment = self._normalize_frame_embedding_alignment(
            frame_embedding_alignment
        )

        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = self.x_embedder.num_patches
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.conditioner = self._build_conditioner(conditioner_config)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.frame_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.max_num_frames, 1, hidden_size)), 0.0, 0.02)
        self.frame_rate_encoder = FrequencyEncoder(hidden_size, freq_min=frequency_range[0], freq_max=frequency_range[1])

        if block_config is None:
            raise ValueError("DiT v2 expects block_config in instantiate_from_config format.")
        if "target" not in block_config:
            raise KeyError("block_config must define `target`.")

        self.block_config = block_config
        self.block_cls = get_obj_from_str(block_config["target"])
        self.blocks = self._build_blocks()
        self._configure_block_logging()

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_layer)
        self.initialize_weights()

        if train_steering_adaln_only:
            self._freeze_non_steering_adaln_parameters()

    def _freeze_non_steering_adaln_parameters(self):
        self.requires_grad_(False)
        self.t_embedder.requires_grad_(True)
        cond_embedders = getattr(self.conditioner, "condition_embedders", {})
        if "steering" in cond_embedders:
            cond_embedders["steering"].requires_grad_(True)
        for block in self.blocks:
            block.adaLN_modulation.requires_grad_(True)
            if hasattr(block, "adaLN_time_attn_modulation"):
                block.adaLN_time_attn_modulation.requires_grad_(True)
        self.final_layer.adaLN_modulation.requires_grad_(True)

    def _validate_legacy_context_args(self, *, ctx_noise_aug_ratio, ctx_noise_aug_prob, drop_ctx_rate):
        unsupported = {
            "ctx_noise_aug_ratio": ctx_noise_aug_ratio,
            "ctx_noise_aug_prob": ctx_noise_aug_prob,
            "drop_ctx_rate": drop_ctx_rate,
        }
        non_zero = {name: value for name, value in unsupported.items() if value not in (None, 0, 0.0)}
        if non_zero:
            raise ValueError(
                "DiT v2 does not implement context dropping or context noise augmentation. "
                f"Received: {non_zero}"
            )

    def _normalize_frame_embedding_alignment(self, frame_embedding_alignment):
        alignment = str(frame_embedding_alignment).strip().lower()
        if alignment not in {"suffix", "prefix"}:
            raise ValueError(
                "frame_embedding_alignment must be one of {'suffix', 'prefix'}, "
                f"got {frame_embedding_alignment!r}."
            )
        return alignment

    def _get_frame_embeddings(self, num_frames, device):
        if self.frame_embedding_alignment == "prefix":
            return self.frame_emb[:, :num_frames].to(device)
        return self.frame_emb[:, self.max_num_frames - num_frames :].to(device)

    def _get_common_block_kwargs(self, layer_idx):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout,
            "norm_layer": self.norm_layer,
            "mlp_block": self.mlp_block,
            "input_shape": self.input_size,
            "layer_idx": layer_idx,
        }

    def _build_single_block(self, layer_idx):
        accepts_kwargs = _signature_accepts_kwargs(self.block_cls)
        common_kwargs = self._get_common_block_kwargs(layer_idx)
        user_kwargs = dict(self.block_config.get("params", {}))

        valid_names = _accepted_init_param_names(self.block_cls)
        if not accepts_kwargs:
            unknown_user_kwargs = sorted(set(user_kwargs) - valid_names)
            if unknown_user_kwargs:
                raise ValueError(
                    f"Unsupported block parameters for {self.block_cls.__name__}: {unknown_user_kwargs}"
                )

        init_kwargs = {}
        for name, value in common_kwargs.items():
            if name in valid_names:
                init_kwargs[name] = value
        init_kwargs.update(user_kwargs)
        return instantiate_from_config(
            {
                "target": self.block_config["target"],
                "params": init_kwargs,
            }
        )

    def _build_blocks(self):
        return nn.ModuleList([self._build_single_block(layer_idx) for layer_idx in range(self.depth)])

    def _configure_block_logging(self):
        for block in self.blocks:
            if hasattr(block, "log_adaln_mean_abs"):
                block.log_adaln_mean_abs = self.log_adaln_mean_abs

    def _build_conditioner(self, conditioner_config):
        if conditioner_config is None:
            return None

        params = dict(conditioner_config.get("params", {}))
        params.setdefault("hidden_size", self.hidden_size)
        return instantiate_from_config(
            {
                "target": conditioner_config["target"],
                "params": params,
            }
        )

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            [self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size],
            cls_token=False,
            extra_tokens=0,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            if hasattr(block, "initialize_adaln_weights"):
                block.initialize_adaln_weights(self.adaln_gate_init_std)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = self.x_embedder.grid_size[0]
        w = self.x_embedder.grid_size[1]

        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
        x = torch.einsum("bfhwpqc->bfchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p, w * p))
        return imgs

    def postprocess_outputs(self, out):
        return self.unpatchify(out)

    def get_condition_embeddings(self, t, **_kwargs):
        if self.conditioner is None:
            return _embed_timesteps_with(t, self.t_embedder)
        return self.conditioner(t=t, t_embedder=self.t_embedder, **_kwargs)

    def _get_frame_rate_embeddings(self, frame_rate, batch_size, num_frames, device):
        if frame_rate is None:
            return torch.zeros(batch_size, 1, 1, self.hidden_size, device=device)

        frame_rate_embeddings = self.frame_rate_encoder.encode(frame_rate).to(device)
        if frame_rate_embeddings.ndim != 2 or frame_rate_embeddings.shape[0] != batch_size:
            raise ValueError(
                "frame_rate must have shape [B]. "
                f"Received embedding source with shape {tuple(frame_rate.shape)}"
            )
        return frame_rate_embeddings.unsqueeze(1).unsqueeze(1)

    def embed_inputs(self, x, frame_rate=None):
        if x.ndim != 5:
            raise ValueError(f"Expected x to have shape [B, F, C, H, W], got {tuple(x.shape)}")

        b, f = x.shape[:2]
        if f > self.max_num_frames:
            raise ValueError(f"Received {f} frames, but max_num_frames={self.max_num_frames}")

        frame_embeddings = self._get_frame_rate_embeddings(frame_rate, batch_size=b, num_frames=f, device=x.device)
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        x = rearrange(x, "(b f) hw c -> b f hw c", b=b, f=f)
        x = x + self._get_frame_embeddings(f, x.device) + frame_embeddings
        return x

    def run_blocks(self, x, c, return_features=False):
        features = []
        adaln_stats = {}
        for block_idx, block in enumerate(self.blocks):
            x = block(x, c)
            if self.log_adaln_mean_abs:
                block_stats = getattr(block, "_last_adaln_mean_abs", None)
                if block_stats:
                    adaln_stats.update(
                        {
                            f"block_{block_idx:02d}/{name}": value
                            for name, value in block_stats.items()
                        }
                    )
            if return_features:
                features.append(x)
        if self.log_adaln_mean_abs:
            self._last_adaln_mean_abs = adaln_stats
        return x, features

    def get_last_adaln_mean_abs(self):
        return self._last_adaln_mean_abs

    def get_last_condition_embeddings(self):
        if self.conditioner is None:
            return {}
        return dict(getattr(self.conditioner, '_last_condition_embeddings', {}))

    def _iter_conditioners(self):
        if self.conditioner is not None:
            yield ("", self.conditioner)

    def get_adaln_grad_stats(self):
        if not self.log_adaln_grad:
            return {}
        stats = {}
        for block_idx, block in enumerate(self.blocks):
            for mod_name in ("adaLN_modulation", "adaLN_time_attn_modulation", "adaLN_registers_modulation"):
                mod = getattr(block, mod_name, None)
                if mod is None:
                    continue
                linear = mod[-1]
                for param_name in ("weight", "bias"):
                    p = getattr(linear, param_name, None)
                    if p is None or p.grad is None:
                        continue
                    stats[f"block_{block_idx:02d}/{mod_name}/{param_name}"] = p.grad.detach()
        for cond_prefix, conditioner in self._iter_conditioners():
            for cond_name, embedder in conditioner.condition_embedders.items():
                for pname, p in embedder.named_parameters():
                    if p.grad is not None:
                        stats[f"{cond_prefix}cond_emb/{cond_name}/{pname}"] = p.grad.detach()
        return stats

    def forward(
        self,
        x,
        t,
        frame_rate=None,
        return_features=False,
        **condition_kwargs,
    ):
        c = self.get_condition_embeddings(t, **condition_kwargs)
        x = self.embed_inputs(x, frame_rate=frame_rate)
        x, features = self.run_blocks(x, c, return_features=return_features)
        out = self.final_layer(x, c)
        out = self.postprocess_outputs(out)
        if return_features:
            return out, features
        return out


class DiTWithRegisters(DiT):
    def __init__(self, registers_conditioner_config=None, train_steering_adaln_only=False, **kwargs):
        super().__init__(train_steering_adaln_only=False, **kwargs)
        self.registers_conditioner = self._build_conditioner(registers_conditioner_config)
        if self.registers_conditioner is not None:
            self.registers_conditioner.apply(self._initialize_linear_module)
        if train_steering_adaln_only:
            self._freeze_non_steering_adaln_parameters()

    @staticmethod
    def _initialize_linear_module(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _freeze_non_steering_adaln_parameters(self):
        self.requires_grad_(False)
        if self.registers_conditioner is not None:
            self.registers_conditioner.requires_grad_(True)
        for block in self.blocks:
            if hasattr(block, "adaLN_registers_modulation"):
                block.adaLN_registers_modulation.requires_grad_(True)

    def get_registers_condition_embeddings(self, **condition_kwargs):
        if self.registers_conditioner is None:
            return None
        return self.registers_conditioner(**condition_kwargs)

    def get_last_condition_embeddings(self):
        result = super().get_last_condition_embeddings()
        if self.registers_conditioner is not None:
            reg_embs = getattr(self.registers_conditioner, '_last_condition_embeddings', {})
            result.update({f"registers/{k}": v for k, v in reg_embs.items()})
        return result

    def _iter_conditioners(self):
        yield from super()._iter_conditioners()
        if self.registers_conditioner is not None:
            yield ("registers/", self.registers_conditioner)

    def run_blocks(self, x, c, c_registers=None, return_features=False):
        features = []
        adaln_stats = {}
        for block_idx, block in enumerate(self.blocks):
            if c_registers is None:
                x = block(x, c=c)
            else:
                x = block(x, c=c, c_registers=c_registers)
            if self.log_adaln_mean_abs:
                block_stats = getattr(block, "_last_adaln_mean_abs", None)
                if block_stats:
                    adaln_stats.update(
                        {
                            f"block_{block_idx:02d}/{name}": value
                            for name, value in block_stats.items()
                        }
                    )
            if return_features:
                features.append(x)
        if self.log_adaln_mean_abs:
            self._last_adaln_mean_abs = adaln_stats
        return x, features

    def forward(
        self,
        x,
        t,
        frame_rate=None,
        return_features=False,
        **condition_kwargs,
    ):
        c = self.get_condition_embeddings(t, **condition_kwargs)
        c_registers = self.get_registers_condition_embeddings(**condition_kwargs)
        x = self.embed_inputs(x, frame_rate=frame_rate)
        x, features = self.run_blocks(x, c, c_registers=c_registers, return_features=return_features)
        out = self.final_layer(x, c)
        out = self.postprocess_outputs(out)
        if return_features:
            return out, features
        return out


class SpatialL2CtxAndLastDiT(DiT):
    """
    v2 DiT equivalent of the v1 STDiTDFSpatialL2_CtxAndLast network.

    Spatial L2 endpoint latents are patchified and injected into blocks that
    accept `block(x, c, z_spatial)`. Only the last context frame and last frame
    receive non-zero L2 tokens; all frames receive a scalar position embedding.
    """

    def __init__(self, sem_in_channels, num_context_frames, frame_embedding_alignment, **kwargs):
        super().__init__(frame_embedding_alignment=frame_embedding_alignment, **kwargs)
        self.sem_in_channels = int(sem_in_channels)
        self.num_context_frames = int(num_context_frames)

        self.l2_patchify = nn.Linear(
            self.patch_size ** 2 * self.sem_in_channels,
            self.hidden_size,
            bias=True,
        )
        nn.init.xavier_uniform_(self.l2_patchify.weight)
        nn.init.zeros_(self.l2_patchify.bias)

        self.position_proj = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        for module in self.position_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _patchify_l2(self, z_l2):
        if z_l2.ndim != 5:
            raise ValueError(f"Expected z_l2 to have shape [B, F, C, H, W], got {tuple(z_l2.shape)}")
        b, f, c, h, w = z_l2.shape
        if c != self.sem_in_channels:
            raise ValueError(
                f"Expected L2 latents with {self.sem_in_channels} channels, got {c}."
            )
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"L2 latent spatial size {(h, w)} must be divisible by patch_size={self.patch_size}."
            )

        z = rearrange(
            z_l2,
            "b f c (h p1) (w p2) -> (b f) (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        z = self.l2_patchify(z)
        return rearrange(z, "(b f) n d -> b f n d", b=b, f=f)

    def _build_spatial_cond(self, z_l2_start, z_l2_end, num_frames):
        if z_l2_start is None or z_l2_end is None:
            raise ValueError("SpatialL2CtxAndLastDiT requires z_l2_start and z_l2_end.")
        if z_l2_start.shape != z_l2_end.shape:
            raise ValueError(
                "z_l2_start and z_l2_end must have matching shapes, got "
                f"{tuple(z_l2_start.shape)} and {tuple(z_l2_end.shape)}."
            )
        if self.num_context_frames < 1:
            raise ValueError("num_context_frames must be at least 1.")
        if self.num_context_frames > num_frames:
            raise ValueError(
                f"num_context_frames={self.num_context_frames} exceeds num_frames={num_frames}."
            )

        device = z_l2_start.device
        dtype = z_l2_start.dtype
        z_start_tok = self._patchify_l2(z_l2_start.unsqueeze(1))
        z_end_tok = self._patchify_l2(z_l2_end.unsqueeze(1))
        z_zero_tok = torch.zeros_like(z_start_tok)
        ctx_idx = self.num_context_frames - 1

        frames_cond = []
        for frame_idx in range(num_frames):
            pos_val = frame_idx / (num_frames - 1) if num_frames > 1 else 0.0
            pos = torch.tensor([[pos_val]], device=device, dtype=dtype)
            pos_emb = self.position_proj(pos)

            if frame_idx == ctx_idx:
                z_tok = z_start_tok
            elif frame_idx == num_frames - 1:
                z_tok = z_end_tok
            else:
                z_tok = z_zero_tok
            frames_cond.append(z_tok + pos_emb.unsqueeze(0))

        return torch.cat(frames_cond, dim=1)

    def run_spatial_l2_blocks(self, x, c, z_spatial, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x, c, z_spatial)
            if return_features:
                features.append(x)
        return x, features

    def forward(
        self,
        x,
        t,
        frame_rate=None,
        z_l2_start=None,
        z_l2_end=None,
        return_features=False,
        **condition_kwargs,
    ):
        if z_l2_start is None or z_l2_end is None:
            raise ValueError("SpatialL2CtxAndLastDiT requires z_l2_start and z_l2_end.")

        c = self.get_condition_embeddings(t, **condition_kwargs)
        x = self.embed_inputs(x, frame_rate=frame_rate)
        z_spatial = self._build_spatial_cond(z_l2_start, z_l2_end, x.size(1))
        x, features = self.run_spatial_l2_blocks(x, c, z_spatial, return_features=return_features)
        out = self.final_layer(x, c)
        out = self.postprocess_outputs(out)
        if return_features:
            return out, features
        return out
