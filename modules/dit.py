"""
Reusable DiT building blocks needed for inference.
Extracted from networks/DiT/dit.py so the inference set avoids importing
the full training-only model classes in that file.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers.mlp import SwiGLU
from timm.models.vision_transformer import Attention, Mlp


def get_norm_layer(norm_layer):
    if isinstance(norm_layer, str):
        if norm_layer == 'layer_norm':
            return nn.LayerNorm
        elif norm_layer == 'rms_norm':
            return nn.RMSNorm
        else:
            raise ValueError(f"Unsupported norm layer: {norm_layer}")
    return norm_layer


def modulate(x, shift, scale):
    n = x.ndim - shift.ndim
    for _ in range(n):
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _broadcast_gate(gate, x):
    n = x.ndim - gate.ndim
    for _ in range(n):
        gate = gate.unsqueeze(-2)
    return gate


class FrequencyEncoder:
   def __init__(self, embed_dim, freq_min=1, freq_max=5):
       """
       Deterministic frequency encoder with fixed normalization.


       Args:
           embed_dim (int): Dimensionality of the token embeddings.
           freq_min (float): Minimum frequency value.
           freq_max (float): Maximum frequency value.
       """
       self.embed_dim = embed_dim
       self.freq_min = freq_min
       self.freq_max = freq_max


   def encode(self, frequencies):
       """
       Encodes frequencies into embeddings using sine-cosine features.


       Args:
           frequencies (torch.Tensor): Tensor of shape (batch_size,) containing frequencies.


       Returns:
           torch.Tensor: Encoded frequency embeddings of shape (batch_size, embed_dim).
       """
       batch_size = frequencies.size(0)


       # Fixed normalization: Scale frequencies to [0, 1]
       normalized_freq = (frequencies - self.freq_min) / (self.freq_max - self.freq_min)


       # Generate positional features using sine and cosine
       positions = torch.arange(0, self.embed_dim, dtype=torch.float32, device=frequencies.device)
       scaling_factors = 1 / (10000 ** (2 * (positions // 2) / self.embed_dim))
       frequency_features = normalized_freq.unsqueeze(1) * scaling_factors  # Shape: (batch_size, embed_dim)


       # Apply sine to even indices and cosine to odd indices
       encoded_freq = torch.zeros(batch_size, self.embed_dim, device=frequencies.device)
       encoded_freq[:, 0::2] = torch.sin(frequency_features[:, 0::2])  # Sine for even indices
       encoded_freq[:, 1::2] = torch.cos(frequency_features[:, 1::2])  # Cosine for odd indices


       return encoded_freq


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
   """
   grid_size: int of the grid height and width
   return:
   pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
   """
   grid_h = np.arange(grid_size[0], dtype=np.float32)
   grid_w = np.arange(grid_size[1], dtype=np.float32)
   grid = np.meshgrid(grid_w, grid_h)  # here w goes first
   grid = np.stack(grid, axis=0)


   grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
   pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
   if cls_token and extra_tokens > 0:
       pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
   return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
   assert embed_dim % 2 == 0


   # use half of dimensions to encode grid_h
   emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
   emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)


   emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
   return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
   """
   embed_dim: output dimension for each position
   pos: a list of positions to be encoded: size (M,)
   out: (M, D)
   """
   assert embed_dim % 2 == 0
   omega = np.arange(embed_dim // 2, dtype=np.float64)
   omega /= embed_dim / 2.
   omega = 1. / 10000**omega  # (D/2,)


   pos = pos.reshape(-1)  # (M,)
   out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product


   emb_sin = np.sin(out) # (M, D/2)
   emb_cos = np.cos(out) # (M, D/2)


   emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
   return emb


class TimestepEmbedder(nn.Module):
   """
   Embeds scalar timesteps into vector representations.
   """
   def __init__(self, hidden_size, frequency_embedding_size=256):
       super().__init__()
       self.mlp = nn.Sequential(
           nn.Linear(frequency_embedding_size, hidden_size, bias=True),
           nn.SiLU(),
           nn.Linear(hidden_size, hidden_size, bias=True),
       )
       self.frequency_embedding_size = frequency_embedding_size


   @staticmethod
   def timestep_embedding(t, dim, max_period=10000):
       """
       Create sinusoidal timestep embeddings.
       :param t: a 1-D Tensor of N indices, one per batch element.
                         These may be fractional.
       :param dim: the dimension of the output.
       :param max_period: controls the minimum frequency of the embeddings.
       :return: an (N, D) Tensor of positional embeddings.
       """
       # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
       half = dim // 2
       freqs = torch.exp(
           -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
       ).to(device=t.device)
       args = t[:, None].float() * freqs[None]
       embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
       if dim % 2:
           embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
       return embedding


   def forward(self, t):
       t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
       t_emb = self.mlp(t_freq)
       return t_emb


class STBlock(nn.Module):
   # Used for temporal compression in context
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0,
                norm_layer=nn.LayerNorm,
                mlp_block='mlp',
                act_layer=lambda: nn.GELU(approximate="tanh"),
                **block_kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                    attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=norm_layer, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                    attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=norm_layer, **block_kwargs)
        if mlp_block == 'mlp':
            self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act_layer, norm_layer=norm_layer, drop=0)
            self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act_layer, norm_layer=norm_layer, drop=0)
        elif mlp_block == 'swiglu':
            self.space_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
            self.time_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
        else:
            raise NotImplementedError(f"mlp_block {mlp_block} not implemented")

   def forward(self, x):
       B, F, N, D = x.shape
       x = rearrange(x, 'b f n d -> (b f) n d')
       x = x + self.space_attn(self.norm1(x))
       x = x + self.space_mlp(self.norm2(x))
       x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
       x = x + self.time_attn(self.norm3(x))
       x = x + self.time_mlp(self.norm4(x))
       x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
       return x


class DiTBlock(nn.Module):
   """
   A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
   """
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=norm_layer, drop=0)
        elif mlp_block == 'swiglu':
            self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

   def initialize_adaln_weights(self, gate_init_std=0.0):
       nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
       nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
       if gate_init_std != 0.0:
           hidden_size = self.adaLN_modulation[-1].out_features // 6
           for gate_idx in (2, 5):
               start = gate_idx * hidden_size
               end = start + hidden_size
               nn.init.normal_(self.adaLN_modulation[-1].weight[start:end], std=gate_init_std)

   def forward(self, x, c):
       shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
       x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
       x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
       return x


class CDiTBlock(nn.Module):
   """
   A DiT block with cross-attention conditioning.
   """
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=norm_layer, **block_kwargs)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=norm_layer, drop=0)
        elif mlp_block == 'swiglu':
            self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=True)

   def initialize_adaln_weights(self, gate_init_std=0.0):
       nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
       nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
       if gate_init_std != 0.0:
           hidden_size = self.adaLN_modulation[-1].out_features // 11
           for gate_idx in (2, 7, 10):
               start = gate_idx * hidden_size
               end = start + hidden_size
               nn.init.normal_(self.adaLN_modulation[-1].weight[start:end], std=gate_init_std)

   def forward(self, x, c, x_cond):
       shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
       x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
       x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
       x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
       x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
       return x


class STDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, causal_time_attn=False, modulate_time_attn=False, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
            self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
        elif mlp_block == 'swiglu':
            self.space_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
            self.time_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3)
        else:
            raise NotImplementedError(f"mlp_block {mlp_block} not implemented")

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.causal_time_attn = causal_time_attn
        self.modulate_time_attn = modulate_time_attn
        self.layer_idx = block_kwargs.get("layer_idx")
        self.log_adaln_mean_abs = False
        self._last_adaln_mean_abs = None

        if modulate_time_attn:
            self.adaLN_time_attn_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
            self.norm_time_attn = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
            # initialize
            nn.init.constant_(self.adaLN_time_attn_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_time_attn_modulation[-1].bias, 0)
        else:
            self.norm_time_attn = nn.Identity()

    def initialize_adaln_weights(self, gate_init_std=0.0):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        if gate_init_std != 0.0:
            hidden_size = self.adaLN_modulation[-1].out_features // 9
            for gate_idx in (2, 5, 8):
                start = gate_idx * hidden_size
                end = start + hidden_size
                nn.init.normal_(self.adaLN_modulation[-1].weight[start:end], std=gate_init_std)

        if hasattr(self, "adaLN_time_attn_modulation"):
            nn.init.constant_(self.adaLN_time_attn_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_time_attn_modulation[-1].bias, 0)
            if gate_init_std != 0.0:
                hidden_size = self.adaLN_time_attn_modulation[-1].out_features // 3
                start = 2 * hidden_size
                end = start + hidden_size
                nn.init.normal_(self.adaLN_time_attn_modulation[-1].weight[start:end], std=gate_init_std)

    def _collect_adaln_mean_abs(self, **tensors):
        self._last_adaln_mean_abs = {
            name: tensor.detach().abs().flatten(1).mean(dim=1)
            for name, tensor in tensors.items()
        }

    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
        shift_mlp_s, scale_mlp_s, gate_mlp_s,
        shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=-1)

        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
        x_ = self.space_attn(x_modulated)
        x_ = rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)
        x = x + _broadcast_gate(gate_msa, x) * x_

        x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
        x = x + _broadcast_gate(gate_mlp_s, x) * self.space_mlp(x_modulated)

        # — temporal attention path —
        if self.modulate_time_attn:
            shift_mta, scale_mta, gate_mta = self.adaLN_time_attn_modulation(c).chunk(3, dim=-1)
        else:
            shift_mta, scale_mta, gate_mta = torch.zeros_like(shift_mlp_t), torch.zeros_like(scale_mlp_t), torch.ones_like(gate_mlp_t)

        if self.log_adaln_mean_abs:
            self._collect_adaln_mean_abs(
                msa_shift=shift_msa,
                msa_scale=scale_msa,
                msa_gate=gate_msa,
                mlp_s_shift=shift_mlp_s,
                mlp_s_scale=scale_mlp_s,
                mlp_s_gate=gate_mlp_s,
                mta_shift=shift_mta,
                mta_scale=scale_mta,
                mta_gate=gate_mta,
                mlp_t_shift=shift_mlp_t,
                mlp_t_scale=scale_mlp_t,
                mlp_t_gate=gate_mlp_t,
            )

        x_modulated = modulate(self.norm_time_attn(x), shift_mta, scale_mta)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b n) f d', b=B, f=F, n=N)
        time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
        x_ = self.time_attn(x_modulated, attn_mask=time_attn_mask)
        x_ = rearrange(x_, '(b n) f d -> b f n d', b=B, n=N, f=F)
        x = x + _broadcast_gate(gate_mta, x) * x_

        x_modulated = modulate(self.norm3(x), shift_mlp_t, scale_mlp_t)
        x = x + _broadcast_gate(gate_mlp_t, x) * self.time_mlp(x_modulated)

        return x


class FinalLayer(nn.Module):
   """
   The final layer of DiT.
   """
   def __init__(self, hidden_size, patch_size, out_channels, norm_layer=nn.LayerNorm, act_layer=nn.SiLU):
       super().__init__()
       self.norm_final = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
       self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
       self.adaLN_modulation = nn.Sequential(
           act_layer(),
           nn.Linear(hidden_size, 2 * hidden_size, bias=True)
       )


   def forward(self, x, c):
       shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
       x = modulate(self.norm_final(x), shift, scale)
       x = self.linear(x)
       return x


class STDiTBlockWithSpatialL2(STDiTBlock):
    """
    STDiTBlock augmented with per-token gated spatial conditioning from L2 tokens.

    z_spatial (B, F, N, D) → (shift_s, scale_s, gate_s) per token via a small MLP.
    Applied as an additive gated adaLN before the standard block operations:
        x = x + gate_s * modulate(spatial_norm(x), shift_s, scale_s)

    Zero-init of spatial_to_mod guarantees a no-op at the start of training,
    so pre-trained STDiTDF weights can be fine-tuned without disruption.
    """

    def __init__(self, hidden_size, num_heads, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(hidden_size, num_heads, norm_layer=norm_layer, **kwargs)
        self.spatial_norm = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.spatial_to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        # Small weight init so shift/scale start as genuine L2-dependent perturbations.
        # Gate bias initialised to 2.0 → sigmoid(2.0) ≈ 0.88, so the gate starts open
        # and the model must learn to close it rather than having to learn to open it.
        nn.init.normal_(self.spatial_to_mod[-1].weight, std=0.02)
        nn.init.zeros_(self.spatial_to_mod[-1].bias)
        self.spatial_to_mod[-1].bias.data[2 * hidden_size:].fill_(2.0)

    def forward(self, x, c, z_spatial):
        # z_spatial: (B, F, N, D) — pre-computed L2 spatial tokens (patchified + position)
        shift_s, scale_s, gate_s = self.spatial_to_mod(z_spatial).chunk(3, dim=-1)
        x = x + torch.sigmoid(gate_s) * modulate(self.spatial_norm(x), shift_s, scale_s)
        return super().forward(x, c)
