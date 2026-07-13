# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import numpy as np


from timm.layers.mlp import SwiGLU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn


from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from omegaconf import ListConfig

from ..swin.swin_free_aspect_ratio import SwinTransformerBlock, SwinAttention



from modules.dit import (
    get_norm_layer, modulate, FrequencyEncoder,
    get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid, TimestepEmbedder,
    STBlock, DiTBlock, CDiTBlock,
    STDiTBlock, FinalLayer, STDiTBlockWithSpatialL2,
)


class Lambda(nn.Module):
   def __init__(self, func):
       super().__init__()
       self.func = func


   def forward(self, x):
       return self.func(x)
 
#################################################################################
#                                 Cross Attention                               #
#################################################################################


class CrossAttention(nn.Module):
   """
   Cross-attention mechanism that computes attention between target (x) and context (y).
   Args:
       dim (int): Input dimension.
       num_heads (int): Number of attention heads.
       qkv_bias (bool): Whether to include bias terms.
       attn_drop (float): Dropout rate for attention weights.
       proj_drop (float): Dropout rate after projection.
   """
   def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm):
       super(CrossAttention, self).__init__()
       self.num_heads = num_heads
       self.head_dim = dim // num_heads
       self.scale = self.head_dim ** -0.5


       self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
       self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
       self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
       self.proj = nn.Linear(dim, dim)
       self.attn_drop = nn.Dropout(attn_drop)
       self.proj_drop = nn.Dropout(proj_drop)
       self.qk_norm = qk_norm
       if qk_norm:
           self.q_norm = norm_layer(self.head_dim, eps=1e-6, elementwise_affine=False, bias=False)
           self.k_norm = norm_layer(self.head_dim, eps=1e-6, elementwise_affine=False, bias=False)


   def forward(self, x, y):
       B, N, C = x.shape
       B, M, _ = y.shape


       q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
       k = self.k_proj(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
       v = self.v_proj(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
       if self.qk_norm:
           q, k = self.q_norm(q), self.k_norm(k)
       attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
       attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
       return self.proj_drop(self.proj(attn_output))








#################################################################################
#                                 Core DiT Model                                #
#################################################################################



class STDiTBlock_tmpadaLN(nn.Module):
   """
   A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
   """
   def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, causal_time_attn=False, **block_kwargs):
       raise NotImplementedError("This is a deprecated version of STDiTBlock with temporary adaLN for time attention. Use STDiTBlock instead.")
       super().__init__()
       self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
       self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
       self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       mlp_hidden_dim = int(hidden_size * mlp_ratio)
       approx_gelu = lambda: nn.GELU(approximate="tanh")
       self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
       self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
       self.adaLN_modulation = nn.Sequential(
           nn.SiLU(),
           nn.Linear(hidden_size, 12 * hidden_size, bias=True)
       )
       self.causal_time_attn = causal_time_attn


   def forward(self, x, c):
       B, F, N, D = x.shape


       # chunk into 12 [B, C] vectors
       (shift_msa_s, scale_msa_s, gate_msa_s,
        shift_msa_t, scale_msa_t, gate_msa_t,
        shift_mlp_s, scale_mlp_s, gate_mlp_s,
        shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(12, dim=1)


       x_modulated = modulate(self.norm1(x), shift_msa_s, scale_msa_s)
       x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
       x_ = self.space_attn(x_modulated)
       x = x + gate_msa_s.unsqueeze(1).unsqueeze(1) * rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)


       x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
       x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(x_modulated)


       # — temporal attention path —
       x_modulated = modulate(self.norm3(x), shift_msa_t, scale_msa_t)
       x_modulated = rearrange(x_modulated, 'b f n d -> (b n) f d', b=B, f=F, n=N)
       time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
       x_ = self.time_attn(x_modulated, attn_mask=time_attn_mask)
       x = x + gate_msa_t.unsqueeze(1).unsqueeze(1) * rearrange(x_, '(b n) f d -> b f n d', b=B, f=F)
       x_modulated = modulate(self.norm4(x), shift_mlp_t, scale_mlp_t)
       x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated) 
       return x


class SwinSTDiTBlock(STDiTBlock):
    def __init__(self, hidden_size, num_heads, input_shape, layer_idx, mlp_ratio=4.0, window_size=[6, 4], dropout_rate=0.0, 
                 causal_time_attn=False, modulate_time_attn=False, 
                 norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate, 
                         causal_time_attn=causal_time_attn, modulate_time_attn=modulate_time_attn, mlp_block=mlp_block)
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = SwinTransformerBlock(
                hidden_size,
                input_resolution=input_shape,
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=(0,0) if (layer_idx % 2 == 0) else [ws//2 for ws in window_size],
                qk_norm=False,
                mlp_ratio=mlp_ratio,
                drop=dropout_rate,
                attn_drop=dropout_rate,
                norm_layer=norm_layer,
                mlp_block=mlp_block,
                **block_kwargs
                )


class SwinSTDiTBlockNoExtraMLP(STDiTBlock):
    def __init__(self, hidden_size, num_heads, input_shape, layer_idx, mlp_ratio=4.0, window_size=[6, 4], dropout_rate=0.0, norm_layer=nn.LayerNorm, mlp_block='mlp', **block_kwargs):
        super().__init__(hidden_size, num_heads, input_shape, layer_idx, mlp_ratio, dropout_rate, norm_layer=norm_layer, mlp_block=mlp_block, **block_kwargs)
        self.space_attn = SwinAttention(
            hidden_size,
            input_resolution=input_shape,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=(0,0) if (layer_idx % 2 == 0) else [ws//2 for ws in window_size],
            proj_drop=dropout_rate,
            attn_drop=dropout_rate,
            qk_norm=True,
            norm_layer=norm_layer,
            **block_kwargs,
        )



class STDiTBlockWithRegisters(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        causal_time_attn=False,
        modulate_time_attn=True,
        norm_layer=nn.LayerNorm,
        mlp_block='mlp',
        num_reg_tokens=1,   # must be > 0
        **block_kwargs
    ):
        super().__init__()
        if num_reg_tokens <= 0:
            raise ValueError("STDiTBlockWithRegisters now assumes num_reg_tokens > 0.")
        assert modulate_time_attn, "STDiTBlockWithRegisters currently requires modulate_time_attn=True."
        self.num_reg_tokens = num_reg_tokens

        # norms...
        self.norm1 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_reg_attn_s = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_reg_s = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_reg_attn_t = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_reg_t = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)

        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                    norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                   norm_layer=norm_layer, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if mlp_block == 'mlp':
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
            self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
            self.space_mlp_reg = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
            self.time_mlp_reg = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, norm_layer=None, drop=0)
        elif mlp_block == 'swiglu':
            self.space_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim * 2) // 3)
            self.time_mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim * 2) // 3)
            self.space_mlp_reg = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim * 2) // 3)
            self.time_mlp_reg = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim * 2) // 3)
        else:
            raise NotImplementedError(f"mlp_block {mlp_block} not implemented")

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))
        self.adaLN_registers_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True),
        )

        self.causal_time_attn = causal_time_attn
        self.modulate_time_attn = modulate_time_attn

        # Always-on time-attn modulation
        self.adaLN_time_attn_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.norm_time_attn = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.constant_(self.adaLN_time_attn_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_time_attn_modulation[-1].bias, 0)

        self.log_adaln_mean_abs = False
        self._last_adaln_mean_abs = {}

    def initialize_adaln_weights(self, gate_init_std=0.0):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_registers_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_registers_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_time_attn_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_time_attn_modulation[-1].bias, 0)

        if gate_init_std != 0.0:
            hidden_size = self.adaLN_modulation[-1].out_features // 9
            for gate_idx in (2, 5, 8):
                start = gate_idx * hidden_size
                end = start + hidden_size
                nn.init.normal_(self.adaLN_modulation[-1].weight[start:end], std=gate_init_std)

            hidden_size = self.adaLN_registers_modulation[-1].out_features // 12
            for gate_idx in (2, 5, 8, 11):
                start = gate_idx * hidden_size
                end = start + hidden_size
                nn.init.normal_(self.adaLN_registers_modulation[-1].weight[start:end], std=gate_init_std)

            hidden_size = self.adaLN_time_attn_modulation[-1].out_features // 3
            start = 2 * hidden_size
            end = start + hidden_size
            nn.init.normal_(self.adaLN_time_attn_modulation[-1].weight[start:end], std=gate_init_std)

    def _collect_adaln_mean_abs(self, **tensors):
        self._last_adaln_mean_abs = {
            name: tensor.detach().abs().flatten(1).mean(dim=1)
            for name, tensor in tensors.items()
        }

    def forward(self, x, c, c_registers):
        B, F, N, D = x.shape
        R = self.num_reg_tokens

        # always has registers
        reg_tokens = x[:, :, :R, :]
        x_patch = x[:, :, R:, :]

        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_mta, scale_mta, gate_mta = self.adaLN_time_attn_modulation(c).chunk(3, dim=-1)
        (shift_reg_attn_s, scale_reg_attn_s, gate_reg_attn_s,
         shift_reg_mlp_s, scale_reg_mlp_s, gate_reg_mlp_s,
         shift_reg_attn_t, scale_reg_attn_t, gate_reg_attn_t,
         shift_reg_mlp_t, scale_reg_mlp_t, gate_reg_mlp_t) = self.adaLN_registers_modulation(c_registers).chunk(12, dim=-1)

        if self.log_adaln_mean_abs:
            self._collect_adaln_mean_abs(
                msa_shift=shift_msa, msa_scale=scale_msa, msa_gate=gate_msa,
                mlp_s_shift=shift_mlp_s, mlp_s_scale=scale_mlp_s, mlp_s_gate=gate_mlp_s,
                mta_shift=shift_mta, mta_scale=scale_mta, mta_gate=gate_mta,
                mlp_t_shift=shift_mlp_t, mlp_t_scale=scale_mlp_t, mlp_t_gate=gate_mlp_t,
                reg_attn_s_shift=shift_reg_attn_s, reg_attn_s_scale=scale_reg_attn_s, reg_attn_s_gate=gate_reg_attn_s,
                reg_mlp_s_shift=shift_reg_mlp_s, reg_mlp_s_scale=scale_reg_mlp_s, reg_mlp_s_gate=gate_reg_mlp_s,
                reg_attn_t_shift=shift_reg_attn_t, reg_attn_t_scale=scale_reg_attn_t, reg_attn_t_gate=gate_reg_attn_t,
                reg_mlp_t_shift=shift_reg_mlp_t, reg_mlp_t_scale=scale_reg_mlp_t, reg_mlp_t_gate=gate_reg_mlp_t,
            )
        
        # spatial attn (always concat regs)
        x_mod = rearrange(modulate(self.norm1(x_patch), shift_msa, scale_msa), 'b f n d -> (b f) n d')
        reg_in = rearrange(
            modulate(self.norm_reg_attn_s(reg_tokens), shift_reg_attn_s, scale_reg_attn_s),
            'b f r d -> (b f) r d'
        )
        x_mod = torch.cat([reg_in, x_mod], dim=1)
        x_out = self.space_attn(x_mod)

        reg_attn_out, x_out = torch.split(x_out, [R, x_patch.size(2)], dim=1)
        reg_attn_out = rearrange(reg_attn_out, '(b f) r d -> b f r d', b=B, f=F)
        x_out = rearrange(x_out, '(b f) n d -> b f n d', b=B, f=F)
        x_patch = x_patch + gate_msa.unsqueeze(1).unsqueeze(1) * x_out if gate_msa.ndim == 2 else x_patch + gate_msa.unsqueeze(2) * x_out
        reg_tokens = reg_tokens + gate_reg_attn_s.unsqueeze(1).unsqueeze(1) * reg_attn_out

        # spatial mlp
        x_patch = x_patch + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(modulate(self.norm2(x_patch), shift_mlp_s, scale_mlp_s)) if gate_mlp_s.ndim == 2 else x_patch + gate_mlp_s.unsqueeze(2) * self.space_mlp(modulate(self.norm2(x_patch), shift_mlp_s, scale_mlp_s))
        reg_mlp_out = self.space_mlp_reg(modulate(self.norm_reg_s(reg_tokens), shift_reg_mlp_s, scale_reg_mlp_s))
        reg_tokens = reg_tokens + gate_reg_mlp_s.unsqueeze(1).unsqueeze(1) * reg_mlp_out

        # temporal attn patches
        x_mod = modulate(self.norm_time_attn(x_patch), shift_mta, scale_mta)
        x_mod = rearrange(x_mod, 'b f n d -> (b n) f d')
        time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
        x_out = self.time_attn(x_mod, attn_mask=time_attn_mask)
        x_out = rearrange(x_out, '(b n) f d -> b f n d', b=B, n=x_patch.size(2), f=F)
        x_patch = x_patch + gate_mta.unsqueeze(1).unsqueeze(1) * x_out if gate_mta.ndim == 2 else x_patch + gate_mta.unsqueeze(2) * x_out

        # temporal attn regs (always)
        reg_time_in = rearrange(
            modulate(self.norm_reg_attn_t(reg_tokens), shift_reg_attn_t, scale_reg_attn_t),
            'b f r d -> (b r) f d'
        )
        reg_time_out = self.time_attn(reg_time_in, attn_mask=time_attn_mask)
        reg_time_out = rearrange(reg_time_out, '(b r) f d -> b f r d', b=B, r=R, f=F)
        reg_tokens = reg_tokens + gate_reg_attn_t.unsqueeze(1).unsqueeze(1) * reg_time_out

        # temporal mlp
        x_patch = x_patch + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(modulate(self.norm3(x_patch), shift_mlp_t, scale_mlp_t)) if gate_mlp_t.ndim == 2 else x_patch + gate_mlp_t.unsqueeze(2) * self.time_mlp(modulate(self.norm3(x_patch), shift_mlp_t, scale_mlp_t))
        reg_mlp_out = self.time_mlp_reg(modulate(self.norm_reg_t(reg_tokens), shift_reg_mlp_t, scale_reg_mlp_t))
        reg_tokens = reg_tokens + gate_reg_mlp_t.unsqueeze(1).unsqueeze(1) * reg_mlp_out
        
        return torch.cat([reg_tokens, x_patch], dim=2)





class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
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
        ctx_noise_aug_ratio=0.1,
        ctx_noise_aug_prob = 0.5,
        drop_ctx_rate=0.2,
        frequency_range=(2, 15),
        learn_sigma=False,
        norm_layer=nn.LayerNorm,
        mlp_block='mlp',
    ):
        super().__init__()
        self.input_size= input_size if isinstance(input_size, (list, tuple, ListConfig)) else [input_size, input_size]
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.ctx_noise_aug_ratio = ctx_noise_aug_ratio
        self.ctx_noise_aug_prob = ctx_noise_aug_prob
        self.drop_ctx_rate = drop_ctx_rate


        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = self.x_embedder.num_patches
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
        ])       
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_layer)
        self.max_num_frames = max_num_frames
        self.frame_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.max_num_frames, 1, hidden_size)), 0., 0.02)
        self.frame_rate_encoder = FrequencyEncoder(hidden_size, freq_min=frequency_range[0], freq_max=frequency_range[1])

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size], cls_token=False, extra_tokens=0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = self.x_embedder.grid_size[0]
        w = self.x_embedder.grid_size[1]


        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
        x = torch.einsum('bfhwpqc->bfchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p, w * p))
        return imgs


    def get_condition_embeddings(self, t):
        """
        Get the condition embeddings for the given timesteps.
        t: (N,) tensor of diffusion timesteps
        returns: (N, D) tensor of condition embeddings
        """
        return self.t_embedder(t)
    
    def preprocess_inputs(self, target, context, t, frame_rate):
        b, f_target = target.size()[:2]
        f_context = context.size(1)
        
        if self.training:
            # Drop the context frame
            if torch.rand(1, device=target.device)<self.drop_ctx_rate:
                context = None
                f_context = 0
            elif torch.rand(1, device=target.device) < self.ctx_noise_aug_prob:
                # Add noise to context frames (if t is less than ctx_noise_aug_ratio, we do not add noise)
                mask = (t >= self.ctx_noise_aug_ratio)
                aug_noise = torch.randn_like(context)
                context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio


        frame_embeddings = self.frame_rate_encoder.encode(frame_rate)
        frame_embeddings = frame_embeddings.unsqueeze(1).unsqueeze(1).to(target.device)
        x = torch.cat((context, target), dim=1) if context is not None else target
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
        x = x + self.frame_emb[:, self.max_num_frames-(f_target+f_context):].to(x.device) + frame_embeddings
        return x

    def get_condition_embeddings(self, t):
        """
        Get the condition embeddings for the given timesteps.
        t: (N,) tensor of diffusion timesteps
        returns: (N, D) tensor of condition embeddings
        """
        return self.t_embedder(t)
    
    def preprocess_inputs(self, target, context, t, frame_rate):
        b, f_target = target.size()[:2]
        f_context = context.size(1) if context is not None else 0
        
        if self.training:
            # Drop the context frame
            if torch.rand(1, device=target.device)<self.drop_ctx_rate:
                context = None
                f_context = 0
            elif torch.rand(1, device=target.device) < self.ctx_noise_aug_prob:
                # Add noise to context frames (if t is less than ctx_noise_aug_ratio, we do not add noise)
                mask = (t >= self.ctx_noise_aug_ratio)
                aug_noise = torch.randn_like(context)
                context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio


        frame_embeddings = self.frame_rate_encoder.encode(frame_rate)
        frame_embeddings = frame_embeddings.unsqueeze(1).unsqueeze(1).to(target.device)
        x = torch.cat((context, target), dim=1) if context is not None else target
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
        x = x + self.frame_emb[:, self.max_num_frames-(f_target+f_context):].to(x.device) + frame_embeddings
        return x

    def postprocess_outputs(self, out):
        return self.unpatchify(out)

    def forward(self, target, context, t, frame_rate, return_features=False):
            """
            Forward pass of DiT.
            x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
            t: (N,) tensor of diffusion timesteps
            y: (N,) tensor of class labels
            """
            
            num_frames_ctx = context.size(1)
            num_frames_pred = target.size(1)
            
            c = self.get_condition_embeddings(t)
            
            x = self.preprocess_inputs(target, context, t, frame_rate)
            
            x = rearrange(x,  'b f hw c -> b (f hw) c')
            features = []
            for block in self.blocks:
                x = block(x, c)
                features.append(x) if return_features else None
            x = rearrange(x,  'b (f hw) c -> b f hw c', f=(num_frames_ctx+num_frames_pred))[:,-num_frames_pred:]
            out = self.final_layer(x, c)
                        
            out = self.postprocess_outputs(out)
            if return_features:
                return out, features
            return out


class CDiT(DiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, norm_layer=nn.LayerNorm, mlp_block='mlp', **kwargs):
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, 
                         ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        self.blocks = nn.ModuleList([
                CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
            ])


    def forward(self, target, context, t, frame_rate, return_features=False):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        num_frames_ctx = context.size(1)
        num_frames_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)                   # (N, D)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)  # (B, F, N, D)


        target = rearrange(x[:,-num_frames_pred:],  'b f hw c -> b (f hw) c')
        ctx = rearrange(x[:,:-num_frames_pred],  'b f hw c -> b (f hw) c') if num_frames_ctx>1 else None

        features = []
        for block in self.blocks:
            target = block(target, c, ctx)                      # (N, T, D)
            features.append(target)

        target = rearrange(target,  'b (f hw) c -> b f hw c', f=(num_frames_pred))
        out = self.final_layer(target, c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        if return_features:
            return out, features
        return out



class STDiT(DiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, 
                 dropout=0.1, ctx_noise_aug_ratio=0.1, ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), 
                 causal_time_attn=False, modulate_time_attn=False, norm_layer=nn.LayerNorm, mlp_block='mlp',
                 **kwargs):
        
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
            
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                         max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, 
                         norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        self.blocks = nn.ModuleList([
                STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, 
                           causal_time_attn=causal_time_attn, modulate_time_attn=modulate_time_attn, 
                           norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
            ])

    def forward(self, target, context, t, frame_rate, return_features=False):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        f_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)                   # (N, D)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)  # (B, F, N, D)

        features = []
        for block in self.blocks:
            x = block(x, c)
            features.append(x) if return_features else None


        out = self.final_layer(x[:,-f_pred:], c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        if return_features:
            return out, features
        return out

class STDiTDF(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, 
                 dropout=0.1, ctx_noise_aug_ratio=0.1, ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), 
                 causal_time_attn=False, modulate_time_attn=False, norm_layer=nn.LayerNorm, mlp_block='mlp',
                 **kwargs):
        
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
            
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                         max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, 
                         frequency_range=frequency_range, causal_time_attn=causal_time_attn, modulate_time_attn=modulate_time_attn, norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)

        self.blocks = nn.ModuleList([
            STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, 
                        causal_time_attn=causal_time_attn, modulate_time_attn=modulate_time_attn, 
                        norm_layer=norm_layer, mlp_block=mlp_block) for _ in range(depth)
        ])
        
    def preprocess_inputs(self, x, frame_rate):
        b, f = x.size()[:2]

        frame_embeddings = self.frame_rate_encoder.encode(frame_rate)
        frame_embeddings = frame_embeddings.unsqueeze(1).unsqueeze(1).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
        x = x + self.frame_emb[:, :f].to(x.device) + frame_embeddings
        return x

    def forward(self, x, t, frame_rate, return_features=False):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        f_pred = x.size(1)
        
        t = rearrange(t, 'b f -> (b f)')
        c = self.get_condition_embeddings(t)                   # (N, D)
        c = rearrange(c, '(b f) d -> b f d', b=x.size(0), f=x.size(1))
        
        x = self.preprocess_inputs(x, frame_rate)  # (B, F, N, D)
        
        features = []
        for block in self.blocks:
            x = block(x, c)
            features.append(x) if return_features else None

        out = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        if return_features:
            return out, features
        return out
    
    
class STDiTWithGlobalBlocks(STDiT):
    """
    A DiT with extra global attention blocks (DiT blocks).
    """
    def __init__(self, *args, global_block_indices, **kwargs):
        
        class DiTBlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, x, c):
                B, F, N, D = x.shape
                x_ = rearrange(x, 'b f n d -> b (f n) d', b=B, f=F)
                x_ = self.block(x_, c)
                x_ = rearrange(x_, 'b (f n) d -> b f n d', b=B, f=F)
                return x_
        
        super().__init__(*args, **kwargs)
        self.global_block_indices = global_block_indices
        for idx in global_block_indices:
            dit_block = DiTBlock(hidden_size=kwargs.get('hidden_size',1152), num_heads=kwargs.get('num_heads',16),
                                 mlp_ratio=kwargs.get('mlp_ratio',4.0), dropout_rate=kwargs.get('dropout',0.0), 
                                 norm_layer=kwargs.get('norm_layer',nn.LayerNorm), mlp_block=kwargs.get('mlp_block','mlp'))
            self.blocks[idx] = DiTBlockWrapper(dit_block)
            
            
class STDiT_tmpadaLN(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.1, ctx_noise_aug_ratio=0.1, ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), causal_time_attn=False, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, norm_layer=norm_layer, **kwargs)
        self.blocks = nn.ModuleList([
                STDiTBlock_tmpadaLN(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, causal_time_attn=causal_time_attn, norm_layer=norm_layer) for _ in range(depth)
            ])


class SwinSTDiT(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, window_size=[6, 4], dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), norm_layer=nn.LayerNorm, mlp_block='mlp', **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        self.blocks = nn.ModuleList([
                SwinSTDiTBlock(hidden_size=hidden_size, num_heads=num_heads, input_shape=input_size, layer_idx=layer_idx, mlp_ratio=mlp_ratio, window_size=window_size, dropout_rate=dropout, norm_layer=norm_layer) for layer_idx in range(depth)
            ])


class SwinSTDiTNoExtraMLP(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, window_size=[6, 4], 
                 dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), 
                 norm_layer='layer_norm', mlp_block='mlp', qk_norm=True, **kwargs):
        
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
            
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, 
                         num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, 
                         ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, 
                         norm_layer=norm_layer, mlp_block=mlp_block, **kwargs)
        
        for block_idx, block in enumerate(self.blocks):
            block.space_attn = SwinAttention(
                hidden_size,
                input_resolution=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0) if (block_idx % 2 == 0) else [ws//2 for ws in window_size],
                proj_drop=dropout,
                attn_drop=dropout,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
            )



class STDiTDFSpatialL2(STDiTDF):
    """
    STDiTDF with per-token spatial conditioning from L2 (semantic) latents.

    The L2 latent (same spatial H×W as the rec latent, sem_in_channels channels) is
    patchified into tokens, combined with a learned frame-position embedding, and
    produces per-token (shift, scale, gate) modulation injected at every block via
    STDiTBlockWithSpatialL2.

    Conditioning scheme (num_frames = 1 context + num_pred_frames generated):
        Frame 0 (input):     z_l2_start, position = 0.0
        Frame k (generated): z_l2_end,   position = k / num_pred_frames

    All new parameters (l2_patchify, position_proj, per-block spatial_to_mod) are
    zero-initialized so that a pre-trained STDiTDF backbone is unchanged at init.

    Args:
        sem_in_channels: Number of channels in the L2 (sem) latent.
        All other args are passed through to STDiTDF.
    """

    def __init__(
        self,
        sem_in_channels,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_num_frames=6,
        dropout=0.1,
        ctx_noise_aug_ratio=0.1,
        ctx_noise_aug_prob=0.5,
        drop_ctx_rate=0.2,
        frequency_range=(2, 15),
        causal_time_attn=False,
        modulate_time_attn=False,
        norm_layer=nn.LayerNorm,
        mlp_block='mlp',
        **kwargs,
    ):
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)

        super().__init__(
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_num_frames=max_num_frames,
            dropout=dropout,
            ctx_noise_aug_ratio=ctx_noise_aug_ratio,
            ctx_noise_aug_prob=ctx_noise_aug_prob,
            drop_ctx_rate=drop_ctx_rate,
            frequency_range=frequency_range,
            causal_time_attn=causal_time_attn,
            modulate_time_attn=modulate_time_attn,
            norm_layer=norm_layer,
            mlp_block=mlp_block,
            **kwargs,
        )

        # Replace blocks with spatially-conditioned variants
        self.blocks = nn.ModuleList([
            STDiTBlockWithSpatialL2(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout,
                causal_time_attn=causal_time_attn,
                modulate_time_attn=modulate_time_attn,
                norm_layer=norm_layer,
                mlp_block=mlp_block,
            )
            for _ in range(depth)
        ])

        # Patchify L2 latent: patch_size^2 * sem_in_channels → hidden_size
        # Xavier init so the patchified tokens carry real signal from step 1.
        self.l2_patchify = nn.Linear(patch_size ** 2 * sem_in_channels, hidden_size, bias=True)
        nn.init.xavier_uniform_(self.l2_patchify.weight)
        nn.init.zeros_(self.l2_patchify.bias)

        # Frame-position embedding: scalar in [0,1] → hidden_size
        # Xavier init so position differences produce non-trivial embeddings from step 1.
        self.position_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for m in self.position_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _patchify_l2(self, z_l2):
        """
        Patchify L2 latent frames.

        Args:
            z_l2: (B, F, C_sem, H, W)
        Returns:
            (B, F, N, hidden_size)  where N = (H/patch_size) * (W/patch_size)
        """
        b, f, c, h, w = z_l2.shape
        p = self.patch_size
        z = rearrange(z_l2, 'b f c (h p1) (w p2) -> (b f) (h w) (p1 p2 c)', p1=p, p2=p)
        z = self.l2_patchify(z)                          # (B*F, N, hidden_size)
        return rearrange(z, '(b f) n d -> b f n d', b=b, f=f)

    def _build_spatial_cond(self, z_l2_start, z_l2_end, num_frames):
        """
        Build per-frame L2 spatial tokens with position embedding.

        Args:
            z_l2_start: (B, C_sem, H, W) — sem latent for frame 0 (context/input)
            z_l2_end:   (B, C_sem, H, W) — sem latent for the last frame (target end)
            num_frames: total frames in x  (= 1 context + num_pred_frames generated)
        Returns:
            (B, num_frames, N, hidden_size)
        """
        device = z_l2_start.device
        dtype  = z_l2_start.dtype
        num_pred = num_frames - 1  # generated frames indexed 1..num_frames-1

        z_start_tok = self._patchify_l2(z_l2_start.unsqueeze(1))  # (B, 1, N, D)
        z_end_tok   = self._patchify_l2(z_l2_end.unsqueeze(1))    # (B, 1, N, D)

        frames_cond = []
        for i in range(num_frames):
            if i == 0:
                pos_val = 0.0
                z_tok = z_start_tok
            else:
                pos_val = i / num_pred if num_pred > 0 else 1.0
                z_tok = z_end_tok

            pos = torch.tensor([[pos_val]], device=device, dtype=dtype)
            pos_emb = self.position_proj(pos)                    # (1, D)
            # Broadcast over batch and spatial tokens
            z_frame = z_tok + pos_emb.unsqueeze(0)              # (B, 1, N, D)
            frames_cond.append(z_frame)

        return torch.cat(frames_cond, dim=1)                     # (B, F, N, D)

    def forward(self, x, t, frame_rate, z_l2_start, z_l2_end, return_features=False):
        """
        Args:
            x:          (B, F, C_rec, H, W) — rec latent (DF style: context + targets)
            t:          (B, F) — per-frame diffusion timesteps
            frame_rate: (B,)
            z_l2_start: (B, C_sem, H, W) — clean L2 latent for frame 0
            z_l2_end:   (B, C_sem, H, W) — clean L2 latent for last frame
        """
        b, f = x.shape[:2]

        t_flat = rearrange(t, 'b f -> (b f)')
        c = self.get_condition_embeddings(t_flat)            # (B*F, D)
        c = rearrange(c, '(b f) d -> b f d', b=b, f=f)     # (B, F, D)

        x = self.preprocess_inputs(x, frame_rate)            # (B, F, N, D)

        z_spatial = self._build_spatial_cond(z_l2_start, z_l2_end, f)  # (B, F, N, D)

        features = []
        for block in self.blocks:
            x = block(x, c, z_spatial)
            features.append(x) if return_features else None

        out = self.final_layer(x, c)
        out = self.postprocess_outputs(out)
        if return_features:
            return out, features
        return out


class STDiTDFSpatialL2_CtxAndLast(STDiTDFSpatialL2):
    """
    Two-point L2 conditioning: only the last context frame and the last
    predicted frame receive a non-zero L2 conditioning token.

    Conditioning scheme (K = num_context_frames, F = num_frames):
        Frame K-1 (last context frame):   z_l2_start, position = (K-1)/(F-1)
        Frame F-1 (last predicted frame): z_l2_end,   position = 1.0
        All other frames:                 zero token,  position = i/(F-1)

    The zero-token frames still receive a position embedding so the network can
    distinguish their temporal location, but carry no semantic conditioning.

    Args:
        num_context_frames: number of context frames in the L1 sequence (K).
            Must match the NUM_CONTEXT_FRAMES of the model class.
        All other args forwarded to STDiTDFSpatialL2.
    """

    def __init__(self, num_context_frames, **kwargs):
        super().__init__(**kwargs)
        self.num_context_frames = int(num_context_frames)

    def _build_spatial_cond(self, z_l2_start, z_l2_end, num_frames):
        device = z_l2_start.device
        dtype  = z_l2_start.dtype

        z_start_tok = self._patchify_l2(z_l2_start.unsqueeze(1))  # (B, 1, N, D)
        z_end_tok   = self._patchify_l2(z_l2_end.unsqueeze(1))    # (B, 1, N, D)
        z_zero_tok  = torch.zeros_like(z_start_tok)                # (B, 1, N, D)

        ctx_idx = self.num_context_frames - 1

        frames_cond = []
        for i in range(num_frames):
            pos_val = i / (num_frames - 1) if num_frames > 1 else 0.0
            pos     = torch.tensor([[pos_val]], device=device, dtype=dtype)
            pos_emb = self.position_proj(pos)                      # (1, D)

            if i == ctx_idx:
                z_tok = z_start_tok
            elif i == num_frames - 1:
                z_tok = z_end_tok
            else:
                z_tok = z_zero_tok

            frames_cond.append(z_tok + pos_emb.unsqueeze(0))      # (B, 1, N, D)

        return torch.cat(frames_cond, dim=1)                       # (B, F, N, D)
