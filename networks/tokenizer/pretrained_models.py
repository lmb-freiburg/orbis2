import torch
import timm
from torch import nn
from omegaconf import ListConfig
from einops import rearrange


from typing import Tuple, Union

class Encoder(nn.Module):
    def __init__(
        self, 
        resolution: Union[Tuple[int, int], int], 
        channels: int = 3, 
        pretrained_encoder = 'MAE',
        patch_size: int = 16,
        z_channels: int = 768,
        e_dim: int = 8,
        normalize_embedding: bool = True,
        use_pretrained_weights = True,
        # **ignore_kwargs
    ) -> None:
        # Initialize parent class with the first patch size
        super().__init__()
        self.image_size = resolution
        self.patch_size = patch_size
        self.channels = channels
        self.normalize_embedding = normalize_embedding
        self.z_channels = z_channels
        self.e_dim = e_dim
        self.pretrained_encoder = pretrained_encoder
        
        self.init_transformer(pretrained_encoder, use_pretrained_weights)

    def init_transformer(self, pretrained_encoder, use_pretrained_weights):
        if pretrained_encoder == 'VIT_DINO':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_encoder == 'VIT_DINOv2':
            pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_encoder == 'MAE_VIT_L':
            pretrained_encoder_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_encoder == 'VIT':
            pretrained_encoder_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_encoder == 'CLIP32':
            pretrained_encoder_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_encoder == 'CLIP':
            pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.openai'
        elif pretrained_encoder == 'base':
            pretrained_encoder_model = 'timm/vit_base_patch16_224'
        elif pretrained_encoder == 'large':
            pretrained_encoder_model = 'timm/vit_large_patch16_224'

        if pretrained_encoder == "VIT_DINOv3":
            self.encoder = torch.hub.load('../dinov3', 'dinov3_vitb16', source='local', weights='./pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth').train()
        else:
            self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()
            if use_pretrained_weights:
                pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
                """Initialize weights of target_model with weights from source_model."""
                with torch.no_grad():
                    for target_param, source_param in zip(self.encoder.parameters(), pretrained_model.parameters()):
                        target_param.data.copy_(source_param.data)

                # Clean up
                del pretrained_model
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        if self.pretrained_encoder == "VIT_DINOv3":
            h = self.encoder.forward_features(img)['x_norm_patchtokens']
        else:
            h = self.encoder.forward_features(img)[:,1:]
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h


class MaskedEncoder(Encoder):
    def __init__(
        self,
        *,
        mask_ratio_min: float = 0.0,
        mask_ratio_max: float = 0.5,
        mask_during_eval: bool = False,
        mask_token_init_std: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= mask_ratio_min <= mask_ratio_max <= 1.0:
            raise ValueError(
                f'Expected 0 <= mask_ratio_min <= mask_ratio_max <= 1, got '
                f'{mask_ratio_min}, {mask_ratio_max}.'
            )

        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.mask_during_eval = bool(mask_during_eval)
        embed_dim = getattr(self.encoder, 'embed_dim', self.z_channels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=mask_token_init_std)

    def _sample_patch_mask(self, batch_size: int, num_patches: int, device: torch.device):
        if self.mask_ratio_max <= 0.0:
            return None
        if not self.training and not self.mask_during_eval:
            return None

        ratios = torch.empty(batch_size, device=device).uniform_(self.mask_ratio_min, self.mask_ratio_max)
        return torch.rand(batch_size, num_patches, device=device) < ratios.unsqueeze(1)

    def _apply_mask_after_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Expected patch embeddings with shape (B, N, C), got {tuple(x.shape)}.')
        mask = self._sample_patch_mask(x.shape[0], x.shape[1], x.device)
        if mask is None:
            return x
        mask_token = self.mask_token.to(dtype=x.dtype).expand(x.shape[0], x.shape[1], -1)
        return torch.where(mask.unsqueeze(-1), mask_token, x)

    def _forward_timm_features(self, img: torch.FloatTensor) -> torch.Tensor:
        x = self.encoder.patch_embed(img)
        if x.dim() == 4:
            embed_dim = getattr(self.encoder, 'embed_dim', x.shape[-1])
            if x.shape[-1] == embed_dim:
                b, h, w, c = x.shape
                x = x.reshape(b, h * w, c)
                x = self._apply_mask_after_patch_embed(x)
                x = x.reshape(b, h, w, c)
            else:
                x = x.flatten(2).transpose(1, 2)
                x = self._apply_mask_after_patch_embed(x)
        else:
            x = self._apply_mask_after_patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)

        blocks = self.encoder.blocks
        if isinstance(blocks, nn.ModuleList):
            for blk in blocks:
                x = blk(x)
        else:
            x = blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        if self.pretrained_encoder == 'VIT_DINOv3':
            raise NotImplementedError('MaskedEncoder currently supports timm-based encoders only.')

        h = self._forward_timm_features(img)
        num_prefix_tokens = getattr(self.encoder, 'num_prefix_tokens', 1)
        if num_prefix_tokens > 0:
            h = h[:, num_prefix_tokens:]
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2) // self.patch_size, img.size(3) // self.patch_size)
        return h