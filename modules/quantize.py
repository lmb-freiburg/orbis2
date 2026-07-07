import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        normalize_embedding,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
        diversity_gamma=1.0,
        frac_per_sample_entropy=1.0,
        token_noise=0.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.normalize_embedding = normalize_embedding
        self.diversity_gamma = diversity_gamma
        self.frac_per_sample_entropy = frac_per_sample_entropy
        self.token_noise = token_noise
        self.sane_index_shape = sane_index_shape

        # Codebook
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        if self.normalize_embedding:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

        # Optional remapping
        self.remap = remap
        if remap is not None:
            self.register_buffer("used", torch.tensor(np.load(remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(
                f"Remapping {n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

    def remap_to_used(self, indices):
        ishape = indices.shape
        indices = indices.view(ishape[0], -1)
        used = self.used.to(indices)
        match = (indices[:, :, None] == used[None, None, :]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(indices.device)
        else:
            new[unknown] = self.unknown_index
        return new.view(ishape)

    def unmap_to_all(self, indices):
        ishape = indices.shape
        indices = indices.view(ishape[0], -1)
        used = self.used.to(indices)
        if self.re_embed > used.shape[0]:
            indices[indices >= used.shape[0]] = 0
        gathered = torch.gather(used.expand(indices.shape[0], -1), 1, indices)
        return gathered.view(ishape)

    def entropy_loss(self, distances, inv_temperature=100.0):
        prob = (-distances * inv_temperature).softmax(dim=-1)

        if self.frac_per_sample_entropy < 1.0:
            num_tokens = prob.shape[0]
            sample_size = int(num_tokens * self.frac_per_sample_entropy)
            mask = torch.randperm(num_tokens, device=prob.device)[:sample_size]
            per_sample_probs = prob[mask]
        else:
            per_sample_probs = prob

        per_sample_entropy = entropy(per_sample_probs).mean()
        avg_prob = reduce(per_sample_probs, "... d -> d", "mean")
        codebook_entropy = entropy(avg_prob).mean()

        return per_sample_entropy - self.diversity_gamma * codebook_entropy

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp in (None, 1.0)
        assert not rescale_logits and not return_logits

        if self.normalize_embedding:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

        # Flatten input
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flat = z.view(-1, self.e_dim)

        # Compute distances
        e = self.embedding.weight
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(e ** 2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flat, e.T)
        )

        min_indices = torch.argmin(d, dim=1)

        # Optional token noise
        if self.token_noise > 0.0 and self.training:
            noise_mask = torch.rand_like(min_indices.float()) < self.token_noise
            rand_indices = torch.randint(0, self.n_e, min_indices.shape, device=z.device)
            min_indices[noise_mask] = rand_indices[noise_mask]

        z_q = self.embedding(min_indices).view_as(z)

        # Compute VQ loss
        if self.legacy:
            loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        else:
            loss = self.beta * F.mse_loss(z_q.detach(), z) + F.mse_loss(z_q, z.detach())

        # Optional entropy loss
        entropy_aux = self.entropy_loss(d) if self.training else None

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Reshape to original
        z_q = rearrange(z_q, "b h w c -> b c h w")
        z = rearrange(z, "b h w c -> b c h w")

        # Remap if needed
        if self.remap is not None:
            min_indices = min_indices.view(z.shape[0], -1)
            min_indices = self.remap_to_used(min_indices).view(-1, 1)

        if self.sane_index_shape:
            min_indices = min_indices.view(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return {
            "quantized": z_q,
            "quantization_loss": loss,
            "entropy_loss": entropy_aux,
            "indices": min_indices,
        }

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.view(shape[0], -1)
            indices = self.unmap_to_all(indices).view(-1)

        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q
