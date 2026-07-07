import torch
import torch.nn as nn


def _make_scalar_embedder(hidden_size):
    embedder = nn.Sequential(
        nn.Linear(1, hidden_size, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size, bias=True),
    )
    nn.init.normal_(embedder[0].weight, std=0.02)
    nn.init.normal_(embedder[0].bias, std=0.02)
    nn.init.normal_(embedder[2].weight, std=0.02)
    nn.init.normal_(embedder[2].bias, std=0.02)
    return embedder


class LinearSteeringEmbedder(torch.nn.Module):
    """
    Generic linear embedding layer for steering signals. When steering is missing, outputs a fixed zero embedding.
    """
    def __init__(self, num_input_features, hidden_size, learnable_no_value_embedding=False):
        super().__init__()
        self.embedders = nn.ModuleList()
        self.register_buffer("no_value_embeddings", torch.zeros(num_input_features, hidden_size))
        if learnable_no_value_embedding:
            self.no_value_embeddings = nn.Parameter(torch.zeros(num_input_features, hidden_size))
        for _i in range(num_input_features):
            self.embedders.append(_make_scalar_embedder(hidden_size))

    def forward(self, steering):
        squeeze_k = False
        if steering.ndim == 2:
            steering = steering.unsqueeze(1)
            squeeze_k = True
        elif steering.ndim != 3:
            raise ValueError(f"Expected steering to have shape [B, D] or [B, K, D], got {tuple(steering.shape)}")

        _b, _k, num_features = steering.shape
        assert num_features == len(self.embedders), f"Expected {len(self.embedders)} features, but got {num_features}"
        embeddings = []
        for i in range(num_features):
            feature = steering[:, :, [i]]
            missing = torch.isnan(feature).squeeze(-1).unsqueeze(-1)
            embedding = self.embedders[i](torch.nan_to_num(feature, nan=0.0))
            if missing.any():
                embedding = torch.where(missing, self.no_value_embeddings[i].to(embedding.dtype).view(1, 1, -1), embedding)
            embeddings.append(embedding)
        steering_embedding = torch.stack(embeddings, dim=1).sum(dim=1)
        if squeeze_k:
            return steering_embedding[:, 0]
        return steering_embedding
