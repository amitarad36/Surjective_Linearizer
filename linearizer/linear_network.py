from abc import abstractmethod

import torch
import math
import torch.nn as nn

from linearizer.base import LinearModule


class SinusoidalPosEmb(nn.Module):
    """Encodes a scalar timestep into a sinusoidal positional embedding vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Map a batch of scalar timesteps [B] to sinusoidal embeddings [B, dim]."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class OneStepLinearModule(LinearModule):
    """Abstract linear module that additionally exposes an explicit matrix form.

    Subclasses must implement get_lin_t(t), which returns the explicit A(t)
    matrix used to precompute the one-step sampling operator B.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_lin_t(self, t):
        """Return the explicit linear matrix A(t) for timestep t.

        Used offline to precompute the composed one-step sampling matrix B.
        """
        pass


class TimeDependentLoRALinearLayer(OneStepLinearModule):
    """Time-dependent low-rank linear operator A(t).

    Computes A(t) = A1(t) @ A2(t)^T where A1, A2 are produced by two
    independent time-conditioned MLPs. The resulting matrix has rank
    lora_features, making it parameter-efficient.

    Being linear in its input x means the entire flow-matching ODE can be
    collapsed into a single matrix multiplication at inference time.
    """

    def __init__(self, out_features, lora_features, t_size):
        super().__init__()
        time_dim = t_size * 4
        self.out_features = out_features
        self.lora_features = lora_features

        self.lin2 = nn.Sequential(
            SinusoidalPosEmb(t_size),
            nn.Linear(t_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, out_features * lora_features),
        )

        self.lin1 = nn.Sequential(
            SinusoidalPosEmb(t_size),
            nn.Linear(t_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, out_features * lora_features),
        )

    def forward(self, x, **kwargs):
        """Apply A(t) to x: x_pred = x @ A(t). Expects kwargs['t'] as the timestep."""
        x_shape = x.shape
        vector_x = x.reshape(x.shape[0], -1)
        A1 = self.lin1(kwargs['t']).reshape(vector_x.shape[0], self.out_features, self.lora_features)
        A2 = self.lin2(kwargs['t']).reshape(vector_x.shape[0], self.out_features, self.lora_features)
        A = torch.bmm(A1, A2.permute(0, 2, 1))
        x_pred = torch.bmm(vector_x.unsqueeze(1), A).squeeze(1)
        x_pred = x_pred.reshape(x_shape)
        return x_pred

    def get_lin_t(self, t):
        """Return the explicit [out_features, out_features] matrix A(t) for timestep t."""
        A1 = self.lin1(t).reshape(t.shape[0], self.out_features, self.lora_features)
        A2 = self.lin2(t).reshape(t.shape[0], self.out_features, self.lora_features)
        A = torch.bmm(A1, A2.permute(0, 2, 1))
        return A
