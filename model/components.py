# model/components.py
import torch
from torch import nn
from typing import Optional


def soft_cap(x: torch.Tensor, cap_value: float) -> torch.Tensor:
    """Soft cap function."""
    return cap_value * torch.tanh(x / cap_value)


class RMSNorm(nn.Module):
    """RMS Normalization Layer.

    RMSNorm: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        elementwise_affine: Optional[bool] = None,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if elementwise_affine is not None:
            use_weight = use_weight and elementwise_affine
            use_bias = use_bias and elementwise_affine

        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.register_parameter("weight", None)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        input_dtype = x.dtype
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        normed = x / rms
        output = normed.to(dtype)

        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias

        return output


class MultiHeadLayerNorm(nn.Module):
    """Multi-Head Layer Normalization Layer."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        else:
            self.register_parameter("weight", None)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()

        input_dtype = x.dtype
        norm_x = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        normed = x / norm_x
        output = normed.to(dtype)

        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias

        return output
