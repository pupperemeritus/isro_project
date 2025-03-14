# model/generate.py
from typing import Callable, Tuple, Optional

import torch
from torch import nn

LLMForwardFnType = Callable[
    [
        torch.Tensor,
        Optional[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ],
    tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
]
SamplingFnType = Callable[[torch.Tensor], torch.Tensor]


def generate_tokens(
    llm_forward: LLMForwardFnType,
    prefill_tokens: torch.Tensor,
    max_length: int,
    token_sample_fn: SamplingFnType,
    state: Optional[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    device: str,
) -> Tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Generate tokens using the given language model forward function."""
    tokens = prefill_tokens
    with torch.no_grad():
        # prefill
        if prefill_tokens is not None:
            _, state = llm_forward(prefill_tokens, state)
        # generation
        for _ in range(max_length):
            logits, state = llm_forward(tokens[:, -1:], state)
            token_sample = token_sample_fn(logits[:, -1, :])
            tokens = torch.cat([tokens, token_sample[:, None]], dim=-1)

    return tokens, state


def get_sampling_fn(sampling_type: str) -> SamplingFnType:
    """Get the sampling function based on the sampling type."""
    if sampling_type == "greedy":
        token_sample_fn = greedy_sample
    elif sampling_type == "multinomial":
        token_sample_fn = multinomial_sample
    else:
        raise ValueError(f"Sampling type `{sampling_type}` not supported.")
    return token_sample_fn


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling."""
    token_sample = torch.argmax(logits, dim=-1)
    return token_sample


def multinomial_sample(logits: torch.Tensor) -> torch.Tensor:
    """Multinomial sampling."""
    probs = torch.softmax(logits, dim=-1)
    token_sample = torch.multinomial(probs, num_samples=1)
    return token_sample
