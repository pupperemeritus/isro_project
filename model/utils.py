# model/utils.py
import torch


def round_up_to_next_multiple_of(x: int, y: int) -> int:
    """Rounds x up to the next multiple of y."""
    return (x + y - 1) // y * y
