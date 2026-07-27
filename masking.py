"""
Copyright 2021 Aditya Gomatam.

This file is part of music-transformer (https://github.com/spectraldoy/music-transformer), my project to build and
train a Music Transformer. music-transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. music-transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. music-transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for music-transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.
"""

import torch
from functools import lru_cache
from vocabulary import pad_token

"""
Implementations of masking functionality for training a transformer:
    padding_mask: mask <pad> tokens in input sequences
    look_ahead_mask: mask subsequent positions for masked self-attention calculation
    combined_mask: elementwise maximum of above two
"""


def create_padding_mask(inp, n=4):
    mask = torch.eq(inp, pad_token).float()
    return mask.view(*mask.shape[:-1], *[1 for _ in range(n-2)], mask.shape[-1]).to(inp.device)


def _device_key(device):
    """Normalize device to a cache-friendly string key."""
    s = str(device)
    if s.startswith("privateuseone"):
        return "directml"
    return s


@lru_cache(maxsize=8)
def _cached_look_ahead_mask(seq_len, device_key, device='cpu'):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).float()
    return mask


def create_look_ahead_mask(seq_len, device='cpu'):
    return _cached_look_ahead_mask(seq_len, _device_key(device), device=device)


def create_mask(inp, n=4):
    padding_mask = create_padding_mask(inp, n=n)
    la_mask = _cached_look_ahead_mask(inp.shape[-1], _device_key(inp.device), device=inp.device)
    combined_mask = (padding_mask + la_mask).clamp(max=1)
    return combined_mask.to(inp.device)


def create_cache_mask(inp, past_length, n=4):
    """
    Creates a mask for a single new token step during autoregressive generation with KV cache.
    The new query token can attend to all past positions and itself (no look-ahead masking needed
    beyond the standard triangular mask for the new tokens).

    Args:
        inp: new tokens, shape (..., seq_len_q)
        past_length: number of cached positions
        n: number of dimensions for padding mask (default 4)

    Returns:
        combined mask of shape (..., 1, seq_len_q, past_length + seq_len_q)
    """
    seq_len_q = inp.shape[-1]
    total_len = past_length + seq_len_q

    padding_mask = create_padding_mask(inp, n=n)

    look_ahead = _cached_look_ahead_mask(seq_len_q, _device_key(inp.device), device=inp.device)
    left_block = torch.zeros(seq_len_q, past_length, device=inp.device, dtype=look_ahead.dtype)
    full_look_ahead = torch.cat([left_block, look_ahead], dim=-1)

    batch_shape = inp.shape[:-1]
    full_padding = torch.zeros(*batch_shape, 1, 1, total_len, device=inp.device)
    full_padding[..., past_length:] = padding_mask

    combined = (full_padding + full_look_ahead).clamp(max=1)
    return combined
