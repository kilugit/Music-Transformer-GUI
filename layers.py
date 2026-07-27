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
import torch.nn.functional as F
from torch import nn
from math import sqrt

"""
Implementation of layers and functionality necessary to build Music Transformer model,
based on Huang et. al, 2018, Vaswani et. al, 2017

Upgraded with:
- RMSNorm (Llama-style) for better training stability
- SwiGLU FFN (PaLM/Llama) for better representational power
- QK-Norm for stable attention logits
- PyTorch SDPA (Flash Attention) for faster attention
"""


class RMSNorm(nn.Module):
    """RMS Layer Normalization as used in Llama, Mistral, etc."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def abs_positional_encoding(max_position, d_model, n=3, device=None):
    if max_position <= 0:
        return torch.empty(0, d_model)

    positions = torch.arange(max_position, device=device).float()
    k = torch.arange(d_model, device=device).float()
    coeffs = 1 / torch.pow(10000, 2 * (k // 2) / d_model)
    angles = positions.view(-1, 1) @ coeffs.view(1, -1)

    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    angles[:, 1::2] = torch.cos(angles[:, 1::2])

    return angles.view(*[1 for _ in range(n-2)], max_position, d_model)


def skew(t):
    """
    Implements Huang et. al, 2018's skewing algorithm to correctly reorder the
    dot(Q, RelativePositionEmbeddings) matrix.

    Algorithm:
        1. Pad T
        2. Reshape
        3. Slice

    Args:
        t (torch.Tensor): tensor to skew

    Returns:
        Srel: skewed t: nth column from the right is skewed into the nth diagonal
    """
    padded = F.pad(t, [1, 0])
    Srel = padded.reshape(-1, t.shape[-1] + 1, t.shape[-2])
    Srel = Srel[:, 1:]
    Srel = Srel.reshape(*t.shape)
    return Srel


def rel_scaled_dot_prod_attention(q, k, v, e=None, mask=None):
    QKt = torch.matmul(q, k.transpose(-1, -2))
    dk = sqrt(k.shape[-1])

    if e is not None:
        Srel = skew(torch.matmul(q, e.transpose(-1, -2)))
        scaled_attention_logits = (QKt + Srel) / dk
    else:
        scaled_attention_logits = QKt / dk

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    return torch.matmul(F.softmax(scaled_attention_logits, dim=-1), v)


class MultiHeadAttention(nn.Module):
    """
    MultiHead Relative Attention Block. Computes attention for input batch along num_heads "heads".
    In the process, attention weights are calculated num_heads times, which allows the network to
    extract information from the input batch through several different representations simultaneously
    """
    def __init__(self, d_model, num_heads, max_rel_dist, bias=True, qk_norm=False, use_sdpa=False):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings; set to 0 to compute normal attention
            bias (bool, optional): if set to False, all Linear layers in the MHA block will not learn
                                   an additive bias. Default: True
            qk_norm (bool): apply RMSNorm to Q and K before attention (stabilizes training)
            use_sdpa (bool): use PyTorch SDPA (Flash Attention) for faster attention computation

        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_rel_dist = max_rel_dist
        self.batch_first = False
        self.qk_norm = qk_norm
        self.use_sdpa = use_sdpa

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible into num_heads heads")

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate Q from input
        self.wk = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate K from input
        self.wv = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate V from input

        self.E = nn.Embedding(max(self.max_rel_dist, 1), self.d_model)      # relative position embeddings

        self.wo = nn.Linear(self.d_model, self.d_model, bias=True)  # final output layer

        if qk_norm:
            self.q_norm = RMSNorm(self.depth, eps=1e-6)
            self.k_norm = RMSNorm(self.depth, eps=1e-6)

    @staticmethod
    def split_heads(x, num_heads, depth=None):
        """
        Helper function to split input x along num_heads heads

        Args:
            x: input tensor to split into heads; shape: (..., L, d_model); d_model = num_heads * depth
            num_heads (int): number of heads along which to calculate attention
            depth (int, optional): desired dimensionality at each head

        Returns:
            input tensor correctly reshaped and transposed to shape (..., num_heads, L, depth)
        """
        # get depth if None
        if depth is None:
            if x.shape[-1] % num_heads != 0:
                raise ValueError("d_model must be divisible into num_heads")
            depth = x.shape[-1] // num_heads

        # reshape and transpose x
        x = x.view(*x.shape[:-1], num_heads, depth)     # (..., L, num_heads, depth)
        return x.transpose(-2, -3)                      # (..., num_heads, L, depth)

    def get_required_embeddings(self, seq_len, max_len=None):
        if max_len is None:
            max_len = self.E.num_embeddings

        E_dev = self.E.weight.device
        n_extra = max(seq_len - max_len, 0)
        start = max(max_len - seq_len, 0)
        total_len = min(seq_len, max_len)

        indices = torch.arange(start, start + total_len, device=E_dev)
        result = self.E(indices)
        if n_extra > 0:
            first = self.E(torch.tensor(0, device=E_dev)).unsqueeze(0)
            result = torch.cat([first.expand(n_extra, -1), result], dim=0)
        return result

    def forward(self, q, k, v, mask=None, past_k=None, past_v=None):
        """
        Computes Multi-Head Attention on input tensors Q, K, V

        Args:
            q: Queries tensor of shape (..., seq_len_q, d_model)
            k: Keys tensor of shape (..., seq_len_k, d_model)
            v: Values tensor of shape (..., seq_len_k, d_model)
            mask (optional): mask for input batch with ones indicating positions to mask. Default: None
            past_k (optional): cached keys from previous steps, shape (..., past_len, d_model)
            past_v (optional): cached values from previous steps, shape (..., past_len, d_model)

        Returns:
            (attention output, updated full keys, updated full values)
        """
        # get Q, K, V
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # KV cache: concatenate with cached keys/values
        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
        if past_v is not None:
            v = torch.cat([past_v, v], dim=-2)

        # get required embeddings from E
        seq_len_k = k.shape[-2]
        e = self.get_required_embeddings(seq_len_k, self.max_rel_dist)  # (seq_len_k, d_model)

        # split into heads
        q_h = self.split_heads(q, self.num_heads, self.depth)  # (batch_size, h, seq_len_q, depth)
        k_h = self.split_heads(k, self.num_heads, self.depth)  # (batch_size, h, seq_len_k, depth)
        v_h = self.split_heads(v, self.num_heads, self.depth)  # (batch_size, h, seq_len_k, depth)
        e = self.split_heads(e, self.num_heads, self.depth)  # (h, seq_len_k, depth)

        # QK-Norm: normalize Q and K per head for stable training
        if self.qk_norm:
            q_h = self.q_norm(q_h)
            k_h = self.k_norm(k_h)

        if self.use_sdpa and self.max_rel_dist > 0:
            # Compute relative position bias and use PyTorch SDPA
            rel_bias = skew(torch.matmul(q_h, e.transpose(-1, -2))) / sqrt(self.depth)
            if mask is not None:
                rel_bias = rel_bias + (mask * -1e9)
            attn_out = F.scaled_dot_product_attention(
                q_h, k_h, v_h, attn_mask=rel_bias
            )
        elif self.use_sdpa:
            # Standard SDPA without relative bias
            attn_mask = (mask * -1e9) if mask is not None else None
            attn_out = F.scaled_dot_product_attention(
                q_h, k_h, v_h, attn_mask=attn_mask
            )
        else:
            # Original custom relative attention
            attn_out = rel_scaled_dot_prod_attention(q_h, k_h, v_h, e, mask=mask)

        # concatenate heads and pass through final layer
        attn_out = attn_out.transpose(-2, -3)  # (batch_size, seq_len_q, h, depth)
        sh = attn_out.shape
        attn_out = self.wo(attn_out.reshape(*sh[:-2], self.d_model))
        return attn_out, k, v  # return updated K, V for caching


class PointwiseFFN(nn.Module):
    """
    Fully-connected Feedforward layer that follows the MHA block in each Transformer layer, which is simply a 2 layer
    Dense network with a ReLU in between
    """
    def __init__(self, d_model, d_ff, bias=True):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            d_ff (int): intermediate dimension of FFN blocks
            bias (bool, optional): if set to False, all Linear layers in the FFN block will not learn
                                   an additive bias. Default: True
        """
        super(PointwiseFFN, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias)
        )

    def forward(self, x):
        return self.main(x)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in PaLM, Llama, GPT-4.
    Uses 3 projections (gate, up, down) with SiLU activation on the gate.
    Maintains roughly the same parameter count as the equivalent ReLU FFN
    by using hidden_dim = int(2/3 * d_ff).
    """
    def __init__(self, d_model, d_ff, bias=True):
        super().__init__()
        hidden_dim = int(2 * d_ff / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    """
    Every TransformerDecoder layer consists of 2 sublayers:
        1. Masked Multi-Head Attention
        2. Pointwise Feedforward Network
    In the original Transformer, each sublayer further employs a residual connection followed by a LayerNorm on the last
    dimension. However, here the LayerNormalization will be placed before the residual connnection, as this Pre-LN
    architecture does not generally require an explicitly designed learning rate schedule.
    """
    def __init__(self, d_model, num_heads, d_ff, max_rel_dist, bias=True, dropout=0.1, layernorm_eps=1e-6,
                 use_swiglu=False, use_qk_norm=False, use_sdpa=False):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings; set to 0 to compute normal attention
            bias (bool, optional): if set to False, all Linear layers in the Decoder will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model
            layernorm_eps (very small positive float, optional): epsilon for LayerNormalization
            use_swiglu (bool): use SwiGLU FFN instead of ReLU FFN
            use_qk_norm (bool): apply RMSNorm to Q and K before attention
            use_sdpa (bool): use PyTorch SDPA for faster attention
        """
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_rel_dist = max_rel_dist

        self.self_attn = MultiHeadAttention(d_model, num_heads, max_rel_dist, bias,
                                             qk_norm=use_qk_norm, use_sdpa=use_sdpa)
        self.ffn = SwiGLUFFN(d_model, d_ff, bias) if use_swiglu else PointwiseFFN(d_model, d_ff, bias)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=None, memory_is_causal=None,
                past_k=None, past_v=None):
        """
        Forward pass through decoder layer.

        Args:
            tgt: input queries tensor from previous layer
            tgt_mask (optional): tensor with 1's indicating positions to mask. Default: None
            past_k (optional): cached keys from previous step for this layer
            past_v (optional): cached values from previous step for this layer

        Returns:
            (output, updated keys, updated values)
        """
        # multi-head attention block
        attn_out = self.layernorm1(tgt)
        attn_out, k, v = self.self_attn(attn_out, attn_out, attn_out, mask=tgt_mask,
                                         past_k=past_k, past_v=past_v)
        attn_out = self.dropout1(attn_out)
        attn_out = tgt + attn_out

        # pointwise ffn block
        ffn_out = self.layernorm2(attn_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out, k, v
