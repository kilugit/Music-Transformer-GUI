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

import copy
import torch
from math import sqrt
from torch import nn
from hparams import hparams
from layers import DecoderLayer, abs_positional_encoding

"""
Implementation of Music Transformer model with KV cache support for fast generation,
based on Huang et. al, 2018, Vaswani et. al, 2017
"""


class Decoder(nn.Module):
    """
    Custom decoder that replaces nn.TransformerDecoder to support KV caching.
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory=None, tgt_mask=None, past_key_values=None, return_kv=True):
        new_key_values = [] if return_kv else None
        for i, layer in enumerate(self.layers):
            pk = past_key_values[i][0] if past_key_values is not None else None
            pv = past_key_values[i][1] if past_key_values is not None else None
            tgt, k, v = layer(tgt, memory=memory, tgt_mask=tgt_mask, past_k=pk, past_v=pv)
            if return_kv:
                new_key_values.append((k, v))
        if self.norm is not None:
            tgt = self.norm(tgt)
        return tgt, new_key_values


class MusicTransformer(nn.Module):
    """
    Transformer Decoder with Relative Attention. Consists of:
        1. Input Embedding
        2. Absolute Positional Encoding
        3. Stack of N DecoderLayers
        4. Final Linear Layer
    """
    def __init__(self,
                 d_model=hparams["d_model"],
                 num_layers=hparams["num_layers"],
                 num_heads=hparams["num_heads"],
                 d_ff=hparams["d_ff"],
                 max_rel_dist=hparams["max_rel_dist"],
                 max_abs_position=hparams["max_abs_position"],
                 vocab_size=hparams["vocab_size"],
                 bias=hparams["bias"],
                 dropout=hparams["dropout"],
                 layernorm_eps=hparams["layernorm_eps"],
                 use_swiglu=hparams.get("use_swiglu", True),
                 use_qk_norm=hparams.get("use_qk_norm", True),
                 use_sdpa=hparams.get("use_sdpa", True)):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings. Set to 0 to compute normal attention
            max_abs_position (int): maximum absolute position for which to create sinusoidal absolute
                                    positional encodings. Set to 0 to compute pure relative attention
                                    make it greater than the maximum sequence length in the dataset if nonzero
            bias (bool, optional): if set to False, all Linear layers in the MusicTransformer will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model. Default: 0.1
            layernorm_eps (very small float, optional): epsilon for LayerNormalization. Default: 1e-6
            use_swiglu (bool): use SwiGLU FFN instead of ReLU FFN
            use_qk_norm (bool): apply RMSNorm to Q and K before attention
            use_sdpa (bool): use PyTorch SDPA for faster attention computation
        """
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        pe = abs_positional_encoding(max_abs_position, d_model)
        self.register_buffer("positional_encoding", pe)
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = Decoder(
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps,
                         use_swiglu=use_swiglu, use_qk_norm=use_qk_norm, use_sdpa=use_sdpa),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )

        self.final = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Llama-style init for better training stability."""
        nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        for layer in self.decoder.layers:
            nn.init.normal_(layer.self_attn.wq.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.self_attn.wk.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.self_attn.wv.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.self_attn.wo.weight, mean=0.0, std=0.02 / (2 * self.num_layers) ** 0.5)
            if hasattr(layer.ffn, 'gate'):
                nn.init.normal_(layer.ffn.gate.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.ffn.up.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.ffn.down.weight, mean=0.0, std=0.02 / (2 * self.num_layers) ** 0.5)
            else:
                for name, param in layer.ffn.named_parameters():
                    if 'weight' in name:
                        if name == 'main.2.weight':
                            nn.init.normal_(param, mean=0.0, std=0.02 / (2 * self.num_layers) ** 0.5)
                        else:
                            nn.init.normal_(param, mean=0.0, std=0.02)
        nn.init.normal_(self.final.weight, mean=0.0, std=0.02 / (2 * self.num_layers) ** 0.5)

    def forward(self, x, mask=None, past_key_values=None, return_kv=True):
        """
        Forward pass through the Music Transformer.

        Args:
            x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
            mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None
            past_key_values (optional, list of (K, V) tuples): cached keys/values for each layer
            return_kv (bool): if False, skip building KV cache (saves memory during training)

        Returns:
            (logits, new_key_values) where new_key_values is the updated KV cache (or None)
        """
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        if self.max_position > 0:
            offset = 0
            if past_key_values is not None:
                offset = past_key_values[0][0].shape[-2]
            x += self.positional_encoding[:, offset:offset + x.shape[-2], :]

        x = self.input_dropout(x)

        x, new_key_values = self.decoder(x, memory=None, tgt_mask=mask, past_key_values=past_key_values, return_kv=return_kv)

        x = self.final(x)

        return x, new_key_values
