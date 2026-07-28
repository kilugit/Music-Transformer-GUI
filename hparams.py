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
import json
import os
from vocabulary import vocab_size

device_type = None


def _get_backend_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("backend", "auto")
        except Exception:
            pass
    return "auto"


def _detect_device():
    global device_type
    backend = _get_backend_config()

    if backend == "cpu":
        device_type = "cpu"
        return torch.device("cpu")

    if backend in ("directml",):
        try:
            import torch_directml
            device_type = "directml"
            return torch_directml.device(0)
        except ImportError:
            device_type = "cpu"
            return torch.device("cpu")

    if backend == "rocm":
        if torch.cuda.is_available() and getattr(torch.version, "hip", None):
            device_type = "rocm"
            return torch.device("cuda:0")
        try:
            import torch_directml
            device_type = "directml"
            return torch_directml.device(0)
        except ImportError:
            device_type = "cpu"
            return torch.device("cpu")

    if backend == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
            return torch.device("xpu:0")
        if torch.cuda.is_available() and getattr(torch.version, "hip", None):
            device_type = "cuda"
            return torch.device("cuda:0")
        if torch.cuda.is_available():
            device_type = "cuda"
            return torch.device("cuda:0")
        try:
            import torch_directml
            device_type = "directml"
            return torch_directml.device(0)
        except ImportError:
            device_type = "cpu"
            return torch.device("cpu")

    if backend in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device_type = "cuda"
            return torch.device("cuda:0")
        try:
            import torch_directml
            device_type = "directml"
            return torch_directml.device(0)
        except ImportError:
            device_type = "cpu"
            return torch.device("cpu")

    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        device_type = "rocm"
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        device_type = "cuda"
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_type = "xpu"
        return torch.device("xpu:0")
    try:
        import torch_directml
        device_type = "directml"
        return torch_directml.device(0)
    except ImportError:
        device_type = "cpu"
        return torch.device("cpu")


device = _detect_device()


def get_amp_context(enabled=True):
    if not enabled:
        return torch.cpu.amp.autocast(enabled=False)
    if device_type == "xpu":
        return torch.amp.autocast(device_type="xpu", dtype=torch.float16)
    if device_type in ("cuda", "rocm"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if device_type == "directml":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    return torch.cpu.amp.autocast(enabled=False)


def get_grad_scaler(enabled=True):
    if not enabled or device_type in ("cpu", "directml"):
        return None
    if device_type == "xpu":
        return torch.amp.GradScaler("xpu")
    if device_type in ("cuda", "rocm"):
        return torch.amp.GradScaler()
    return None


hparams = {
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6,
    "use_swiglu": True,
    "use_qk_norm": True,
    "use_sdpa": True,
}

hparams_8gb = {
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 4,
    "d_ff": 256,
    "max_rel_dist": 512,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6,
    "use_swiglu": True,
    "use_qk_norm": True,
    "use_sdpa": True,
}

hparams_large = {
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 1024,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": vocab_size,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6,
    "use_swiglu": True,
    "use_qk_norm": True,
    "use_sdpa": True,
}
