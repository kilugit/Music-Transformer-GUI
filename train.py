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

import argparse
import sys
import time
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from hparams import device, get_grad_scaler, get_amp_context
from masking import create_mask
from model import MusicTransformer
from vocabulary import pad_token
import math

"""
Functionality to train a Music Transformer on a single CPU or single GPU

The transformer is an autoregressive model, which means that at the inference stage, it will make next predictions 
based on its previous outputs. However, while training, we can use teacher forcing - feeding the target into the 
model as previous output regardless of the true output of the model. This significantly cuts down on the compute 
required, while usually reducing loss (at the expense of generalizability of the model). Since we are training a 
generative model, the targets are simply the inputs shifted right by 1 position.
"""


class ExponentialMovingAverage:
    """
    Maintains EMA of model parameters for smoother inference.
    Updates: ema_param = decay * ema_param + (1 - decay) * param
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def transformer_lr_schedule(d_model, step_num, warmup_steps=4000):
    """
    As per Vaswani et. al, 2017, the post-LayerNorm transformer performs vastly better a custom learning rate
    schedule. Though the PyTorch implementation of the Music-Transformer uses pre-LayerNorm, which has been observed
    not to require a custom schedule, this function is here for utility.

    Args:
        d_model: embedding / hidden dimenision of the transformer
        step_num: current training step
        warmup_steps: number of transformer schedule warmup steps. Set to 0 for a continuously decaying learning rate

    Returns:
        learning rate at current step_num
    """
    if warmup_steps <= 0:
        step_num += 4000
        warmup_steps = 4000
    step_num = step_num + 1e-6  # avoid division by 0

    if type(step_num) == torch.Tensor:
        arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    else:
        arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

    return (d_model ** -0.5) * arg


def cosine_lr_schedule(d_model, step_num, warmup_steps=4000, total_steps=100000, min_lr_ratio=0.1):
    """
    Cosine learning rate schedule with linear warmup.
    Linearly warms up for warmup_steps, then cosine decays to min_lr_ratio * peak_lr.
    """
    peak_lr = d_model ** -0.5
    if step_num < warmup_steps:
        return peak_lr * (step_num / max(warmup_steps, 1))
    progress = (step_num - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)))


def loss_fn(prediction, target, label_smoothing=0.0):
    return F.cross_entropy(prediction, target, ignore_index=pad_token, label_smoothing=label_smoothing)


def train_step(model: MusicTransformer, opt, sched, inp, tar,
               scaler=None, accumulation_steps=1, micro_step=0,
               max_grad_norm=None, label_smoothing=0.0):
    logits, _ = model(inp, mask=create_mask(inp))
    loss = loss_fn(logits.transpose(-1, -2), tar, label_smoothing=label_smoothing)
    loss = loss / accumulation_steps

    if scaler is not None:
        scaler.scale(loss).backward()
        if (micro_step + 1) % accumulation_steps == 0:
            if max_grad_norm is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
    else:
        loss.backward()
        if (micro_step + 1) % accumulation_steps == 0:
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad()

    return float(loss.detach() * accumulation_steps)


def val_step(model: MusicTransformer, inp, tar):
    predictions, _ = model(inp, mask=create_mask(inp))
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    return float(loss)


class MusicTransformerTrainer:
    """
    As the transformer is a large model and takes a while to train on a GPU, or even a TPU, I wrote this Trainer
    class to make it easier to load and save checkpoints with the model. The way I've designed it instantiates the
    model, optimizer, and scheduler within the class itself, as there are some problems with passing them in. But,
    to get these objects back just call:
        trainer.model
        trainer.optimizer
        trainer.scheduler

    This class also tracks the cumulative losses while training, which you can get back with:
        trainer.train_losses
        trainer.val_losses
    as lists of floats

    To save a checkpoint, call trainer.save()
    To load a checkpoint, call trainer.load( (optional) ckpt_path)
    """

    def __init__(self, hparams_, datapath, batch_size, warmup_steps=4000,
                 ckpt_path="music_transformer_ckpt.pt", load_from_checkpoint=False,
                 use_amp=False, accumulation_steps=1,
                 lr_schedule="transformer", weight_decay=0.01, max_grad_norm=1.0,
                 label_smoothing=0.05, use_ema=False, ema_decay=0.999, total_steps=100000):
        """
        Args:
            hparams_: hyperparameters of the model
            datapath: path to the data to train on
            batch_size: batch size to batch the data
            warmup_steps: number of warmup steps for transformer learning rate schedule
            ckpt_path: path at which to save checkpoints while training; MUST end in .pt or .pth
            load_from_checkpoint (bool, optional): if true, on instantiating the trainer, this will load a previously
                                                   saved checkpoint at ckpt_path
            lr_schedule: learning rate schedule type ("transformer" or "cosine")
            weight_decay: AdamW weight decay coefficient
            max_grad_norm: maximum gradient norm for gradient clipping (None to disable)
            label_smoothing: label smoothing factor for cross-entropy loss
            use_ema: whether to use exponential moving average of model weights
            ema_decay: decay rate for EMA
            total_steps: total training steps (for cosine schedule)
        """
        # get the data
        self.datapath = datapath
        self.batch_size = batch_size
        data = torch.load(datapath).long()  # kept on CPU to avoid filling VRAM

        # max absolute position must be able to account for the largest sequence in the data
        if hparams_["max_abs_position"] > 0:
            hparams_ = dict(hparams_)
            hparams_["max_abs_position"] = max(hparams_["max_abs_position"], data.shape[-1])

        # train / validation split: 80 / 20
        train_len = round(data.shape[0] * 0.8)
        train_data = data[:train_len]
        val_data = data[train_len:]
        print(f"There are {data.shape[0]} samples in the data, {len(train_data)} training samples and {len(val_data)} "
              "validation samples")

        # datasets and dataloaders: split data into first (n-1) and last (n-1) tokens
        self.train_ds = TensorDataset(train_data[:, :-1], train_data[:, 1:])
        num_workers = min(4, os.cpu_count() or 1)
        if sys.platform == "win32" and device.type in ("cuda", "xpu"):
            num_workers = 0
        pin = device.type in ("cuda", "xpu")
        self.train_dl = DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin,
                                   persistent_workers=num_workers > 0)

        self.val_ds = TensorDataset(val_data[:, :-1], val_data[:, 1:])
        self.val_dl = DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin,
                                 persistent_workers=num_workers > 0)

        # create model
        self.model = MusicTransformer(**hparams_).to(device)
        self.hparams = hparams_

        # setup training
        self.warmup_steps = warmup_steps
        self.lr_schedule = lr_schedule
        self.total_steps = total_steps
        self.max_grad_norm = max_grad_norm
        self.label_smoothing = label_smoothing

        # Use AdamW with decoupled weight decay for better regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.0, betas=(0.9, 0.98),
                                      weight_decay=weight_decay)
        if lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda x: cosine_lr_schedule(self.hparams['d_model'], x, self.warmup_steps,
                                             self.total_steps)
            )
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
            )
        self.scaler = get_grad_scaler(use_amp)
        self.accumulation_steps = accumulation_steps

        # EMA
        self.ema = None
        if use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)

        # setup checkpointing / saving
        self.ckpt_path = ckpt_path
        self.train_losses = []
        self.val_losses = []

        # load checkpoint if necessesary
        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()

    def save(self, ckpt_path=None):
        """
        Saves a checkpoint at ckpt_path

        Args:
            ckpt_path (str, optional): if None, saves the checkpoint at the previously stored self.ckpt_path
                                       else saves the checkpoints at the new passed-in path, and stores this new path at
                                       the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "validation_losses": self.val_losses,
            "warmup_steps": self.warmup_steps,
            "hparams": self.hparams,
            "lr_schedule": self.lr_schedule,
            "max_grad_norm": self.max_grad_norm,
            "label_smoothing": self.label_smoothing,
        }
        if self.ema is not None:
            ckpt["ema_shadow"] = self.ema.shadow
        ckpt["weight_decay"] = self.optimizer.param_groups[0].get("weight_decay", 0.01)

        torch.save(ckpt, self.ckpt_path)
        return

    def load(self, ckpt_path=None):
        """
        Loads a checkpoint from ckpt_path
        NOTE: OVERWRITES THE MODEL STATE DICT, OPTIMIZER STATE DICT, SCHEDULER STATE DICT, AND HISTORY OF LOSSES

        Args:
            ckpt_path (str, optional): if None, loads the checkpoint at the previously stored self.ckpt_path
                                       else loads the checkpoints from the new passed-in path, and stores this new path
                                       at the member variable self.ckpt_path
        """
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path

        ckpt = torch.load(self.ckpt_path)

        # ensure backward compatibility with old checkpoints
        ckpt_hparams = ckpt["hparams"]
        for key in ["use_swiglu", "use_qk_norm", "use_sdpa"]:
            if key not in ckpt_hparams:
                ckpt_hparams[key] = False

        # create and load model
        self.model = MusicTransformer(**ckpt_hparams).to(device)
        self.hparams = ckpt_hparams
        print("Loading the model...", end="")
        print(self.model.load_state_dict(ckpt["model_state_dict"], strict=False))

        # create and load load optimizer and scheduler
        self.warmup_steps = ckpt.get("warmup_steps", 4000)
        self.lr_schedule = ckpt.get("lr_schedule", "transformer")
        self.max_grad_norm = ckpt.get("max_grad_norm", self.max_grad_norm)
        self.label_smoothing = ckpt.get("label_smoothing", self.label_smoothing)

        wd = ckpt.get("weight_decay", 0.01)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.0, betas=(0.9, 0.98), weight_decay=wd)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda x: cosine_lr_schedule(self.hparams['d_model'], x, self.warmup_steps, self.total_steps)
            )
        else:
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
            )
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # load EMA if available
        if self.ema is not None and "ema_shadow" in ckpt:
            self.ema.shadow = ckpt["ema_shadow"]

        # load loss histories
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["validation_losses"]

        return

    def fit(self, epochs):
        """
        Training loop to fit the model to the data stored at the passed in datapath. If KeyboardInterrupt at anytime
        during the training loop, and if progresss being printed, this method will save a checkpoint at the 
        passed-in ckpt_path

        Args:
            epochs: number of epochs to train for.

        Returns:
            history of training and validation losses for this training session
        """
        train_losses = []
        val_losses = []
        start = time.time()

        print("Beginning training...")
        print(time.strftime("%Y-%m-%d %H:%M"))
        model = self.model

        amp_ctx = get_amp_context(self.scaler is not None)

        try:
            for epoch in range(epochs):
                train_epoch_losses = []
                val_epoch_losses = []

                model.train()
                self.optimizer.zero_grad()
                for micro_step, (train_inp, train_tar) in enumerate(self.train_dl):
                    train_inp = train_inp.to(device)
                    train_tar = train_tar.to(device)
                    with amp_ctx:
                        loss = train_step(model, self.optimizer, self.scheduler,
                                          train_inp, train_tar, self.scaler,
                                          self.accumulation_steps, micro_step,
                                          max_grad_norm=self.max_grad_norm,
                                          label_smoothing=self.label_smoothing)
                    train_epoch_losses.append(loss)
                if self.scaler is not None:
                    self.optimizer.zero_grad()

                # Update EMA after each epoch
                if self.ema is not None:
                    self.ema.update(model)

                model.eval()
                # Apply EMA for validation if enabled
                if self.ema is not None:
                    self.ema.apply(model)
                for val_inp, val_tar in self.val_dl:
                    val_inp = val_inp.to(device)
                    val_tar = val_tar.to(device)
                    loss = val_step(model, val_inp, val_tar)
                    val_epoch_losses.append(loss)
                if self.ema is not None:
                    self.ema.restore(model)

                # mean losses for the epoch
                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)

                # store complete history of losses in member lists and relative history for this session in output lists
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)

                print(f"Epoch {epoch } Time taken {round(time.time() - start, 2)} seconds "
                    f"Train Loss {train_losses[-1]} Val Loss {val_losses[-1]}")
                start = time.time()

        except KeyboardInterrupt:
            pass

        # Save final model with EMA weights if available
        if self.ema is not None:
            self.ema.apply(model)

        print("Checkpointing...")
        self.save()
        if self.ema is not None:
            self.ema.restore(model)
        print("Done")
        print(time.strftime("%Y-%m-%d %H:%M"))

        return train_losses, val_losses


if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a Music Transformer on single tensor dataset of preprocessed MIDI files"
    )

    # trainer arguments
    parser.add_argument("datapath", help="path at which preprocessed MIDI files are stored as a single tensor after "
                                         "being translated into an event vocabulary")
    parser.add_argument("ckpt_path", help="path at which to load / store checkpoints while training; "
                                          "KeyboardInterrupt while training to checkpoint the model; MUST end in .pt "
                                          "or .pth", type=str)
    parser.add_argument("save_path", help="path at which to save the model's state dict and hyperparameters after "
                                          "training; model will only be saved if the training loop finishes before a "
                                          "KeyboardInterrupt; MUST end in .pt or .pth", type=str)
    parser.add_argument("epochs", help="number of epochs to train for", type=check_positive)
    parser.add_argument("-bs", "--batch-size", help="number of sequences to batch together to compute a single "
                                                     "training step while training; default: 16", type=check_positive)
    parser.add_argument("-l", "--load-checkpoint", help="flag to load a previously saved checkpoint from which to "
                                                        "resume training; default: False", action="store_true")
    parser.add_argument("-w", "--warmup-steps", help="number of warmup steps for learning rate scheduler; "
                                                     "default: 4000", type=int)

    # hyperparameters
    parser.add_argument("-d", "--d-model",
                        help="music transformer hidden dimension size; default: 128", type=check_positive)
    parser.add_argument("-nl", "--num-layers",
                        help="number of transformer decoder layers; default: 3", type=check_positive)
    parser.add_argument("-nh", "--num-heads",
                        help="number of attention heads; default: 8", type=check_positive)
    parser.add_argument("-dff", "--d-feedforward",
                        help="hidden dimension size of FFN layers; default: 512", type=check_positive)
    parser.add_argument("-mrd", "--max-rel-dist",
                        help="maximum relative distance for relative attention; default: 1024", type=check_positive)
    parser.add_argument("-map", "--max-abs-position",
                        help="maximum absolute sequence length; 0 = no absolute PE; default: 0", type=int)
    parser.add_argument("-vs", "--vocab-size",
                        help="vocabulary size; default: 416", type=check_positive)
    parser.add_argument("-nb", "--no-bias",
                        help="disable bias in linear layers", action="store_false")
    parser.add_argument("-dr", "--dropout", help="dropout rate; default: 0.1")
    parser.add_argument("-le", "--layernorm-eps", help="layernorm epsilon; default: 1e-6")

    # VRAM control arguments
    parser.add_argument("--use-amp", help="enable automatic mixed precision training",
                        action="store_true")
    parser.add_argument("-ga", "--gradient-accumulation-steps",
                        help="accumulate gradients over N micro-batches; default: 1",
                        type=check_positive, default=1)

    # Upgraded training arguments
    parser.add_argument("--lr-schedule", help="learning rate schedule: 'transformer' or 'cosine'; default: cosine",
                        choices=["transformer", "cosine"], default="cosine")
    parser.add_argument("-wd", "--weight-decay", help="AdamW weight decay; default: 0.01",
                        type=float, default=0.01)
    parser.add_argument("-gn", "--max-grad-norm", help="max gradient norm for clipping; default: 1.0",
                        type=float, default=1.0)
    parser.add_argument("-ls", "--label-smoothing", help="label smoothing factor; default: 0.05",
                        type=float, default=0.05)
    parser.add_argument("--use-ema", help="enable EMA of model weights for smoother generation",
                        action="store_true")
    parser.add_argument("--ema-decay", help="EMA decay rate; default: 0.999", type=float, default=0.999)
    parser.add_argument("--total-steps", help="total training steps for cosine schedule; default: 100000",
                        type=int, default=100000)

    # Architecture upgrade toggles
    parser.add_argument("--no-swiglu", help="disable SwiGLU FFN (use ReLU instead)", action="store_true")
    parser.add_argument("--no-qk-norm", help="disable QK-Norm", action="store_true")
    parser.add_argument("--no-sdpa", help="disable PyTorch SDPA (use custom attention)", action="store_true")

    args = parser.parse_args()

    # fix optional parameters
    batch_size_ = 16 if args.batch_size is None else args.batch_size
    warmup_steps_ = 2000 if args.warmup_steps is None else args.warmup_steps

    # fix hyperparameters
    hparams["d_model"] = args.d_model if args.d_model else hparams["d_model"]
    hparams["num_layers"] = args.num_layers if args.num_layers else hparams["num_layers"]
    hparams["num_heads"] = args.num_heads if args.num_heads else hparams["num_heads"]
    hparams["d_ff"] = args.d_feedforward if args.d_feedforward else hparams["d_ff"]
    hparams["max_rel_dist"] = args.max_rel_dist if args.max_rel_dist else hparams["max_rel_dist"]
    hparams["max_abs_position"] = args.max_abs_position if args.max_abs_position else hparams["max_abs_position"]
    hparams["vocab_size"] = args.vocab_size if args.vocab_size else hparams["vocab_size"]
    hparams["bias"] = args.no_bias
    hparams["dropout"] = args.dropout if args.dropout else hparams["dropout"]
    hparams["layernorm_eps"] = args.layernorm_eps if args.layernorm_eps else hparams["layernorm_eps"]
    hparams["use_swiglu"] = not args.no_swiglu
    hparams["use_qk_norm"] = not args.no_qk_norm
    hparams["use_sdpa"] = not args.no_sdpa

    # set up the trainer
    print("Setting up the trainer...")
    trainer = MusicTransformerTrainer(hparams, args.datapath, batch_size_, warmup_steps_,
                                      args.ckpt_path, args.load_checkpoint,
                                      use_amp=args.use_amp,
                                      accumulation_steps=args.gradient_accumulation_steps,
                                      lr_schedule=args.lr_schedule,
                                      weight_decay=args.weight_decay,
                                      max_grad_norm=args.max_grad_norm,
                                      label_smoothing=args.label_smoothing,
                                      use_ema=args.use_ema,
                                      ema_decay=args.ema_decay,
                                      total_steps=args.total_steps)
    print()

    # train the model
    trainer.fit(args.epochs)

    # done training, save the model
    print("Saving...")
    # Save final model with EMA weights if used
    if trainer.ema is not None:
        trainer.ema.apply(trainer.model)
    save_file = {
        "state_dict": trainer.model.state_dict(),
        "hparams": trainer.hparams
    }
    torch.save(save_file, args.save_path)
    if trainer.ema is not None:
        trainer.ema.restore(trainer.model)
    print("Done!")
