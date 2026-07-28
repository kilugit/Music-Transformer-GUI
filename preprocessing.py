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

import os
import argparse
import torch
import torch.nn.functional as F
from random import randint, sample
from sys import exit
from vocabulary import *
from tokenizer import midi_parser
import glob
from multiprocessing import Pool, cpu_count

"""
Functionality to preprocess MIDI files translated into indices in the event vocabulary from command line
"""


def sample_end_data(seqs, lth, factor=6):
    """
    Randomly samples sequences of length ~lth from an input set of sequences seqs. Prepares data for augmentation.
    Returns a list. Samples from the end of each sequence.

    Args:
        seqs (list): list of sequences in the event vocabulary
        lth (int): approximate length to cut sequences into
        factor (int): factor to vary range of output lengths; Default: 6. Higher factor will narrow the output range

    Returns:
        input sequs cut to length ~lth
    """
    data = []
    for seq in seqs:
        lower_bound = max(len(seq) - lth, 0)
        idx = randint(lower_bound, lower_bound + lth // factor)
        data.append(seq[idx:])

    return data


def sample_data(seqs, lth, factor=6):
    """
    Randomly samples sequences of length ~lth from an input set of sequences seqs. Prepares data for augmentation.
    Returns a list.

    Args:
        seqs (list): list of sequences in the event vocabulary
        lth (int): approximate length to cut sequences into
        factor (int): factor to vary range of output lengths; Default: 6. Higher factor will narrow the output range

    Returns:
        input sequs cut to length ~lth
    """
    data = []
    for seq in seqs:
        length = randint(lth - lth // factor, lth + lth // factor)
        idx = randint(0, max(0, len(seq) - length))
        data.append(seq[idx:idx+length])
        
    return data


def aug(data, note_shifts=None, time_stretches=None, verbose=False):
    """
    Augments data up and down in pitch by note_shifts and faster and slower in time by time_stretches. Adds start
    and end tokens and pads to max sequence length in data

    Args:
        data (list of lists of ints): sequences to augment
        note_shifts (list): pitch transpositions to be made
        time_stretches (list): stretches in time to be made
        verbose (bool): set to True to periodically print augmentation progress

    Returns:
        input data with pitch transpositions and time stretches, concatendated to one tensor
    """
    if note_shifts is None:
        note_shifts = torch.arange(-2, 3)
    if time_stretches is None:
        time_stretches = [1, 1.05, 1.1]
    if any([i <= 0 for i in time_stretches]):
        raise ValueError("time_stretches must all be positive")

    time_stretches = _expand_time_stretches(time_stretches)
    if 1 not in time_stretches:
        time_stretches = [1.0] + time_stretches

    # convert shifts to plain ints once
    shift_vals = [s.item() if isinstance(s, torch.Tensor) else s for s in note_shifts]

    # vectorized note shifting
    note_shifted_data = []
    count = 0
    note_on_hi = note_on_events
    note_off_lo = note_on_events + 1
    note_off_hi = note_events
    for seq in data:
        seq = seq.to(torch.long)
        for _shift in shift_vals:
            shifted = seq + _shift
            valid_note_on = (seq > 0) & (seq <= note_on_hi) & (shifted > 0) & (shifted <= note_on_hi)
            valid_note_off = (seq >= note_off_lo) & (seq <= note_off_hi) & (shifted >= note_off_lo) & (shifted <= note_off_hi)
            note_shifted_data.append(torch.where(valid_note_on | valid_note_off, shifted, seq))
            count += 1
            if verbose:
                print(f"Transposed {count} sequences")

    # optimized time stretching via run-length grouping
    time_stretched_data = []
    ts_lo = note_events + 1
    ts_hi = note_events + time_shift_events
    count = 0
    for seq in note_shifted_data:
        seq = seq.to(torch.long)
        for time_stretch in time_stretches:
            is_ts = (seq >= ts_lo) & (seq <= ts_hi)
            non_ts_positions = torch.where(~is_ts)[0]

            # Build segments: [ts_run_0, non_ts_0, ts_run_1, non_ts_1, ..., ts_run_n]
            segments = []
            prev = 0
            for pos in non_ts_positions:
                if pos > prev:
                    segments.append(seq[prev:pos])
                segments.append(seq[pos:pos+1])
                prev = pos + 1
            if prev < len(seq):
                segments.append(seq[prev:])

            result = []
            for seg in segments:
                if len(seg) > 0 and ts_lo <= seg[0].item() <= ts_hi:
                    ts_times = (seg - note_events).float()
                    for tt in ts_times:
                        stretched = int(tt * DIV * time_stretch + 0.5)
                        if stretched > 0:
                            time_to_events(stretched, index_list=result)
                else:
                    result.append(seg[0].item() if len(seg) > 0 else 0)

            time_stretched_data.append(torch.tensor(result, dtype=torch.long))
            count += 1
            if verbose:
                print(f"Stretched {count} sequences")

    # preface and suffix with start and end tokens
    aug_data = []
    for seq in time_stretched_data:
        aug_data.append(F.pad(F.pad(seq, (1, 0), value=start_token), (0, 1), value=end_token))

    # pad all sequences to max length
    aug_data = torch.nn.utils.rnn.pad_sequence(aug_data, padding_value=pad_token).transpose(-1, -2)
    return aug_data


def _expand_time_stretches(time_stretches):
    """Expand time stretches to include reciprocal values, deduplicated and sorted."""
    ts_set = set()
    for t in time_stretches:
        ts_set.add(t)
        if t != 1:
            ts_set.add(1 / t)
    return sorted(ts_set)


def randomly_sample_aug_data(aug_data, k, note_shifts=None, time_stretches=None):
    """
    Randomly samples k sets of augmented data to cut down dataset

    Args:
        aug_data (torch.Tensor): augmented dataset
        k (int): number of original sequences to sample
        note_shifts (list): pitch transpositions used during augmentation
        time_stretches (list): time stretches used during augmentation
    """
    if note_shifts is None:
        note_shifts = torch.arange(-2, 3)
    if time_stretches is None:
        time_stretches = [1, 1.05, 1.1]
    ts_expanded = _expand_time_stretches(time_stretches)
    if 1.0 not in ts_expanded:
        ts_expanded = [1.0] + ts_expanded
    augs = len(note_shifts) * len(ts_expanded)
    random_indices = sample(range(len(aug_data) // augs), k=k)
    out = torch.cat(
        [t[i * augs:i * augs + augs] for i in random_indices],
        dim=0
    )
    return out


def _parse_midi_file(file):
    """Worker for multiprocessed MIDI parsing."""
    try:
        return midi_parser(fname=file)[0]
    except (OSError, ValueError, EOFError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocessing.py",
        description="Preprocess MIDI files into single tensor for ML"
    )
    parser.add_argument("source", help="source directory of MIDI files to preprocess")
    parser.add_argument("destination", help="destination path at which to save preprocessed data as a single tensor, "
                                            "including filename and extension")
    parser.add_argument("length", help="approximate sequence length to cut data into (length will be randomly sampled)",
                        type=int)
    parser.add_argument("-a", "--from-augmented-data", help="flag to specify whether or not the source contains "
                                                            "already augmented data",  action="store_true")
    parser.add_argument("-t", "--transpositions", help="list of pitch transpositions to make in data augmentation",
                        nargs="+", type=int)
    parser.add_argument("-s", "--time-stretches", help="list of stretches in time to make in data augmentation",
                        nargs="+", type=float)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")
    args = parser.parse_args()

    # if source directory doesn't exist, exit
    if not os.path.isdir(args.source):
        print("Error: source must be an existing directory")
        exit(1)

    # fix save path if necessary
    if os.path.isdir(args.destination):
        args.destination = os.path.join(args.destination, "gnershk.pt")
    elif not args.destination.endswith((".pt", ".pth")):
        args.destination += ".pt"

    # turn length into int
    args.length = int(args.length)

    DATA = []
    PATH = args.source

    # load parsed midi files
    if not args.from_augmented_data:
        files = list(glob.iglob(os.path.join(PATH, '**', '*.mid*'), recursive=True))
        num_workers = min(cpu_count(), 4)
        print(f"Translating {len(files)} midi files to event vocabulary (using {num_workers} workers)...") if args.verbose else None
        try:
            with Pool(num_workers) as pool:
                results = list(pool.imap_unordered(_parse_midi_file, files))
        except Exception:
            print("Multiprocessing failed, falling back to sequential processing...") if args.verbose else None
            results = [_parse_midi_file(f) for f in files]
        DATA = [r for r in results if r is not None]
        failed = len(files) - len(DATA)
        if failed:
            print(f"Skipped {failed} files due to errors.") if args.verbose else None
        print("Done!") if args.verbose else None
    else:
        # when loading pre-augmented data, load the tensor directly
        print("Loading augmented data from file...") if args.verbose else None
        DATA = torch.load(args.source, weights_only=True)
        print("Done!") if args.verbose else None

    # randomly sample endings
    if len(DATA) > 0:
        print("Randomly sampling and cutting data to length...") if args.verbose else None
        DATA = sample_data(DATA, lth=args.length) + sample_end_data(DATA, lth=args.length)
        print("Done!") if args.verbose else None

    # augment data
    if not args.from_augmented_data and len(DATA) > 0:
        print("Augmenting data (NOTE: may take even longer)...") if args.verbose else None
        DATA = aug(DATA, note_shifts=args.transpositions, time_stretches=args.time_stretches,
                   verbose=args.verbose)
        print("Done!") if args.verbose else None

    if len(DATA) == 0:
        print("Error: no data to process. Check your source directory.")
        exit(1)

    # Ensure DATA is a 2D tensor (handle --from-augmented-data list case)
    if isinstance(DATA, list):
        print("Padding sequences from augmented data...") if args.verbose else None
        DATA = [F.pad(F.pad(seq, (1, 0), value=start_token), (0, 1), value=end_token) for seq in DATA]
        DATA = torch.nn.utils.rnn.pad_sequence(DATA, padding_value=pad_token).transpose(-1, -2)
        print("Done!") if args.verbose else None

    # shuffle data
    DATA = DATA[torch.randperm(DATA.shape[0])]
    
    # save
    print("Saving...") if args.verbose else None
    torch.save(DATA, args.destination)
    print("Done!")
