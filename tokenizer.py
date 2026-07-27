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

import mido
import torch
from vocabulary import *

"""
Implementation of translators of MIDI files to and from the event-based
vocabulary representation of MIDI files according to Oore et al., 2018

NOTE: multi-track MIDI files are automatically merged into a single track by sorting
     all messages by absolute tick time before processing
"""


def _ticks_to_ms(ticks, tempo, ticks_per_beat):
    """Convert MIDI ticks to milliseconds using current tempo (µs/beat) and ticks_per_beat."""
    if ticks_per_beat <= 0 or tempo <= 0:
        return ticks  # fallback: assume 1 tick = 1ms
    return int(round(ticks * tempo / (ticks_per_beat * 1000.0)))


def midi_parser(fname=None, mid=None):
    """
    Translates a multi-track midi file (merged into a single stream) into Oore et. al, 2018 vocabulary
    Multi-track files are properly combined by sorting all messages by their absolute tick time.
    Ticks are converted to real milliseconds using the file's tempo for tempo-consistent training.

    Args:
        fname (str): path to midi file to load OR
        mid (mido.MidiFile): loaded midi file

    Returns:
        index_list (torch.Tensor): list of indices in vocab
        event_list (list): list of events in vocab
        tempo (int): tempo in µs/beat (120 BPM = 500000)
        ticks_per_beat (int): MIDI ticks per quarter note
    """
    # take only one of fname or mid
    if not ((fname is None) ^ (mid is None)):
        raise ValueError("Input one of fname or mid, not both or neither")

    # load midi file
    if fname is not None:
        try:
            mid = mido.MidiFile(fname)
        except mido.midifiles.meta.KeySignatureError as e:
            raise ValueError(e)

    ticks_per_beat = mid.ticks_per_beat if mid.ticks_per_beat > 0 else DEFAULT_TPB

    # Merge multi-track messages into a single stream sorted by absolute tick time
    abs_messages = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            abs_messages.append((abs_tick, msg))

    # Sort by absolute tick time for proper temporal ordering across tracks
    abs_messages.sort(key=lambda x: x[0])

    # Convert back to delta-time-based message sequence
    prev_tick = 0
    merged_messages = []
    for abs_tick, msg in abs_messages:
        delta = abs_tick - prev_tick
        merged_messages.append(msg.copy(time=delta))
        prev_tick = abs_tick

    # things needed for conversion
    delta_ticks = 0         # accumulated ticks between significant messages
    event_list = []         # list of events in vocab
    index_list = []         # list of indices in vocab
    pedal_events = {}       # dict to handle pedal events
    pedal_flag = False      # flag to handle pedal events

    tempo = DEFAULT_TEMPO   # current tempo in µs/beat (updates on set_tempo)

    def _flush_time():
        """Convert accumulated ticks to ms and emit time shift events, then reset."""
        nonlocal delta_ticks
        if delta_ticks > 0:
            delta_ms = _ticks_to_ms(delta_ticks, tempo, ticks_per_beat)
            time_to_events(delta_ms, event_list=event_list, index_list=index_list)
            delta_ticks = 0

    # translate midi file to event list
    for msg in merged_messages:
        # increase delta_ticks by msg time for all messages
        delta_ticks += msg.time

        # meta events
        if msg.is_meta:
            if msg.type == "set_tempo":
                new_tempo = msg.tempo
                if new_tempo != tempo:
                    _flush_time()  # flush at old tempo before changing
                    tempo = new_tempo
            continue

        # process by message type
        t = msg.type
        vel = 0   # velocity

        if t == "note_on":  # key pressed
            # +1 or-1 everywhere accounts for <pad> token
            idx = msg.note + 1  # idx in vocab to help appending to output lists

            # get velocity to append after time events
            vel = velocity_to_bin(msg.velocity)

        elif t == "note_off":  # key released
            note = msg.note

            # if note_off while pedal down, add to pedal_events
            if pedal_flag:
                if note not in pedal_events:
                    pedal_events[note] = 0
                pedal_events[note] += 1
                # to prevent adding more events to output lists, continue
                continue
            else:  # else get idx to append to output lists
                idx = note_on_events + note + 1
        # if pedal on or off and pedal_events is not empty
        elif t == "control_change":
            if msg.control == 64:
                if msg.value >= 64:
                    # pedal down
                    pedal_flag = True
                else:
                    # lift pedal — clear flag regardless of pending notes
                    pedal_flag = False
                if pedal_events:
                    _flush_time()
                    # perform note_offs that occurred when pedal was down now that pedal is up
                    for note in pedal_events:
                        idx = note_on_events + note + 1
                        for i in range(pedal_events[note]):
                            event_list.append(vocab[idx])
                            index_list.append(idx)
                    pedal_events = {}
            continue
        # if it's not a type of msg we care about, continue to avoid adding to output lists
        else:
            continue

        # flush accumulated ticks as time shift events (converted to ms)
        _flush_time()

        # append velocity if note_on
        if t == "note_on":
            event_list.append(vocab[note_on_events + note_off_events + time_shift_events + vel])
            index_list.append(note_on_events + note_off_events + time_shift_events + vel)
        # append event and idx note events
        event_list.append(vocab[idx])
        index_list.append(idx)

    # return the lists of events
    return torch.tensor(index_list, dtype=torch.long), event_list, tempo, ticks_per_beat


def _ms_to_ticks(ms, tempo, ticks_per_beat):
    """Convert milliseconds to MIDI ticks using tempo (µs/beat) and ticks_per_beat."""
    if ticks_per_beat <= 0 or tempo <= 0:
        return ms  # fallback: assume 1 tick = 1ms
    return int(round(ms * ticks_per_beat * 1000.0 / tempo))


def list_parser(index_list=None, event_list=None, fname="bloop", tempo=DEFAULT_TEMPO,
                ticks_per_beat=DEFAULT_TPB, speed=1.0):
    """
    Translates a set of events or indices in the Oore et. al, 2018 vocabulary into a midi file

    Args:
        index_list (list or torch.Tensor): list of indices in vocab OR
        event_list (list): list of events in vocab
        fname (str, optional): name for single track of midi file returned
        tempo (int, optional): tempo of midi file returned in µs / beat,
                               tempo_in_µs_per_beat = 60 * 10e6 / tempo_in_bpm
        ticks_per_beat (int, optional): MIDI ticks per quarter note
        speed (float, optional): playback speed multiplier; >1 = faster, <1 = slower

    Returns:
        mid (mido.MidiFile): single-track piano midi file translated from vocab
                             NOTE: mid IS NOT SAVED BY THIS FUNCTION, IT IS ONLY RETURNED
    """
    # take only one of event_list or index_list to translate
    if not ((index_list is None) ^ (event_list is None)):
        raise ValueError("Input one of index_list or event_list, not both or neither")

    if index_list is not None:
        if isinstance(index_list, torch.Tensor):
            if index_list.dtype not in (torch.int32, torch.int64):
                raise ValueError("All indices in index_list must be int type")
        elif not all(isinstance(i, int) for i in index_list):
            raise ValueError("All indices in index_list must be int type")

    if event_list is not None:
        if not all(isinstance(i, str) for i in event_list):
            raise ValueError("All events in event_list must be str type")
        index_list = events_to_indices(event_list)

    # set up midi file
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()

    # meta messages; meta time is 0 everywhere to prevent delay in playing notes
    meta_track.append(mido.MetaMessage("track_name").copy(name=fname, time=0))
    meta_track.append(mido.MetaMessage("smpte_offset"))
    # assume time_signature is 4/4
    time_sig = mido.MetaMessage("time_signature")
    time_sig = time_sig.copy(numerator=4, denominator=4, time=0)
    meta_track.append(time_sig)
    # assume key_signature is C
    key_sig = mido.MetaMessage("key_signature", time=0)
    meta_track.append(key_sig)
    # assume tempo is constant at input tempo
    set_tempo = mido.MetaMessage("set_tempo")
    set_tempo = set_tempo.copy(tempo=tempo, time=0)
    meta_track.append(set_tempo)
    # end of meta track
    end = mido.MetaMessage("end_of_track").copy(time=0)
    meta_track.append(end)

    # set up the piano; default channel is 0 everywhere; program=0 -> piano
    program = mido.Message("program_change", channel=0, program=0, time=0)
    track.append(program)
    # dummy pedal off message; control should be < 64
    cc = mido.Message("control_change", time=0)
    track.append(cc)

    # things needed for conversion
    delta_ticks = 0
    vel = 0

    for idx in index_list:
        idx = idx.item() if isinstance(idx, torch.Tensor) else idx
        # if pad token, continue
        if idx <= 0:
            continue
        # adjust idx to ignore pad token
        idx = idx - 1

        # note messages
        if 0 <= idx < note_on_events + note_off_events:
            # note on event
            if 0 <= idx < note_on_events:
                note = idx
                t = "note_on"
                v = vel  # get velocity from previous event
            # note off event
            else:
                note = idx - note_on_events
                t = "note_off"
                v = 127

            # create note message and append to track
            msg = mido.Message(t)
            msg = msg.copy(note=note, velocity=v, time=delta_ticks)
            track.append(msg)

            # reinitialize delta_ticks and velocity to handle subsequent notes
            delta_ticks = 0
            vel = 0

        # time shift event
        elif note_on_events + note_off_events <= idx < note_on_events + note_off_events + time_shift_events:
            # find cut time in range (1, time_shift_events)
            cut_time = idx - (note_on_events + note_off_events - 1)
            # each time shift represents DIV ms; convert ms to ticks at current tempo
            ms = cut_time * DIV / speed
            delta_ticks += _ms_to_ticks(ms, tempo, ticks_per_beat)

        # velocity event
        elif note_on_events + note_off_events + time_shift_events <= idx < total_midi_events:
            # get velocity for next note_on in range (0, 127)
            vel = bin_to_velocity(idx - (note_on_events + note_off_events + time_shift_events))

    # end the track
    end = mido.MetaMessage("end_of_track").copy(time=0)
    track.append(end)

    # append finalized track and return midi file
    mid.tracks.append(meta_track)
    mid.tracks.append(track)
    return mid
