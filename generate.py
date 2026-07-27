import os

# AMD ROCm: enable fast AOTriton SDPA backend before importing torch
if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") is None:
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch.nn.functional as F
import mido
import time
import argparse
from masking import *
from tokenizer import *
from vocabulary import *
from hparams import device, get_amp_context


def load_model(filepath, compile=False):
    from model import MusicTransformer
    from hparams import hparams, device

    file = torch.load(filepath, map_location=device)
    if "hparams" not in file:
        file["hparams"] = hparams.copy()

    # Ensure backward compat with old checkpoints
    ckpt_hparams = file["hparams"]
    for key in ["use_swiglu", "use_qk_norm", "use_sdpa"]:
        if key not in ckpt_hparams:
            ckpt_hparams[key] = False

    model = MusicTransformer(**ckpt_hparams).to(device)
    model.load_state_dict(file["state_dict"], strict=False)

    if compile:
        model = torch.compile(model)

    model.eval()
    return model


def top_p_top_k_filtering(logits, top_k=None, top_p=None, min_tokens_to_keep=1):
    """
    Apply top-k and/or top-p (nucleus) filtering to logits.
    """
    dtype = logits.dtype
    neg_inf = torch.tensor(-float('Inf'), device=logits.device, dtype=dtype)

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        top_k_values, _ = torch.topk(logits, top_k)
        logits[logits < top_k_values[..., -1, None]] = neg_inf

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 0:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = neg_inf

    return logits


def sample_from_logits(logits, mode="categorical", temperature=1.0, top_k=None, top_p=None,
                        repetition_penalty=None, past_token_ids=None):
    logits = logits.clone()

    # Apply temperature
    if temperature > 0:
        logits = logits / temperature

    # Apply repetition penalty (based on the paper "Repetition Penalty" from CTRL)
    if repetition_penalty is not None and repetition_penalty != 1.0 and past_token_ids is not None:
        past_tokens = torch.tensor(list(set(past_token_ids)), device=logits.device, dtype=torch.long)
        past_tokens = past_tokens[past_tokens < logits.shape[-1]]
        if past_tokens.numel() > 0:
            past_logits = logits[past_tokens]
            dtype = logits.dtype
            rp_t = torch.tensor(repetition_penalty, device=logits.device, dtype=dtype)
            inv_rp_t = torch.tensor(1.0 / repetition_penalty, device=logits.device, dtype=dtype)
            scale = torch.where(past_logits < 0, rp_t, inv_rp_t)
            logits[past_tokens] = (past_logits * scale).to(dtype)

    # Apply top-k and top-p filtering
    logits = top_p_top_k_filtering(logits, top_k=top_k, top_p=top_p)

    if mode == "argmax":
        return logits.argmax()

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def greedy_decode(model, inp, target_duration_ms, mode="categorical", temperature=1.0, k=None,
                  use_amp=True, top_p=None, repetition_penalty=None, max_steps=50000):
    inp = events_to_indices(inp)

    if inp[0] != start_token:
        inp = [start_token] + inp

    inp = torch.tensor(inp, dtype=torch.int64, device=device)
    inp = inp.unsqueeze(0)

    if not callable(temperature):
        _temperature = float(temperature)
        temperature = lambda _: _temperature

    if k is not None and not callable(k):
        _k = int(k)
        k = lambda _: _k

    if top_p is not None and not callable(top_p):
        _p = float(top_p)
        top_p = lambda _: _p

    torch.set_float32_matmul_precision("high")

    amp_ctx = get_amp_context(use_amp)

    accumulated_ms = 0
    time_shift_start = note_on_events + note_off_events + 1
    time_shift_end = note_on_events + note_off_events + time_shift_events

    past_key_values = None
    cache_len = 0
    # For repetition penalty: maintain a sliding window of recent tokens
    past_ids_window = [] if repetition_penalty is not None else None

    def _truncate_context():
        nonlocal inp, accumulated_ms, past_ids_window, past_key_values, cache_len
        max_rel = model.max_rel_dist
        prefix_len = max_rel // 4
        suffix_len = max_rel - prefix_len
        inp = torch.cat([inp[:, :prefix_len], inp[:, -suffix_len:]], dim=-1)
        accumulated_ms = 0
        for t in inp[0]:
            idx = t.item()
            if time_shift_start <= idx <= time_shift_end:
                accumulated_ms += (idx - (note_on_events + note_off_events)) * DIV
        if past_ids_window is not None:
            past_ids_window = inp[0].tolist()[-500:]
        past_key_values = None
        cache_len = 0

    try:
        with torch.no_grad(), amp_ctx:
            for step in range(max_steps):
                # Slide context window when it exceeds max_rel_dist
                if (model.max_rel_dist > 0 and past_key_values is not None
                        and cache_len >= model.max_rel_dist):
                    _truncate_context()

                if past_key_values is None:
                    # First pass: process entire prompt, cache K/V for all positions
                    logits, past_key_values = model(inp, mask=create_mask(inp, 4), past_key_values=None)
                    logits_last = logits[0, -1, :]
                    cache_len = inp.shape[-1]
                    if past_ids_window is not None:
                        past_ids_window = inp[0].tolist()[-500:]
                    # Truncate prompt if it alone exceeds max_rel_dist
                    if (model.max_rel_dist > 0 and past_key_values is not None
                            and cache_len >= model.max_rel_dist):
                        _truncate_context()
                        # Re-run first pass on truncated context
                        logits, past_key_values = model(inp, mask=create_mask(inp, 4), past_key_values=None)
                        logits_last = logits[0, -1, :]
                        cache_len = inp.shape[-1]
                else:
                    # KV cache step: inp[-1] is the token just generated (not yet cached).
                    # cache_len tracks how many positions are already in the cache.
                    new_token = inp[:, -1:]
                    mask = create_cache_mask(new_token, cache_len, 4)
                    logits, past_key_values = model(new_token, mask=mask, past_key_values=past_key_values)
                    logits_last = logits[0, -1, :]
                    cache_len += 1

                # Sample next token with all enhancements
                prediction = sample_from_logits(
                    logits_last,
                    mode=mode if k is None and top_p is None else "categorical",
                    temperature=temperature(step),
                    top_k=k(step) if k is not None else None,
                    top_p=top_p(step) if top_p is not None else None,
                    repetition_penalty=repetition_penalty,
                    past_token_ids=past_ids_window,
                )
                prediction = prediction.squeeze()

                inp = torch.cat([inp, prediction.view(1, 1)], dim=-1)
                if past_ids_window is not None:
                    past_ids_window.append(prediction.item())
                    if len(past_ids_window) > 500:
                        past_ids_window.pop(0)

                if prediction == end_token:
                    return inp.squeeze(0)

                idx = prediction.item()
                if time_shift_start <= idx <= time_shift_end:
                    accumulated_ms += (idx - (note_on_events + note_off_events)) * DIV

    except (KeyboardInterrupt, RuntimeError):
        pass
    return inp.squeeze(0)


def beam_search_decode(model, inp, target_duration_ms, beam_width=3, temperature=1.0,
                        use_amp=True, repetition_penalty=None, max_steps=50000):
    """
    Beam search decoding for higher quality generation.
    Maintains beam_width parallel sequences and picks the best one.
    """
    inp = events_to_indices(inp)
    if inp[0] != start_token:
        inp = [start_token] + inp

    inp = torch.tensor(inp, dtype=torch.int64, device=device)
    prompt_len = inp.shape[0]

    torch.set_float32_matmul_precision("high")
    amp_ctx = get_amp_context(use_amp)

    time_shift_start = note_on_events + note_off_events + 1
    time_shift_end = note_on_events + note_off_events + time_shift_events

    # Initialize beams: each is (sequence, past_key_values, score, accumulated_ms)
    beams = [(inp.unsqueeze(0), None, 0.0, 0)]
    completed = []

    try:
        with torch.no_grad(), amp_ctx:
            for step in range(max_steps):
                new_beams = []

                for seq, past_kv, score, acc_ms in beams:
                    if step == 0:
                        # First pass: process entire prompt
                        logits, new_pkv = model(seq, mask=create_mask(seq, 4), past_key_values=None)
                        logits_last = logits[0, -1, :] / temperature
                    else:
                        new_token = seq[:, -1:]
                        mask = create_cache_mask(new_token, seq.shape[-1] - 1, 4)
                        logits, new_pkv = model(new_token, mask=mask, past_key_values=past_kv)
                        logits_last = logits[0, -1, :] / temperature

                    # Apply repetition penalty
                    if repetition_penalty is not None and repetition_penalty != 1.0:
                        past_ids = torch.tensor(list(set(seq[0].tolist()[-200:])), device=logits_last.device, dtype=torch.long)
                        past_ids = past_ids[past_ids < logits_last.shape[-1]]
                        if past_ids.numel() > 0:
                            past_logits = logits_last[past_ids]
                            scale = torch.where(past_logits < 0, repetition_penalty, 1.0 / repetition_penalty)
                            logits_last[past_ids] = past_logits * scale

                    # Get top beam_width candidates
                    probs = F.softmax(logits_last, dim=-1)
                    top_k = min(beam_width, probs.shape[-1])
                    top_probs, top_indices = torch.topk(probs, top_k)

                    for bp, bi in zip(top_probs, top_indices):
                        new_seq = torch.cat([seq, bi.view(1, 1)], dim=-1)
                        new_score = score + torch.log(bp + 1e-10).item()
                        new_acc_ms = acc_ms

                        # Track accumulated time
                        idx = bi.item()
                        if time_shift_start <= idx <= time_shift_end:
                            new_acc_ms += (idx - (note_on_events + note_off_events)) * DIV

                        if idx == end_token:
                            completed.append((new_seq, new_score))
                        elif new_acc_ms >= target_duration_ms:
                            completed.append((new_seq, new_score))
                        else:
                            new_beams.append((new_seq, new_pkv, new_score, new_acc_ms))

                # Sort by score (higher is better) and keep top beams
                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

                if not beams:
                    break

                # Early stopping if we have enough completed sequences
                if len(completed) >= beam_width:
                    break

            # Add remaining beams as completed
            for seq, _, score, _ in beams:
                completed.append((seq, score))

    except (KeyboardInterrupt, RuntimeError):
        pass

    if not completed:
        return beams[-1][0].squeeze(0) if beams else inp.squeeze(0)

    # Return the best completed sequence (highest score = lowest negative log-likelihood)
    best_seq = max(completed, key=lambda x: x[1])[0]
    return best_seq.squeeze(0)


def calculate_duration_ms(token_indices):
    """Calculate total milliseconds elapsed from a list of token indices."""
    total_ms = 0
    for idx in token_indices:
        if note_on_events + note_off_events + 1 <= idx <= note_on_events + note_off_events + time_shift_events:
            total_ms += (idx - (note_on_events + note_off_events)) * DIV
    return total_ms


def audiate(token_ids, save_path="gneurshk.mid", tempo=512820, verbose=False):
    save_path = os.path.splitext(save_path)[0] + ".mid"

    print(f"Saving midi file at {save_path}...") if verbose else None
    mid = list_parser(index_list=token_ids, fname=save_path[:-4], tempo=tempo)
    mid.save(save_path)

    print("Done")
    return


def generate(model_, inp, duration, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None,
             tempo=512820, verbose=False, use_amp=True, top_p=None, repetition_penalty=None,
             beam_width=None):
    print("Generating...") if verbose else None
    start = time.time()
    target_ms = int(duration * 1000)

    all_tokens = []
    total_ms = 0
    current_prompt = inp
    seg = 0
    max_segments = 100

    while total_ms < target_ms and seg < max_segments:
        if beam_width is not None and beam_width > 1:
            token_ids = beam_search_decode(model=model_, inp=current_prompt, target_duration_ms=target_ms,
                                            beam_width=beam_width, temperature=temperature,
                                            use_amp=use_amp, repetition_penalty=repetition_penalty)
        else:
            token_ids = greedy_decode(model=model_, inp=current_prompt, target_duration_ms=target_ms,
                                      mode=mode, temperature=temperature, k=k, use_amp=use_amp,
                                      top_p=top_p, repetition_penalty=repetition_penalty)

        tokens = token_ids.tolist()
        # Strip <start> and <end> tokens
        if tokens and tokens[0] == start_token:
            tokens = tokens[1:]
        if tokens and tokens[-1] == end_token:
            tokens = tokens[:-1]
        if not tokens:
            break

        # On continuation rounds, remove the prompt overlap
        if seg > 0:
            prompt_indices = events_to_indices(current_prompt)
            new_tokens = tokens[len(prompt_indices):] if len(tokens) > len(prompt_indices) else tokens
        else:
            new_tokens = tokens
        if not new_tokens:
            break

        all_tokens.extend(new_tokens)
        total_ms = calculate_duration_ms(all_tokens)

        if verbose:
            print(f"  Segment {seg + 1}: +{len(new_tokens)} tokens, total {total_ms/1000:.1f}s / {duration}s")

        if total_ms >= target_ms:
            break

        # Feed the tail of current output as the next prompt
        window = tokens[-512:] if len(tokens) > 512 else tokens
        current_prompt = indices_to_events(window)
        seg += 1

    all_tokens.append(end_token)
    end = time.time()
    if verbose:
        print(f"Generated {len(all_tokens)} tokens across {seg + 1} segments.")
        print(f"Time taken: {round(end - start, 2)} secs.")

    return audiate(token_ids=torch.tensor(all_tokens), save_path=save_path, tempo=tempo, verbose=verbose)


if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    def check_nonnegative_float(x):
        if x is None:
            return x
        x = float(x)
        if x < 0:
            raise argparse.ArgumentTypeError(f"{x} is not a non-negative float")
        return x

    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate midi audio with a Music Transformer!"
    )
    parser.add_argument("path_to_model", help="string path to a .pt file at which has been saved a dictionary "
                                              "containing the model state dict and hyperparameters", type=str)
    parser.add_argument("save_path", help="path at which to save the generated midi file", type=str)

    parser.add_argument("-c", "--compile", help="if true, model will be `torch.compile`d for potentially better "
                                                "speed; default: false", action="store_true")
    parser.add_argument("-m", "--mode", help="specify 'categorical' or 'argmax' mode of decode sampling", type=str)
    parser.add_argument("-k", "--top-k", help="top k samples to consider while decode sampling; default: all",
                        type=check_positive)
    parser.add_argument("-p", "--top-p", help="top-p (nucleus) sampling threshold; default: 0.9",
                        type=check_nonnegative_float)
    parser.add_argument("-rp", "--repetition-penalty",
                        help="repetition penalty factor (>1.0 discourages repetition, <1.0 encourages it); default: 1.0",
                        type=float)
    parser.add_argument("-t", "--temperature",
                        help="temperature for decode sampling; lower = more focused, higher = more diverse; default: 1.0",
                        type=float)
    parser.add_argument("-tm", "--tempo", help="approximate tempo of generated sample in BPM", type=check_positive)
    parser.add_argument("-i", "--midi-prompt", help="midi file to use as a prompt for continuation", type=str)
    parser.add_argument("-it", "--midi-prompt-tokens", help="number of tokens to sample from the prompt", type=int)
    parser.add_argument("-bw", "--beam-width", help="beam search width (>1 enables beam search); default: 1",
                        type=check_positive)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")

    # Generation control
    parser.add_argument("-d", "--duration", help="duration in seconds", type=float, required=True)
    parser.add_argument("--no-amp", help="disable automatic mixed precision", action="store_true")

    args = parser.parse_args()

    temperature_ = float(args.temperature) if args.temperature else 1.0
    mode_ = args.mode if args.mode else "categorical"
    k_ = int(args.top_k) if args.top_k else None
    p_ = float(args.top_p) if args.top_p else 0.9
    rp_ = float(args.repetition_penalty) if args.repetition_penalty else 1.05
    bw_ = int(args.beam_width) if args.beam_width else None
    tempo_ = int(60 * 1e6 / int(args.tempo)) if args.tempo else 512820

    if args.midi_prompt:
        midi_parser_output = midi_parser(args.midi_prompt)
        tempo_ = midi_parser_output[2]
        midi_input = (midi_parser_output[1])[0:args.midi_prompt_tokens] if args.midi_prompt_tokens else midi_parser_output[1]
    else:
      midi_input = ["<start>"]

    music_transformer = load_model(args.path_to_model, args.compile)
    generate(model_=music_transformer, inp=midi_input, duration=args.duration, save_path=args.save_path,
             temperature=temperature_, mode=mode_, k=k_, top_p=p_, repetition_penalty=rp_,
             beam_width=bw_, tempo=tempo_, verbose=args.verbose,
             use_amp=not args.no_amp)
