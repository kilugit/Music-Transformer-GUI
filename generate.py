import os

# AMD ROCm: enable fast AOTriton SDPA backend before importing torch
if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") is None:
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch.nn.functional as F
import mido
import time
import argparse
import signal

if hasattr(signal, 'SIGBREAK'):
    def _sigbreak_handler(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGBREAK, _sigbreak_handler)

from masking import *
from tokenizer import *
from vocabulary import *
from hparams import device, get_amp_context


def load_model(filepath, compile=False):
    from model import MusicTransformer
    from hparams import hparams

    file = torch.load(filepath, map_location=device, weights_only=True)
    if "hparams" not in file:
        file["hparams"] = hparams.copy()

    # Ensure backward compat with old checkpoints
    ckpt_hparams = file["hparams"]
    for key in ["use_swiglu", "use_qk_norm", "use_sdpa"]:
        if key not in ckpt_hparams:
            ckpt_hparams[key] = False

    model = MusicTransformer(**ckpt_hparams).to(device)
    state_dict = file.get("state_dict", file.get("model_state_dict"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys in checkpoint: {unexpected}")

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


def greedy_decode(model, inp, mode="categorical", temperature=1.0, k=None,
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

    past_key_values = None
    cache_len = 0
    past_ids_window = [] if repetition_penalty is not None else None
    tokens_since_reprocess = 0

    def _truncate_context():
        nonlocal inp, past_ids_window, past_key_values, cache_len, tokens_since_reprocess
        max_rel = model.max_rel_dist
        prefix_len = max_rel // 4
        inp = torch.cat([inp[:, :prefix_len], inp[:, -(max_rel - prefix_len):]], dim=-1)
        if past_ids_window is not None:
            past_ids_window = inp[0].tolist()[-500:]
        past_key_values = None
        cache_len = 0
        tokens_since_reprocess = 0

    try:
        with torch.no_grad(), amp_ctx:
            for step in range(max_steps):
                if (model.max_rel_dist > 0 and past_key_values is not None
                        and tokens_since_reprocess >= model.max_rel_dist):
                    _truncate_context()

                if past_key_values is None:
                    if (model.max_rel_dist > 0
                            and inp.shape[-1] > model.max_rel_dist):
                        _truncate_context()
                    logits, past_key_values = model(inp, mask=create_mask(inp, 4), past_key_values=None)
                    logits_last = logits[0, -1, :]
                    cache_len = inp.shape[-1]
                    tokens_since_reprocess = 0
                    if past_ids_window is not None:
                        past_ids_window = inp[0].tolist()[-500:]
                else:
                    new_token = inp[:, -1:]
                    mask = create_cache_mask(new_token, cache_len, 4)
                    logits, past_key_values = model(new_token, mask=mask, past_key_values=past_key_values)
                    logits_last = logits[0, -1, :]
                    cache_len += 1
                    tokens_since_reprocess += 1

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

    except RuntimeError as e:
        print(f"ERROR during generation: {e}")
        raise
    except KeyboardInterrupt:
        print("Generation interrupted.")
    return inp.squeeze(0)


def beam_search_decode(model, inp, beam_width=3, temperature=1.0,
                        use_amp=True, repetition_penalty=None, top_k=None, top_p=None,
                        max_steps=50000):
    inp = events_to_indices(inp)
    if inp[0] != start_token:
        inp = [start_token] + inp

    inp = torch.tensor(inp, dtype=torch.int64, device=device)

    torch.set_float32_matmul_precision("high")
    amp_ctx = get_amp_context(use_amp)

    beams = [(inp.unsqueeze(0), None, 0.0)]
    completed = []

    def _clone_kv(pkv):
        if pkv is None:
            return None
        return [(k.clone(), v.clone()) for k, v in pkv]

    try:
        with torch.no_grad(), amp_ctx:
            for step in range(max_steps):
                new_beams = []

                for seq, past_kv, score in beams:
                    if step == 0:
                        logits, new_pkv = model(seq, mask=create_mask(seq, 4), past_key_values=None)
                        logits_last = logits[0, -1, :] / temperature
                    else:
                        new_token = seq[:, -1:]
                        mask = create_cache_mask(new_token, seq.shape[-1] - 1, 4)
                        logits, new_pkv = model(new_token, mask=mask, past_key_values=past_kv)
                        logits_last = logits[0, -1, :] / temperature

                    if repetition_penalty is not None and repetition_penalty != 1.0:
                        past_ids = torch.tensor(list(set(seq[0].tolist()[-200:])), device=logits_last.device, dtype=torch.long)
                        past_ids = past_ids[past_ids < logits_last.shape[-1]]
                        if past_ids.numel() > 0:
                            past_logits = logits_last[past_ids]
                            scale = torch.where(past_logits < 0, repetition_penalty, 1.0 / repetition_penalty)
                            logits_last[past_ids] = past_logits * scale

                    logits_last = top_p_top_k_filtering(logits_last, top_k=top_k, top_p=top_p)

                    probs = F.softmax(logits_last, dim=-1)
                    top_k_val = min(beam_width, probs.shape[-1])
                    top_probs, top_indices = torch.topk(probs, top_k_val)

                    for bp, bi in zip(top_probs, top_indices):
                        new_seq = torch.cat([seq, bi.view(1, 1)], dim=-1)
                        new_score = score + torch.log(bp + 1e-10).item()

                        if bi.item() == end_token:
                            completed.append((new_seq, new_score))
                        else:
                            new_beams.append((new_seq, _clone_kv(new_pkv), new_score))

                if not new_beams:
                    break

                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

                if len(completed) >= beam_width:
                    break

            for seq, _, score in beams:
                completed.append((seq, score))

    except RuntimeError as e:
        print(f"ERROR during beam search: {e}")
        raise
    except KeyboardInterrupt:
        print("Beam search interrupted.")

    if not completed:
        return beams[0][0].squeeze(0) if beams else inp.squeeze(0)

    best_seq = max(completed, key=lambda x: x[1])[0]
    return best_seq.squeeze(0)


def calculate_duration_ms(token_indices):
    """Calculate total milliseconds elapsed from a list of token indices."""
    total_ms = 0
    for idx in token_indices:
        if note_on_events + note_off_events + 1 <= idx <= note_on_events + note_off_events + time_shift_events:
            total_ms += (idx - (note_on_events + note_off_events)) * DIV
    return total_ms


def audiate(token_ids, save_path="gneurshk.mid", tempo=DEFAULT_TEMPO,
            ticks_per_beat=DEFAULT_TPB, verbose=False, speed=1.0):
    save_path = os.path.splitext(save_path)[0] + ".mid"

    print(f"Saving midi file at {save_path}...") if verbose else None
    mid = list_parser(index_list=token_ids, fname=save_path[:-4], tempo=tempo,
                      ticks_per_beat=ticks_per_beat, speed=speed)
    mid.save(save_path)

    print("Done")
    return


def generate(model_, inp, num_segments, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None,
             tempo=DEFAULT_TEMPO, ticks_per_beat=DEFAULT_TPB, verbose=False, use_amp=True,
             top_p=None, repetition_penalty=None, beam_width=None, speed=1.0):
    print("Generating...") if verbose else None
    start = time.time()

    all_tokens = []
    current_prompt = inp
    seg = 0

    while seg < num_segments:
        if beam_width is not None and beam_width > 1:
            token_ids = beam_search_decode(model=model_, inp=current_prompt,
                                            beam_width=beam_width, temperature=temperature,
                                            use_amp=use_amp, repetition_penalty=repetition_penalty,
                                            top_k=k, top_p=top_p)
        else:
            token_ids = greedy_decode(model=model_, inp=current_prompt,
                                      mode=mode, temperature=temperature, k=k, use_amp=use_amp,
                                      top_p=top_p, repetition_penalty=repetition_penalty)

        tokens = token_ids.tolist()
        if tokens and tokens[0] == start_token:
            tokens = tokens[1:]
        if tokens and tokens[-1] == end_token:
            tokens = tokens[:-1]
        if not tokens:
            break

        if seg > 0:
            prompt_indices = events_to_indices(current_prompt)
            new_tokens = tokens[len(prompt_indices):] if len(tokens) > len(prompt_indices) else tokens
        else:
            new_tokens = tokens
        if not new_tokens:
            break

        all_tokens.extend(new_tokens)

        if verbose:
            print(f"  Segment {seg + 1}: +{len(new_tokens)} tokens")

        window_size = max(1, model_.max_rel_dist // 2)
        window = tokens[-window_size:] if len(tokens) > window_size else tokens
        current_prompt = indices_to_events(window)
        seg += 1

    all_tokens.append(end_token)
    end = time.time()
    if verbose:
        print(f"Generated {len(all_tokens)} tokens across {seg} segments.")
        print(f"Time taken: {round(end - start, 2)} secs.")

    return audiate(token_ids=torch.tensor(all_tokens), save_path=save_path, tempo=tempo,
                   ticks_per_beat=ticks_per_beat, verbose=verbose, speed=speed)


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
    parser.add_argument("-n", "--num-segments", help="number of segments to generate", type=check_positive, required=True)
    parser.add_argument("--speed", help="playback speed multiplier (e.g., 2.0 = 2x faster); default: 1.0",
                        type=float, default=1.0)
    parser.add_argument("--no-amp", help="disable automatic mixed precision", action="store_true")

    args = parser.parse_args()

    temperature_ = float(args.temperature) if args.temperature else 1.0
    mode_ = args.mode if args.mode else "categorical"
    k_ = int(args.top_k) if args.top_k else None
    p_ = float(args.top_p) if args.top_p else 0.9
    rp_ = float(args.repetition_penalty) if args.repetition_penalty else 1.05
    bw_ = int(args.beam_width) if args.beam_width else None
    tempo_ = int(60 * 1e6 / int(args.tempo)) if args.tempo else DEFAULT_TEMPO
    tpb_ = DEFAULT_TPB

    if args.midi_prompt:
        midi_parser_output = midi_parser(args.midi_prompt)
        tempo_ = midi_parser_output[2]
        tpb_ = midi_parser_output[3]
        midi_input = (midi_parser_output[1])[0:args.midi_prompt_tokens] if args.midi_prompt_tokens else midi_parser_output[1]
    else:
      midi_input = ["<start>"]

    music_transformer = load_model(args.path_to_model, args.compile)
    generate(model_=music_transformer, inp=midi_input, num_segments=args.num_segments, save_path=args.save_path,
             temperature=temperature_, mode=mode_, k=k_, top_p=p_, repetition_penalty=rp_,
             beam_width=bw_, tempo=tempo_, ticks_per_beat=tpb_, verbose=args.verbose,
             use_amp=not args.no_amp, speed=args.speed)
