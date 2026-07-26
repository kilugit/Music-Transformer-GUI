import torch
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
        file["hparams"] = hparams

    model = MusicTransformer(**file["hparams"]).to(device)
    model.load_state_dict(file["state_dict"], strict=False)

    if compile:
        model = torch.compile(model)

    model.eval()
    return model


def greedy_decode(model, inp, target_duration_ms, mode="categorical", temperature=1.0, k=None,
                  use_amp=True):
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

    torch.set_float32_matmul_precision("high")

    amp_ctx = get_amp_context(use_amp)

    accumulated_ms = 0
    time_shift_start = note_on_events + note_off_events + 1
    time_shift_end = note_on_events + note_off_events + time_shift_events

    past_key_values = None
    cache_len = 0

    try:
        with torch.no_grad(), amp_ctx:
            for step in range(100000):
                if past_key_values is None:
                    # First pass: process entire prompt, cache K/V for all positions
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

                logits_last /= temperature(step)

                if mode == "argmax":
                    prediction = logits_last.argmax()
                elif k is not None:
                    k_val = k(step)
                    top_k_preds = torch.topk(logits_last, k_val)
                    predicted_idx = torch.distributions.Categorical(logits=top_k_preds.values).sample()
                    prediction = top_k_preds.indices[predicted_idx]
                elif mode == "categorical":
                    prediction = torch.distributions.Categorical(logits=logits_last).sample()
                else:
                    raise ValueError("Invalid mode or top k passed in")

                if prediction == end_token:
                    if accumulated_ms >= target_duration_ms:
                        return inp.squeeze()
                    continue

                inp = torch.cat([inp, prediction.view(1, 1)], dim=-1)

                idx = prediction.item()
                if time_shift_start <= idx <= time_shift_end:
                    accumulated_ms += (idx - (note_on_events + note_off_events)) * DIV
                    if accumulated_ms >= target_duration_ms:
                        return inp.squeeze()

    except (KeyboardInterrupt, RuntimeError):
        pass

    return inp.squeeze()


def audiate(token_ids, save_path="gneurshk.mid", tempo=512820, verbose=False):
    if save_path.endswith(".midi"):
        save_path = save_path[:-1]
    elif save_path.endswith(".mid"):
        pass
    else:
        save_path += ".mid"

    print(f"Saving midi file at {save_path}...") if verbose else None
    mid = list_parser(index_list=token_ids, fname=save_path[:-4], tempo=tempo)
    mid.save(save_path)

    print("Done")
    return


def generate(model_, inp, duration, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None,
             tempo=512820, verbose=False, use_amp=True):
    print("Greedy decoding...") if verbose else None
    start = time.time()
    token_ids = greedy_decode(model=model_, inp=inp, target_duration_ms=int(duration * 1000),
                              mode=mode, temperature=temperature, k=k, use_amp=use_amp)
    end = time.time()
    print(f"Generated {len(token_ids)} tokens.", end=" ") if verbose else None
    print(f"Time taken: {round(end - start, 2)} secs.") if verbose else None

    return audiate(token_ids=token_ids, save_path=save_path, tempo=tempo, verbose=verbose)


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
    parser.add_argument("-t", "--temperature",
                        help="temperature for decode sampling; lower temperature, more sure the sampling, "
                             "higher temperature, more diverse the output; default: 1.0 (categorical sample of true "
                             "model output)",
                        type=float)
    parser.add_argument("-tm", "--tempo", help="approximate tempo of generated sample in BMP", type=check_positive)
    parser.add_argument("-i", "--midi-prompt", help="if specified, the program will "
                        "generate music that continues the input midi file", type=str)
    parser.add_argument("-it", "--midi-prompt-tokens", help="number of tokens to sample "
                        "from the midi-prompt input as a prefix to continue, if it has been specified", type=int)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")

    # Generation control
    parser.add_argument("-d", "--duration", help="duration in seconds", type=float, required=True)
    parser.add_argument("--no-amp", help="disable automatic mixed precision (uses full float32)", action="store_true")

    args = parser.parse_args()

    temperature_ = float(args.temperature) if args.temperature else 1.0
    mode_ = args.mode if args.mode else "categorical"
    k_ = int(args.top_k) if args.top_k else None
    tempo_ = int(60 * 1e6 / int(args.tempo)) if args.tempo else 512820

    if args.midi_prompt:
        midi_parser_output = midi_parser(args.midi_prompt)
        tempo_ = midi_parser_output[2]
        midi_input = (midi_parser_output[1])[0:args.midi_prompt_tokens] if args.midi_prompt_tokens else midi_parser_output[1]
    else:
      midi_input = ["<start>"]

    music_transformer = load_model(args.path_to_model, args.compile)
    generate(model_=music_transformer, inp=midi_input, duration=args.duration, save_path=args.save_path,
             temperature=temperature_, mode=mode_, k=k_, tempo=tempo_, verbose=args.verbose,
             use_amp=not args.no_amp)
