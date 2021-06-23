import numpy as np
import torch
import sys
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import muspy as mp
import random


def generate_sequence(N):
    """Generates a sequence of N timesteps
    Arguments
    ---------
    N : int
        Number of sequences to generate using the model.
    Returns
    -------
    sequence: nd.array (N,128)
        binarized piano roll ready for MIDI generation
    """
    notes_len = hparams["emb_dim"]
    zeros = 85
    midi_len = 128

    # Initial input to the sequence is a randomized binary vector with 4 ones
    inp = np.array([0] * zeros + [1] * (notes_len - zeros))
    np.random.shuffle(inp)
    inp = torch.tensor(inp, dtype=torch.float32).to(run_opts["device"])
    inp = inp.view((1, notes_len))

    # Sequence to return for MIDI processing
    sequence = np.zeros((N, midi_len))

    # Recover best checkpoint for evaluation using loss min_key
    checkpointer = hparams["checkpointer"]
    if checkpointer is not None:
        checkpointer.recover_if_possible(
            min_key="loss", device=torch.device(run_opts["device"])
        )
    model = checkpointer.recoverables["model"]
    model = model.eval()

    # Generate N timesteps
    for i in range(N):
        if i == 0:
            out, h = model(inp)
        else:
            out, h = model(inp, h)

        out = torch.squeeze(out)

        # Convert probabilities to binary vector
        for j in range(len(out)):
            thresh = random.random()
            out[j] = (thresh < out[j] and out[j] > 0.05).type(torch.int32)

        # Store and pad vectors to match MIDI size
        sequence[i] = np.pad(
            out.cpu().detach().numpy(),
            (20, 20),
            "constant",
            constant_values=(0, 0),
        )
        inp = torch.unsqueeze(out, dim=0)

    return sequence


def scale_time(old_time):
    """Scales each time step in generated music
    Arguments
    ---------
    old_time : int
        Note time in 1 timestep.
    Returns
    -------
    scaled old_time: int
        input scaled by some factor
    """
    return old_time * hparams["time_scale"]


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    run_opts["device"] = "cpu"

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Generate sequence of specified length
    gen_notes = generate_sequence(hparams["sequence_length"])

    # Create Music object from binary piano roll
    music = mp.from_pianoroll_representation(
        gen_notes,
        resolution=hparams["resolution"],
        encode_velocity=False,
        default_velocity=hparams["velocity"],
    )

    # Increase duration of each note
    for note in music.tracks[0].notes:
        note.duration *= hparams["note_duration"]

    # Increate time of each note
    mp.adjust_time(music, scale_time)

    # Write to MIDI
    music.write(hparams["midi_file"])
