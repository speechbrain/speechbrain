import sys
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import muspy as mp
from speechbrain.pretrained.interfaces import SymbolicGeneration


def generate_sequence(N):
    """ Call inference class that generates a sequence of N timesteps
    Arguments
    ---------
    N : int
        Number of sequences to generate using the model.
    Returns
    -------
    sequence: nd.array (N,128)
        binarized piano roll ready for MIDI generation
    """

    inferer = SymbolicGeneration.from_hparams(source="test/",)
    sequence = inferer.generate_timestep(N)

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
