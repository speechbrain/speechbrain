import sys
import muspy as mp
from speechbrain.pretrained.interfaces import SymbolicGeneration


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


# Generation begins!
if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print(
            "Please provide the model path in the first argument which includes hyperparams.yaml"
        )

    # Load the model
    inferer = SymbolicGeneration.from_hparams(source=model_path)

    # get the hyperparameter file
    hparams = inferer.hparams.__dict__

    # Generate sequence of specified length
    gen_notes = inferer.generate_timestep(hparams["sequence_length"])

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
