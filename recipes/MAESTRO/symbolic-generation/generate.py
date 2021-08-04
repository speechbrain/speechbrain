import sys
from speechbrain.pretrained.interfaces import SymbolicMusicGeneration


# Generation begins!
if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print(
            "Please provide the model path in the first argument which includes hyperparams.yaml"
        )

    # Load the model
    inferer = SymbolicMusicGeneration.from_hparams(source=model_path)

    inferer.generateMIDI("generated.MIDI", N=2000)
