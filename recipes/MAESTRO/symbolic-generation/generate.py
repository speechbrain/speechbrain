import sys
from speechbrain.pretrained.interfaces import SymbolicGeneration


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

    inferer.generateMIDI()
