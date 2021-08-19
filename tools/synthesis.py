"""A wrapper for speech synthesis

Authors
* Artem Ploujnikov 2021
"""
import os.path
import torch
import sys
from argparse import ArgumentParser
from speechbrain.pretrained.interfaces import SpeechSynthesizer
from speechbrain.dataio.dataio import write_audio


DEFAULT_TEXT = "Please call Stella"
MSG_MODEL_NOT_FOUND = "Model path not found"
MSG_HPARAMS_NOT_FILE = "Hyperparameters file not found"


def main():
    parser = ArgumentParser(description="Command-line speech synthesizer")
    parser.add_argument(
        "--model", required=True, help="The path to the pretrained model"
    )
    parser.add_argument(
        "--hparams",
        help="The name of the hyperparameter file",
        default="hyperparams.yaml",
    )
    parser.add_argument(
        "--output", default="output.wav", help="The name of the output file"
    )
    parser.add_argument(
        "--device", help="The name of the device on which it will be run"
    )

    parser.add_argument("--text", default=DEFAULT_TEXT)
    arguments = parser.parse_args()
    if not os.path.isdir(arguments.model):
        print(MSG_MODEL_NOT_FOUND, file=sys.stderr)
        sys.exit(1)
    hparams_file_name = os.path.join(arguments.model, arguments.hparams)
    if not os.path.isfile(hparams_file_name):
        print(MSG_HPARAMS_NOT_FILE, file=sys.stderr)
        sys.exit(1)
    device = arguments.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    synthesizer = SpeechSynthesizer.from_hparams(
        hparams_file=arguments.hparams,
        source=arguments.model,
        run_opts={"device": device},
    )
    sample = synthesizer([arguments.text])
    sample = sample.squeeze().cpu()
    write_audio(
        filepath=arguments.output,
        audio=sample,
        samplerate=synthesizer.hparams.sample_rate,
    )


if __name__ == "__main__":
    main()
