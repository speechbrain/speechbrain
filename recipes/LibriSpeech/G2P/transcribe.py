"""A convenience script to transcribe text into phonemes using a
pretrained model

Authors
* Artem Ploujnikov 2021
"""
import os
import sys
from argparse import ArgumentParser
from speechbrain.pretrained.interfaces import GraphemeToPhoneme


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
        '--text',
        help='The text to transcribe',
        required=True
    )
    arguments = parser.parse_args()
    if not os.path.isdir(arguments.model):
        print(MSG_MODEL_NOT_FOUND, file=sys.stderr)
        sys.exit(1)
    hparams_file_name = os.path.join(arguments.model, arguments.hparams)
    if not os.path.isfile(hparams_file_name):
        print(MSG_HPARAMS_NOT_FILE, file=sys.stderr)
        sys.exit(1)

    g2p = GraphemeToPhoneme.from_hparams(
        hparams_file=arguments.hparams, source=arguments.model)

    output = g2p(arguments.text)
    print(output)


if __name__ == "__main__":
    main()