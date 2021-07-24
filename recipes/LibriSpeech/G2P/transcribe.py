"""A convenience script to transcribe text into phonemes using a
pretrained model

Authors
* Artem Ploujnikov 2021
"""
import itertools
import math
import os
import sys
import speechbrain as sb
from argparse import ArgumentParser
from speechbrain.pretrained.interfaces import GraphemeToPhoneme
from tqdm.auto import tqdm

MSG_MODEL_NOT_FOUND = "Model path not found"
MSG_HPARAMS_NOT_FILE = "Hyperparameters file not found"


def transcribe_text(g2p, text):
    """
    Transcribes a single line of text and outputs it

    Arguments
    ---------
    g2p: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance

    text: str
        the text to transcribe
    """
    output = g2p(text)
    print(" ".join(output))


def transcribe_file(g2p, text_file_name,  output_file_name=None, batch_size=64):
    """
    Transcribes a file with one example per line

    g2p: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance

    text_file_name: str
        the name of the source text file

    output_file_name: str
        the name of the output file. If omitted, the phonemes will
        be output to stdout

    batch_size: str
        the number of examples per batch


    """
    line_count = get_line_count(text_file_name)
    with open(text_file_name) as text_file:
        if output_file_name is None:
            transcribe_stream(g2p, text_file, sys.stdout, batch_size,
                              total=line_count)
        else:
            with open(output_file_name, "w") as output_file:
                transcribe_stream(g2p, text_file, output_file, batch_size,
                                  total=line_count)


def get_line_count(text_file_name):
    """
    Counts the lines in a file (without loading the entire file into memory)

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    line_count: int
        the number of lines in the file
    """
    with open(text_file_name) as text_file:
        return sum(1 for _ in text_file)

_substitutions = {" ": "<spc>"}

def transcribe_stream(g2p, text_file, output_file, batch_size=64, total=None):
    """
    Transcribes a file stream

    Arguments
    ---------
    g2p: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance
    text_file: file
        a file object from which text samples will be read
    output_file: file
        the file object to which phonemes will be output
    batch_size: 64
        the size of the batch passed to the model
    total: int
        the total number of examples (used for the progress bar)
    """
    batch_count = math.ceil(total // batch_size)
    for batch in tqdm(chunked(text_file, batch_size), total=batch_count):
        phoneme_results = g2p(batch)
        for result in phoneme_results:
            line = " ".join(
                _substitutions.get(phoneme, phoneme) for phoneme in result)
            print(line, file=output_file)
            output_file.flush()


def chunked(iterable, batch_size):
    """Break *iterable* into lists of length *n*:

        >>> list(chunked([1, 2, 3, 4, 5, 6], 3))
        [[1, 2, 3], [4, 5, 6]]

    By the default, the last yielded list will have fewer than *n* elements
    if the length of *iterable* is not divisible by *n*:

        >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
        [[1, 2, 3], [4, 5, 6], [7, 8]]


    Adopted and simplified from more-itertools
    https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#chunked

    Arguments
    ---------
    iterable: iterable
        any iterable of individual samples

    batch_size: int
        the size of each chunk

    Returns
    -------
    batched_iterable: iterable
        an itearble of batches

    """
    iterable = iter(iterable)
    iterator = iter(lambda: list(itertools.islice(iterable, batch_size)), [])
    return iterator

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
    parser.add_argument("--text", help="The text to transcribe")
    parser.add_argument("--text-file", help="the text file to transcribe")
    parser.add_argument("--output-file", help="")
    arguments, override_arguments = parser.parse_known_args()
    _, run_opts, overrides = sb.parse_arguments([arguments.hparams] + override_arguments)

    if not os.path.isdir(arguments.model):
        print(MSG_MODEL_NOT_FOUND, file=sys.stderr)
        sys.exit(1)
    hparams_file_name = os.path.join(arguments.model, arguments.hparams)
    if not os.path.isfile(hparams_file_name):
        print(MSG_HPARAMS_NOT_FILE, file=sys.stderr)
        sys.exit(1)

    g2p = GraphemeToPhoneme.from_hparams(
        hparams_file=arguments.hparams, source=arguments.model,
        overrides=overrides,
        run_opts=run_opts
    )
    if getattr(g2p.hparams, "use_language_model", False):
        g2p.hparams.beam_searcher = g2p.hparams.beam_searcher_lm
    if arguments.text:
        transcribe_text(g2p, arguments.text)
    elif arguments.text_file:
        transcribe_file(
            g2p=g2p,
            text_file_name=arguments.text_file,
            output_file_name=arguments.output_file,
            batch_size=g2p.hparams.eval_batch_size
        )
    else:
        print("ERROR: Either --text or --text-file is required", file=sys.stderr)


if __name__ == "__main__":
    main()
