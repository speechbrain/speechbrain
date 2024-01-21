"""
Recipe to train kenlm ngram model.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder=/path/to/LibriSpeech

Authors
 * Adel Moumen 2024
 * Pierre Champion 2023
"""

import os
import csv
import sys
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import speechbrain.k2_integration as sbk2
from speechbrain.utils.data_utils import (
    download_file,
    get_list_from_csv,
)
import gzip
import shutil

logger = logging.getLogger(__name__)

OPEN_SLR_11_LINK = "http://www.openslr.org/resources/11/"
OPEN_SLR_11_NGRAM_MODELs = [
    "3-gram.arpa.gz",
    "3-gram.pruned.1e-7.arpa.gz",
    "3-gram.pruned.3e-7.arpa.gz",
    "4-gram.arpa.gz",
]


def download_librispeech_lm_training_text(destination):
    """Download librispeech lm training and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    f = "librispeech-lm-norm.txt.gz"
    download_file_and_extract(
        OPEN_SLR_11_LINK + f, os.path.join(destination, f)
    )


def download_librispeech_vocab_text(destination):
    """Download librispeech vocab file and unpack it.

    Arguments
    ---------
    destination : str
        Place to put vocab file.
    """
    f = "librispeech-vocab.txt"
    download_file(OPEN_SLR_11_LINK + f, destination)


def download_openslr_librispeech_lm(destination, rescoring_lm=True):
    """Download openslr librispeech lm and unpack it.

    Arguments
    ---------
    destination : str
        Place to put lm.
    rescoring_lms : bool
        Also download bigger 4grams model
    """
    os.makedirs(destination, exist_ok=True)
    for f in OPEN_SLR_11_NGRAM_MODELs:
        if f.startswith("4") and not rescoring_lm:
            continue
        d = os.path.join(destination, f)
        download_file_and_extract(OPEN_SLR_11_LINK + f, d)


def download_sb_librispeech_lm(destination, rescoring_lm=True):
    """Download sb librispeech lm and unpack it.

    Arguments
    ---------
    destination : str
        Place to put lm.
    rescoring_lms : bool
        Also download bigger 4grams model
    """
    os.makedirs(destination, exist_ok=True)
    download_file(
        "https://www.dropbox.com/scl/fi/3fkkdlliavhveb5n3nsow/3gram_lm.arpa?rlkey=jgdrluppfut1pjminf3l3y106&dl=1",
        os.path.join(destination, "3-gram_sb.arpa"),
    )
    if rescoring_lm:
        download_file(
            "https://www.dropbox.com/scl/fi/roz46ee0ah2lvy5csno4z/4gram_lm.arpa?rlkey=2wt8ozb1mqgde9h9n9rp2yppz&dl=1",
            os.path.join(destination, "4-gram_sb.arpa"),
        )


def download_file_and_extract(link, destination):
    """Download link and unpack it.

    Arguments
    ---------
    link : str
        File to download
    destination : str
        Place to put result.
    """
    download_file(link, destination)
    out = destination.replace(".gz", "")
    if os.path.exists(out):
        logger.debug("Skipping, already downloaded")
        return
    with gzip.open(destination, "rb") as f_in:
        with open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def dataprep_lm_training(
    lm_dir, output_arpa, csv_files, external_lm_corpus, vocab_file
):
    """Prepare lm txt corpus file for lm training with kenlm (https://github.com/kpu/kenlm)
    Does nothing if output_arpa exists.
    Else display to the user how to use kenlm in command line, then exit
    (return code 1), the user has to run the command manually.
    Instruction on how to compile kenlm (lmplz binary) is available in the
    above link.

    Arguments
    ---------
    lm_dir : str
        Path to where to store txt corpus
    output_arpa : str
        File to write arpa lm
    csv_files : List[str]
        CSV files to use to increase lm txt corpus
    external_lm_corpus : List[str]
        (Big) text dataset corpus
    vocab_file : str
       N-grams that contain vocabulary items not in this file be pruned.
    """
    download_librispeech_lm_training_text(lm_dir)
    column_text_key = "wrd"  # defined in librispeech_prepare.py
    lm_corpus = os.path.join(lm_dir, "libri_lm_corpus.txt")
    line_seen = set()
    with open(lm_corpus, "w") as corpus:
        for file in csv_files:
            for line in get_list_from_csv(file, column_text_key):
                corpus.write(line + "\n")
                line_seen.add(line + "\n")
        for file in external_lm_corpus:
            with open(file) as f:
                for line in f:
                    if line not in line_seen:
                        corpus.write(line)
    cmd = f"lmplz -o 3 --prune 0 1 2 --limit_vocab_file {vocab_file} < {lm_corpus}| sed  '1,20s/<unk>/<UNK>/1' > {output_arpa}"
    logger.info(
        f"Running training with kenlm with: \t{cmd}\n"
    )
    os.system(cmd)

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    import librispeech_prepare

    # multi-gpu (ddp) save data preparation
    run_on_main(
        librispeech_prepare.prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Download the vocabulary file for librispeech
    download_librispeech_vocab_text(
        destination=hparams["vocab_file"]
    )

    # Create the lexicon.txt for k2
    run_on_main(
        sbk2.lexicon.prepare_char_lexicon,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "vocab_files": [hparams["vocab_file"]],
            "extra_csv_files": [hparams["output_folder"] + "/train.csv"]
            if not hparams["skip_prep"]
            else [],
            "add_word_boundary": hparams["add_word_boundary"],
        },
    )

    caching = (
        {"cache": False}
        if "caching" in hparams and hparams["caching"] is False
        else {}
    )

    # Create the lang directory for k2
    run_on_main(
        sbk2.prepare_lang.prepare_lang,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "sil_prob": hparams["sil_prob"],
            **caching,
        },
    )

    dataprep_lm_training(
        lm_dir=hparams["output_folder"],
        output_arpa=hparams["output_arpa"],
        csv_files=[hparams["train_csv"]],
        external_lm_corpus=[
            os.path.join(hparams["output_folder"], "librispeech-lm-norm.txt")
        ],
        vocab_file=os.path.join(hparams["lang_dir"], "words.txt"),
    )