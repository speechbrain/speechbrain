"""
Recipe to train kenlm ngram model.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder=/path/to/corpus (**/*.stm)

Authors
 * Adel Moumen 2024
 * Pierre Champion 2023
"""

import os
import sys
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import speechbrain.k2_integration as sbk2
from speechbrain.utils.data_utils import get_list_from_csv

logger = logging.getLogger(__name__)


def dataprep_lm_training(
    lm_dir,
    output_arpa,
    csv_files,
    external_lm_corpus,
    vocab_file,
    arpa_order=3,
    prune_level=[0, 1, 2],
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
    arpa_order : int
        Order of the arpa lm
    prune_level : List[int]
        The numbers must be non-decreasing and the last number will be extended to any higher order.
        For example, --prune 0 disables pruning (the default) while --prune 0 0 1 prunes singletons for orders three and higher.
        Please refer to https://kheafield.com/code/kenlm/estimation/ for more details.
    """
    column_text_key = "wrd"  # defined in librispeech_prepare.py
    lm_corpus = os.path.join(lm_dir, "lm_corpus.txt")
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
    prune_level = " ".join(map(str, prune_level))
    cmd = f"lmplz -o {arpa_order} --prune {prune_level} --limit_vocab_file {vocab_file} < {lm_corpus} | sed  '1,20s/<unk>/<UNK>/1' > {output_arpa}"
    logger.critical(
        f"RUN the following kenlm command to build a {arpa_order}-gram arpa LM (https://github.com/kpu/kenlm):"
    )
    logger.critical(f"$ {cmd}")
    sys.exit(0)


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

    # Dataset prep
    import stm_prepare

    # multi-gpu (ddp) save data preparation
    run_on_main(
        stm_prepare.prepare_stm,
        kwargs={
            "stm_directory": hparams["stm_directory"],
            "wav_directory": hparams["wav_directory"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_train_csv": hparams["merge_train_csv"].split(","),
            "train_csv": hparams["train_csv"],
            "skip_prep": hparams["skip_prep"],
            "new_word_on_apostrophe": hparams["for_token_type"] in ["char"],
        },
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
        external_lm_corpus=[],
        vocab_file=os.path.join(hparams["lang_dir"], "words.txt"),
        arpa_order=hparams["arpa_order"],
        prune_level=hparams["prune_level"],
    )
