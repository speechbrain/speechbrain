#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with timers-and-such.
The tokenizer converts semantics into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).

To run this recipe, do the following:
> python train.py hparams/tokenizer_bpe51.yaml


Authors
 * Abdel Heba 2021
 * Mirco Ravanelli 2021
 * Loren Lugosch 2021
"""

import sys

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing timers-and-such)
    from prepare import prepare_TAS  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_TAS,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "type": "direct",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()
