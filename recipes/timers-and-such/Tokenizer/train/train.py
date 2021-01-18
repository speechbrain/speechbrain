#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with timers-and-such.
The tokenizer coverts semantics into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).

To run this recipe, do the following:
> python train.py hyperparams/tokenizer_bpe51.yaml


Authors
 * Abdel Heba 2021
 * Mirco Ravanelli 2021
 * Loren Lugosch 2021
"""

import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing timers-and-such)
    from prepare import prepare_TAS  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_TAS,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_splits": hparams["train_splits"],
            "type": "direct",
        },
    )

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Train tokenizer
    hparams["tokenizer"]()
