#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with Tedlium2.
The tokenizer converts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.

To run this recipe, do the following:
> python train.py hyperparams/tedlium2_500_bpe.yaml

Authors
 * Shucong Zhang 2023
"""

import shutil
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

    # 1.  # Dataset prep (parsing Tedlium2)
    from tedlium2_prepare import prepare_tedlium2  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_tedlium2,
        kwargs={
            "data_folder": hparams["data_folder"],
            "utt_save_folder": hparams["clipped_utt_folder"],
            "csv_save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
            "avoid_if_shorter_than": hparams["avoid_if_shorter_than"],
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()

    output_path = hparams["output_folder"]

    token_output = hparams["token_output"]
    token_type = hparams["token_type"]
    bpe_model = f"{output_path}/{token_output}_{token_type}.model"

    tokenizer_ckpt = f"{output_path}/tokenizer.ckpt"
    shutil.copyfile(bpe_model, tokenizer_ckpt)
