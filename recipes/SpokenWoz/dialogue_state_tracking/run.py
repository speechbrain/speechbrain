#!/usr/bin/env python3
"""
Recipe to fine-tune and infer for cascade and/or end-to-end spoken Dialogue State Tracking (DST).
This model fine-tunes a T5 backbone model on a spoken DST dataset.

To run this recipe, do the following:
> python run.py hparams/train_spokenwoz[_with_whisper_enc].yaml --data_folder YOUR_DATA_FOLDER

Author
    * Lucas Druart 2024
"""

import sys
import torch
from speechbrain.dataio.dataio import read_audio
from transformers import AutoTokenizer
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.augment.time_domain import Resample
from hyperpyyaml import load_hyperpyyaml

from pathlib import Path
import os
import logging
import pandas as pd

from model import SpokenWozUnderstanding

logger = logging.getLogger(__name__)

def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration", key_max_value={"duration": hparams["avoid_if_longer_than"]})
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]}
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(key_max_value={"duration": hparams["avoid_if_longer_than"]})

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_datasets = {}
    for csv_file in hparams["valid_csv"]:
        name = Path(csv_file).stem
        valid_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        valid_datasets[name] = valid_datasets[name].filtered_sorted(
            sort_key="duration"
        )
    valid_data = valid_datasets[Path(hparams["valid_csv"][0]).stem]

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem

        if hparams["sorting_turns"]:
            # Sorting by turn nbr to always have the previous dialogue state already processed, 
            # default is ascending
            ordered_csv = csv_file.replace(".csv", "_sorted.csv")
            df = pd.read_csv(csv_file)
            df.sort_values(by="turnID", inplace=True)
            df.to_csv(ordered_csv, header=True, index=False)
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=ordered_csv, replacements={"data_root": data_folder}
            )
        else:
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_file, replacements={"data_root": data_folder}
            )
            test_datasets[name] = test_datasets[name].filtered_sorted(
                sort_key="duration"
            )
        hparams["valid_loader_kwargs"]["shuffle"] = False

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("id", "wav", "start", "end")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(id, wav, start, end):
        resampler = Resample(orig_freq=8000, new_freq=16000)
        sig = read_audio(wav)
        sig = sig.unsqueeze(0) # Must be B*T*C
        resampled = resampler(sig)
        # Fusing both channels
        resampled = torch.mean(resampled, dim=2)
        # Selecting the correct frames: start*2 bc resampled
        sig = torch.squeeze(resampled)[int(start)*2:int(end)*2]
        
        # sig = sb.dataio.dataio.read_audio(os.path.join(hparams["data_folder"], 'turns', os.path.basename(id) + '.wav'))
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("id", "previous_state", "agent", "user", "current_state")
    @sb.utils.data_pipeline.provides(
        "semantics", "semantics_tokens", "outputs", "outputs_tokens", "outputs_tokens_nobos"
    )
    def text_pipeline(ID, previous_state, agent, user, current_state):

        if hparams["gold_previous_state"]:
            state = previous_state
        else:
            dialogue_id = ID.split("/")[-1].split("_")[0]
            turn_id = int(ID.split("/")[-1].split("_")[1].replace("Turn-", ""))
            if turn_id > 1:
                assert(os.path.isfile(os.path.join(hparams["output_folder"], "last_turns", f'{dialogue_id}.txt')))
                with open(os.path.join(hparams["output_folder"], "last_turns", f'{dialogue_id}.txt'), "r") as last_turn:
                    for line in last_turn:
                        dialogue_last_turn = line.strip()
                state = dialogue_last_turn
            else:
                state = ""

        if "cascade" in hparams["version"]:
            semantics = f"[State] {state} [Agent] {agent} [User] {user}"
        elif hparams["version"] == "e2e":
            semantics = f"[State] {state}"
        else:
            raise KeyError('hparams attribute "version" should be set to "cascade_model" or "e2e".')
        yield semantics

        semantics_tokens = tokenizer.encode(semantics)
        semantics_tokens = torch.LongTensor(semantics_tokens)
        yield semantics_tokens

        outputs = current_state
        yield outputs

        # T5 uses the pad_token_id as bos token
        tokens_list = [tokenizer.pad_token_id] 
        tokens_list += tokenizer.encode(f'[State] {current_state}')
        
        yield torch.LongTensor(tokens_list[:-1])

        yield torch.LongTensor(tokens_list[1:])

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "semantics_tokens", "sig", "outputs_tokens", "outputs_tokens_nobos"],
    )

    return train_data, valid_data, test_datasets

if __name__=="__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not hparams["skip_prep"]:

        # Dataset preparation
        from spokenwoz_prepare_slu import prepare_spokenwoz

        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_spokenwoz,
            kwargs={
                "data_folder": hparams["data_folder"],
                "version": hparams["version"],
                "tr_splits": hparams["train_splits"],
                "dev_splits": hparams["dev_splits"],
                "te_splits": hparams["test_splits"],
                "save_folder": hparams["save_folder"],
                "merge_lst": hparams["train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams["skip_prep"],
                "select_n_sentences": hparams["select_n_sentences"],
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(hparams["t5_hub"])

    if hparams["gold_previous_state"]:
        hparams["sorting_turns"] = False
    else:
        hparams["sorting_turns"] = True

    # Datasets creation (tokenization)
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    slu_brain = SpokenWozUnderstanding(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )
    slu_brain.tokenizer = tokenizer

    if not hparams["inference"]:

        slu_brain.fit(
            slu_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_loader_kwargs"],
            valid_loader_kwargs=hparams["valid_loader_kwargs"]
        )

    else:
        # Testing
        for k in test_datasets.keys():
            if not hparams["gold_previous_state"]:
                # Storing the last dialog's turn prediction
                if not os.path.isdir(os.path.join(hparams["output_folder"], "last_turns")):
                    os.mkdir(os.path.join(hparams["output_folder"], "last_turns"))
                slu_brain.hparams.output_file = os.path.join(
                    hparams["pred_folder"], "{}_previous.csv".format(k)
                )
            else:
                slu_brain.hparams.output_file = os.path.join(
                    hparams["pred_folder"], "{}.csv".format(k)
                )
            if not os.path.isfile(slu_brain.hparams.output_file):
                slu_brain.evaluate(
                    test_datasets[k], test_loader_kwargs=hparams["valid_loader_kwargs"]
                    )
