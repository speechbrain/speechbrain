#!/usr/bin/env python3
"""
Recipe for training a gpt_based response generation model with MultiWOZ.
The system employs GPT2 (https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf).
This recipe takes the GPT2LMHeadModel to fine-tune for the response generation task on the NLL.

To run this recipe, do the following:
> python train_with_gpt.py hparams/train_gpt.yaml

Authors
 * Pooneh Mousavi 2023
 * Simone Alghisi 2023
"""


import math
import sys
from itertools import chain

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main


class ResGenBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a gpt decoder."""
        # Get required data from batch
        batch = batch.to(self.device)
        input_ids, _ = batch.input_ids
        token_type_ids, _ = batch.token_type_ids

        # Forward Pass
        padding_mask = ~self.hparams.padding_mask(
            input_ids, pad_idx=tokenizer.unk_token_id
        )
        outputs = self.modules.gpt_model(
            input_ids, token_type_ids, padding_mask
        ).logits

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the NLL-loss using reply as label."""
        # Get required data from batch
        batch = batch.to(self.device)
        ids = batch.id
        lm_labels, labels_lens = batch.lm_labels
        history_bos, history_lens = batch.history_bos
        reply_eos, reply_lens = batch.reply_eos
        history_token_type, _ = batch.history_token_type

        loss = self.hparams.ce_loss(
            predictions.flatten(end_dim=-2), lm_labels.flatten()
        )

        if stage == sb.Stage.VALID:
            # hyps = None
            # current_epoch = self.hparams.epoch_counter.current
            # if current_epoch % self.hparams.valid_search_interval == 0:
            # history_bos = torch.LongTensor([hparams["bos_index"]] + (history_bos))
            padding_mask = ~self.hparams.padding_mask(
                history_bos, pad_idx=tokenizer.unk_token_id
            )
            hyps = self.modules.gpt_model.generate(
                history_bos.detach(),
                history_token_type.detach(),
                padding_mask.detach(),
            )
        elif stage == sb.Stage.TEST:
            padding_mask = ~self.hparams.padding_mask(
                history_bos, pad_idx=tokenizer.unk_token_id
            )
            hyps = self.modules.gpt_model.generate(
                history_bos.detach(),
                history_token_type.detach(),
                padding_mask.detach(),
                "beam",
            )

        if stage != sb.Stage.TRAIN:
            reply_truncated = [
                reply_eos[i][
                    : int(reply_lens[i].item() * reply_eos.shape[1] - 1)
                ].detach()
                for i in range(reply_eos.shape[0])
            ]
            predicted_words = tokenizer.batch_decode(
                hyps[:, history_bos.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            target_words = tokenizer.batch_decode(
                reply_truncated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            self.bleu_4_metric.append(ids, predicted_words, target_words)
            self.bleu_2_metric.append(ids, predicted_words, target_words)
            if stage != sb.Stage.TRAIN:
                self.hyps.extend(predicted_words)
                self.references.extend(target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.bleu_4_metric = self.hparams.bleu_4_computer()
            self.bleu_2_metric = self.hparams.bleu_2_computer()
            self.hyps = []
            self.references = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        stage_stats["PPL"] = math.exp(stage_loss)
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["BLEU_4"] = self.bleu_4_metric.summarize("BLEU")
            stage_stats["BLEU_2"] = self.bleu_2_metric.summarize("BLEU")
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"PPL": stage_stats["PPL"]},
                min_keys=["PPL"],
            )
            if epoch == hparams["number_of_epochs"] - 1:
                with open(
                    self.hparams.bleu_4_valid_file, "w", encoding="utf-8"
                ) as w:
                    self.bleu_4_metric.write_stats(w)
                    for i in range(len(self.hyps)):
                        w.write("target: " + str(self.references[i]) + "\n")
                        w.write("predicted:" + str(self.hyps[i]) + "\n")
                        w.write(
                            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                        )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(
                self.hparams.bleu_4_test_file, "w", encoding="utf-8"
            ) as w:
                self.bleu_4_metric.write_stats(w)
                for i in range(len(self.hyps)):
                    w.write("target: " + str(self.references[i]) + "\n")
                    w.write("predicted:" + str(self.hyps[i]) + "\n")
                    w.write(
                        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                    )

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "optimizer": self.optimizer,
        }


def add_special_tokens_(model, tokenizer, attr_to_special_token) -> None:
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        attr_to_special_token  # type: ignore
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens
        )


def dataio_prep(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_multiwoz` to have been called before
    this, so that the `train.json`, `dev.json`,  and `test.json` manifest
    files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    tokenizer : tokenizer
        Object for converting text to tokens.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # convert special tokens to their ids
    bos, eos, system, user = tokenizer.convert_tokens_to_ids(
        hparams["special_tokens"]
    )
    # history_window, i.e. how many user-system exchanges consider as context (+1 to consider at least the last user turn)
    history_window = 2 * hparams["max_history"] + 1

    #  Define history pipeline:
    @sb.utils.data_pipeline.takes("history")
    @sb.utils.data_pipeline.provides(
        "history",
        "history_tokens_lists",
        "history_ids",
        "history_bos",
        "history_token_type",
    )
    def history_pipeline(history):
        yield history

        # encode each turn of the history
        history_tokens_lists = [tokenizer.encode(turn) for turn in history]
        yield history_tokens_lists

        # add speaker tokens to the history turns (user is even, system is odd)
        # BEFORE:  [Hi how are you?], [I'm fine, thanks]
        # AFTER:   [SPK_1 Hi how are you?], [SPK_2 I'm fine, thanks]
        history_input_lists = [
            [user if i % 2 == 0 else system] + encoded_turn
            for i, encoded_turn in enumerate(history_tokens_lists)
        ]

        history_ids = history_input_lists[-history_window:]
        # concatenate every token into a single list
        # list(chain(*[[1, 2], [3, 4], [5]]))
        # >>> [1, 2, 3, 4, 5]
        history_ids = torch.LongTensor(list(chain(*history_ids)))
        # without bos for lm_labels
        yield history_ids

        # create bos version for the input
        history_bos = torch.cat((torch.tensor([bos]), history_ids))
        yield history_bos

        # create a mapping that associates each token in the input to a speaker
        # INPUT: [SPK_1 Hi    how   are   you? ], [SPK_2 I'm   fine, thanks]
        # TYPE:  [SPK_1 SPK_1 SPK_1 SPK_1 SPK_1], [SPK_2 SPK_2 SPK_2 SPK_2 ]
        history_token_type_lists = [
            [user if i % 2 == 0 else system] * len(encoded_turn)
            for i, encoded_turn in enumerate(history_input_lists)
        ]
        history_token_type = torch.LongTensor(
            list(
                chain(
                    *([[system]] + history_token_type_lists[-history_window:])
                )
            )
        )

        yield history_token_type

    #  Define reply pipeline:
    @sb.utils.data_pipeline.takes("reply")
    @sb.utils.data_pipeline.provides(
        "reply",
        "reply_tokens_list",
        "reply_ids",
        "reply_eos",
        "reply_token_type",
    )
    def reply_pipeline(reply):
        yield reply

        reply_tokens_list = tokenizer.encode(reply)
        yield reply_tokens_list

        # specify that the system will say the reply
        reply_input_list = [system] + reply_tokens_list
        reply_ids = torch.LongTensor(reply_input_list)
        yield reply_ids

        # create eos version of the reply for lm_labels
        reply_eos = torch.cat((reply_ids, torch.tensor([eos])))
        yield reply_eos

        # specify the speaker for each token in the reply
        reply_token_type = torch.LongTensor([system] * len(reply_input_list))
        yield reply_token_type

    # Define input_and_token_type_pipeline
    @sb.utils.data_pipeline.takes(
        "history_ids",
        "history_bos",
        "history_token_type",
        "reply_ids",
        "reply_eos",
        "reply_token_type",
    )
    @sb.utils.data_pipeline.provides("input_ids", "token_type_ids", "lm_labels")
    def input_and_token_type_pipeline(
        history_ids,
        history_bos,
        history_token_type,
        reply_ids,
        reply_eos,
        reply_token_type,
    ):
        # put history and reply together
        # N.B. input_sequence = history_bos + reply_ids, we don't have eos in the input
        input_ids = torch.cat((history_bos, reply_ids), -1)
        yield input_ids

        token_type_ids = torch.cat((history_token_type, reply_token_type), -1)
        yield token_type_ids

        # create the language model label (ground truth) for the current input
        # -100 is a special tokens that is ignored during the loss computation
        # the idea is to mask everything except the reply (without the speaker token)
        # N.B. we don't have bos in the input
        lm_labels = (
            [hparams["ignore_index"]] * history_ids.shape[0]
            + [hparams["ignore_index"]]
            + reply_eos[1:].tolist()
        )
        lm_labels = torch.LongTensor(lm_labels)

        yield lm_labels

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                reply_pipeline,
                history_pipeline,
                input_and_token_type_pipeline,
            ],
            output_keys=[
                "id",
                "input_ids",
                "token_type_ids",
                "history_bos",
                "reply_eos",
                "history_token_type",
                "reply_token_type",
                "lm_labels",
            ],
        )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing MultiWOZ)
    from multiwoz_prepare import prepare_mwoz_21

    run_on_main(
        prepare_mwoz_21,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "replacements_path": hparams["replacements_path"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load tokenizer and add special tokens
    tokenizer = hparams["gpt_model"].tokenizer

    #  Load pretrained GPT
    hparams["gpt_model"] = hparams["gpt_model"].to(device=run_opts["device"])

    # Add special tokens to the tokenizer and resize model embedding
    add_special_tokens_(
        hparams["gpt_model"].model, tokenizer, hparams["attr_to_special_tokens"]
    )

    class CustomPaddedBatch(PaddedBatch):
        """PaddedBatch with custom padding values.

        See the documentation of `speechbrain.dataio.batch.PaddedBatch`.

        """

        def __init__(self, examples, *args, **kwargs):
            _, _, system, _ = tokenizer.convert_tokens_to_ids(
                hparams["special_tokens"]
            )
            for k in [
                "input_ids",
                "history_bos",
                "lm_labels",
                "token_type_ids",
                "history_token_type",
            ]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0
                if k in [
                    "input_ids",
                    "history_bos",
                    "token_type_ids",
                    "history_token_type",
                ]:
                    pad_value = tokenizer.unk_token_id
                elif k == "lm_labels":
                    pad_value = hparams["ignore_index"]
                for example in examples:
                    x = example[k]
                    if k in ["history_bos", "history_token_type"]:
                        x = torch.cat(
                            (example[k], torch.LongTensor([system])), -1
                        )
                        example[k] = torch.nn.functional.pad(
                            x, [max_len - len(x), 0], value=pad_value
                        )
                    else:
                        example[k] = torch.nn.functional.pad(
                            x, [0, max_len - len(x)], value=pad_value
                        )
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_options"]["collate_fn"] = CustomPaddedBatch
    hparams["test_dataloader_options"]["collate_fn"] = CustomPaddedBatch

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams, tokenizer)

    # Initialize the Brain object to prepare for mask training.
    res_gen_brain = ResGenBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(res_gen_brain.device)

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    res_gen_brain.fit(
        epoch_counter=res_gen_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = res_gen_brain.evaluate(
        test_set=datasets["test"],
        min_key="PPL",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
