#!/usr/bin/env python3
"""Recipe for training a gpt_based response generation model  with MultiWOZ.
The system employs GPT2 (https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf).
This recipe take the GPT2LMHeadModel to fine-tune for response generation task on the NLL.

To run this recipe, do the following:
> python train_with_gpt.py hparams/train_with_gpt.yaml

Authors
 * Pooneh Mousavi 2023
 * Simone Alghisi 2023
"""


import sys
import speechbrain as sb
import torch
from itertools import chain
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from transformers import GPT2Tokenizer


class ResGenBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a gpt decoder.
        """
        # Get required data from batch
        batch = batch.to(self.device)
        input_ids, _ = batch.input_ids
        token_type_ids, _ = batch.token_type_ids

        # Forward Pass
        outputs = self.modules.gpt_model(input_ids, token_type_ids,).logits

        #  apply softmax if necessary
        outputs = self.hparams.log_softmax(outputs)

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the NLL-loss using reply as label.
        """
        # Get required data from batch
        lm_labels, labels_lens = batch.lm_labels

        loss = self.hparams.compute_cost(predictions, lm_labels, labels_lens)

        # TODO:
        # add greedy and beamsearch for inference time
        # add some scores suitable for response generation

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

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
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats={"loss": stage_loss},
            )
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss},
            )

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)


def add_special_tokens_(model, tokenizer, attr_to_special_token,) -> None:
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
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `dev.json`,  and `test.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
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

    #  Define histoy pipeline:
    @sb.utils.data_pipeline.takes("history")
    @sb.utils.data_pipeline.provides(
        "history",
        "history_tokens_lists",
        "history_input_lists",
        "history_token_type_lists",
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
        yield history_input_lists

        # create a mapping that associates each token in the input to a speaker
        # INPUT: [SPK_1 Hi    how   are   you? ], [SPK_2 I'm   fine, thanks]
        # TYPE:  [SPK_1 SPK_1 SPK_1 SPK_1 SPK_1], [SPK_2 SPK_2 SPK_2 SPK_2 ]
        history_token_type_lists = [
            [user if i % 2 == 0 else system] * len(encoded_turn)
            for i, encoded_turn in enumerate(history_input_lists)
        ]
        yield history_token_type_lists

    #  Define reply pipeline:
    @sb.utils.data_pipeline.takes("reply")
    @sb.utils.data_pipeline.provides(
        "reply",
        "reply_tokens_list",
        "reply_input_list",
        "reply_token_type_list",
    )
    def reply_pipeline(reply):
        yield reply

        # same as history
        reply_tokens_list = tokenizer.encode(reply)
        yield reply_tokens_list

        # specify that the system will say the reply
        reply_input_list = [system] + reply_tokens_list
        yield reply_input_list

        # specify the speaker for each token in the reply
        reply_token_type_list = [system] * len(reply_input_list)
        yield reply_token_type_list

    # Define input_and_token_type_pipeline
    @sb.utils.data_pipeline.takes(
        "history_input_lists",
        "history_token_type_lists",
        "reply_input_list",
        "reply_token_type_list",
    )
    @sb.utils.data_pipeline.provides("input_ids", "token_type_ids", "lm_labels")
    def input_and_token_type_pipeline(
        history_input_lists,
        history_token_type_lists,
        reply_input_list,
        reply_token_type_list,
    ):
        # optionally add eos to reply
        reply_input_list = (
            reply_input_list + [eos] if hparams["with_eos"] else []
        )

        # add bos and to the history
        history_input_lists = [[bos]] + history_input_lists[-history_window:]

        # put history and reply together
        input_sequence = history_input_lists + [reply_input_list]

        # concatenate every token into a single list
        # list(chain(*[[1, 2], [3, 4], [5]]))
        # >>> [1, 2, 3, 4, 5]
        input_ids = list(chain(*input_sequence))
        input_ids = torch.LongTensor(input_ids)
        yield input_ids

        # do the same for the token_type
        reply_token_type_list = (
            reply_token_type_list + [system] if hparams["with_eos"] else []
        )

        # bos token belongs to the system
        history_token_type_lists = [[system]] + history_token_type_lists[
            -history_window:
        ]

        token_type_ids = history_token_type_lists + [reply_token_type_list]

        token_type_ids = list(chain(*token_type_ids))
        token_type_ids = torch.LongTensor(token_type_ids)
        yield token_type_ids

        # create the language model label (ground truth) for the current input
        # -100 is a special tokens that is ignored during the loss computation
        # the idea is to mask everything except the reply (withouth the speaker token)
        lm_labels = (
            ([-100] * sum(len(s) for s in input_sequence[:-1]))
            + [-100]
            + input_sequence[-1][1:]
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
            output_keys=["id", "input_ids", "token_type_ids", "lm_labels"],
        )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
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
    tokenizer = GPT2Tokenizer.from_pretrained(hparams["gpt_hub"])

    #  Load pretrained GPT
    hparams["gpt_model"] = hparams["gpt_model"].to(device=run_opts["device"])

    # Add special tokens to the tokenizer and resize model embedding
    add_special_tokens_(
        hparams["gpt_model"].model, tokenizer, hparams["attr_to_special_tokens"]
    )

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

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    res_gen_brain.fit(
        epoch_counter=res_gen_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = res_gen_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
