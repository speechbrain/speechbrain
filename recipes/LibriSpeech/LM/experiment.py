#!/usr/bin/env python3
import os
import sys
import torch
import logging
import glob
import sentencepiece as spm

from datasets import load_dataset

import speechbrain as sb
from speechbrain.utils.data_utils import download_file


logger = logging.getLogger(__name__)


# Define training procedure
class LM(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the sentence batches to the output probabilities."""
        tokens_bos = batch["tokens_bos"].to(self.device)
        logits = self.hparams.model(tokens_bos)
        pred = self.hparams.log_softmax(logits)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        tokens_eos = batch["tokens_eos"].to(self.device)
        tokens_len = batch["tokens_len"].to(self.device)

        # convert to speechbrain-style relative length
        rel_length = tokens_len / tokens_eos.shape[-1]
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=rel_length
        )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        (loss / self.hparams.accu_steps).backward()

        if self.step % self.hparams.accu_steps == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["loss"],
            )


def data_io_prepare(hparams, run_opts):
    """Loads the sentence piece tokenizer specified in the yaml file"""
    save_model_path = os.path.join(
        hparams["save_folder"],
        "{}_unigram.model".format(hparams["output_neurons"]),
    )

    if "tokenizer_file" in hparams:
        download_file(
            source=hparams["tokenizer_file"],
            dest=save_model_path,
            replace_existing=True,
        )

    # Defining tokenizer and loading it
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(save_model_path)

    """grap all the .txt files for transcripts"""
    logging.info("generating datasets...")
    data_folder = hparams["data_folder"]
    train_transcripts = glob.glob(
        os.path.join(data_folder, "train*/**/*.trans.txt"), recursive=True
    )
    dev_transcripts = glob.glob(
        os.path.join(data_folder, "dev*/**/*.trans.txt"), recursive=True
    )
    test_transcripts = glob.glob(
        os.path.join(data_folder, "test*/**/*.trans.txt"), recursive=True
    )

    """prepare data and generate datasets"""
    datasets = load_dataset(
        "dataset.py",
        data_files={
            "train": train_transcripts,
            "dev": dev_transcripts,
            "test": test_transcripts,
        },
    )
    if not os.path.exists(hparams["dataset_cache_path"]):
        logging.info("Cannot find pre-made dataset")
        logging.info(
            "tokenizing and batching dataset to {}".format(
                hparams["dataset_cache_path"]
            )
        )

        def encode(data):  # encode the data using the pretrained dataset
            text = data["text"]
            tokens_list = [tokenizer.sp.encode_as_ids(t) for t in text]
            tokens_bos = [
                torch.tensor([hparams["bos_index"]] + (tl))
                for tl in tokens_list
            ]
            tokens_eos = [
                torch.tensor(tl + [hparams["eos_index"]]) for tl in tokens_list
            ]
            token_len = [torch.tensor([len(t_eos)]) for t_eos in tokens_eos]
            tokens_bos = torch.nn.utils.rnn.pad_sequence(
                tokens_bos, batch_first=True
            )
            tokens_eos = torch.nn.utils.rnn.pad_sequence(
                tokens_eos, batch_first=True
            )
            tokens_len = torch.cat(token_len, dim=0)
            return {
                "tokens_bos": tokens_bos.tolist(),
                "tokens_eos": tokens_eos.tolist(),
                "tokens_len": tokens_len.tolist(),
            }

        datasets = datasets.map(
            encode, batched=True, batch_size=hparams["batch_size"] * 20
        )
        datasets.save_to_disk(hparams["dataset_cache_path"])
        logging.info("Complete!")
    else:
        logging.info(
            "Found exiting pre-made dataset, load it from {}".format(
                hparams["dataset_cache_path"]
            )
        )
        datasets = datasets.load_from_disk(hparams["dataset_cache_path"])

    datasets.set_format(
        type="torch", columns=["tokens_bos", "tokens_eos", "tokens_len"]
    )

    if run_opts["distributed_launch"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets["train"],
            num_replicas=torch.distributed.get_world_size(),
            rank=run_opts["local_rank"],
            shuffle=False,
        )
    else:
        train_sampler = None

    train_data = sb.data_io.dataloader.SaveableDataLoader(
        datasets["train"],
        batch_size=hparams["batch_size"],
        shuffle=False,
        sampler=train_sampler,
    )
    valid_data = sb.data_io.dataloader.SaveableDataLoader(
        datasets["train"], batch_size=hparams["batch_size"], shuffle=False,
    )
    test_data = sb.data_io.dataloader.SaveableDataLoader(
        datasets["train"], batch_size=hparams["batch_size"], shuffle=False,
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the dataloader objects as well as tokenization and encoding
    train_data, valid_data, test_data = data_io_prepare(hparams, run_opts)

    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lm_brain.fit(
        lm_brain.hparams.epoch_counter, train_data, valid_data,
    )

    # evaluation
    test_stats = lm_brain.evaluate(test_data, min_key="loss",)
