#!/usr/bin/env python3
"""Recipe for training a Language Model with librispeech train-960 transcript and lm_corpus.

To run this recipe, do the following:
> pip install datasets
> python train.py hparams/<hparam_file>.yaml --data_folder <local_path_to_librispeech_dataset>

Authors
 * Jianyuan Zhong 2021
 * Ju-Chieh Chou 2020
"""
import os
import sys
import logging
import glob
import torch
from datasets import load_dataset
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


logger = logging.getLogger(__name__)


# Define training procedure
class LM(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the sentence batches to the output probabilities."""
        batch = batch.to(self.device)
        tokens_bos, _ = batch.tokens_bos
        logits = self.hparams.model(tokens_bos)
        pred = self.hparams.log_softmax(logits)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
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

            if isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams.lr_annealing,
                sb.nnet.schedulers.CyclicCosineScheduler,
            ):
                self.hparams.lr_annealing(self.optimizer)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            if not (
                isinstance(
                    self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
                )
                or isinstance(
                    self.hparams.lr_annealing,
                    sb.nnet.schedulers.CyclicCosineScheduler,
                )
            ):
                old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            else:
                old_lr = self.hparams.lr_annealing.current_lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["loss"],
            )


def dataio_prepare(hparams):
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
        lm_corpus_path=hparams["lm_corpus_path"],
        data_files={
            "train": train_transcripts,
            "dev": dev_transcripts,
            "test": test_transcripts,
        },
    )

    train_data, valid_data, test_data = (
        datasets["train"],
        datasets["dev"],
        datasets["test"],
    )

    """convert huggingface's dataset to DynamicItemDataset via a magical function"""
    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        train_data
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        valid_data
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        test_data
    )

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    """Define text pipeline"""
    # TODO: implement text augmentations pipelines
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("text", "tokens_bos", "tokens_eos")
    def text_pipeline(text):
        yield text
        tokens_list = tokenizer.encode_as_ids(text)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "text", "tokens_bos", "tokens_eos"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the dataloader objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # We download the tokenizer from HuggingFace (or elsewhere depending on
    # the path given in the YAML file).
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # evaluation
    test_stats = lm_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
