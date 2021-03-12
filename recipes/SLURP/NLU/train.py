#!/usr/bin/env/python3
"""
Text-only NLU recipe. This recipes takes the golden ASR
transcriptions and tries to estimate the semantics on
the top of that.

Authors
 * Loren Lugosch, Mirco Ravanelli 2020

"""

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd


# Define training procedure
class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computations from input to semantic outputs"""
        batch = batch.to(self.device)
        transcript_tokens, transcript_tokens_lens = batch.transcript_tokens
        (
            semantics_tokens_bos,
            semantics_tokens_bos_lens,
        ) = batch.semantics_tokens_bos

        embedded_transcripts = self.hparams.input_emb(transcript_tokens)
        encoder_out = self.hparams.slu_enc(embedded_transcripts)
        e_in = self.hparams.output_emb(semantics_tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, transcript_tokens_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            return p_seq, transcript_tokens_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(
                encoder_out, transcript_tokens_lens
            )
            return p_seq, transcript_tokens_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, transcript_tokens_lens = predictions
        else:
            p_seq, transcript_tokens_lens, predicted_tokens = predictions

        ids = batch.id
        (
            semantics_tokens_eos,
            semantics_tokens_eos_lens,
        ) = batch.semantics_tokens_eos
        semantics_tokens, semantics_tokens_lens = batch.semantics_tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, semantics_tokens_eos, length=semantics_tokens_eos_lens
        )

        # (No ctc loss)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_semantics = [
                slu_tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            self.log_outputs(predicted_semantics, target_semantics)

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

            if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
                with jsonlines.open(
                    hparams["output_folder"] + "/predictions.jsonl", mode="a"
                ) as writer:
                    for i in range(len(predicted_semantics)):
                        try:
                            dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                        except SyntaxError:  # need this if the output is not a valid dictionary
                            dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                        dict["file"] = id_to_file[ids[i]]
                        writer.write(dict)

        return loss

    def log_outputs(self, predicted_semantics, target_semantics):
        """ TODO: log these to a file instead of stdout """
        for i in range(len(target_semantics)):
            print(" ".join(predicted_semantics[i]).replace("|", ","))
            print(" ".join(target_semantics[i]).replace("|", ","))
            print("")

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:

            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    asr_tokenizer = hparams["asr_tokenizer"]
    slu_tokenizer = hparams["slu_tokenizer"]

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "transcript_tokens")
    def transcript_pipeline(transcript):
        yield transcript
        transcript_tokens_list = asr_tokenizer.encode_as_ids(transcript)
        transcript_tokens = torch.LongTensor(transcript_tokens_list)
        yield transcript_tokens

    sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    # 3. Define output pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics",
        "semantics_token_list",
        "semantics_tokens_bos",
        "semantics_tokens_eos",
        "semantics_tokens",
    )
    def semantics_pipeline(semantics):
        yield semantics
        semantics_tokens_list = slu_tokenizer.encode_as_ids(semantics)
        yield semantics_tokens_list
        semantics_tokens_bos = torch.LongTensor(
            [hparams["bos_index"]] + (semantics_tokens_list)
        )
        yield semantics_tokens_bos
        semantics_tokens_eos = torch.LongTensor(
            semantics_tokens_list + [hparams["eos_index"]]
        )
        yield semantics_tokens_eos
        semantics_tokens = torch.LongTensor(semantics_tokens_list)
        yield semantics_tokens

    sb.dataio.dataset.add_dynamic_item(datasets, semantics_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "transcript",
            "transcript_tokens",
            "semantics",
            "semantics_tokens_bos",
            "semantics_tokens_eos",
            "semantics_tokens",
        ],
    )
    return train_data, valid_data, test_data, asr_tokenizer, slu_tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # plots results every N iterations

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing SLURP)
    from prepare import prepare_SLURP  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_SLURP,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "slu_type": "decoupled",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_set,
        valid_set,
        test_set,
        asr_tokenizer,
        slu_tokenizer,
    ) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Brain class initialization
    slu_brain = SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    # slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test
    print("Creating id_to_file mapping...")
    id_to_file = {}
    df = pd.read_csv(hparams["csv_test"])
    for i in range(len(df)):
        id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]

    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
