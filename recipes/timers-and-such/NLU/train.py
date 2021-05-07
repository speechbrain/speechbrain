#!/usr/bin/env/python3
"""
Recipe for "decoupled" (speech -> ASR -> text -> NLU -> semantics) SLU.

The NLU part is trained on the ground truth transcripts, and at test time
we use the ASR to transcribe the audio and use that transcript as the input to the NLU.

Run using:
> python train.py hparams/train.yaml

Authors
 * Loren Lugosch, Mirco Ravanelli 2020
"""

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)

        tokens_bos, tokens_bos_lens = batch.tokens_bos

        input_ids = batch.input_ids
        input_masks = batch.input_masks

        # Bert expects a Tensor not a Padded Batch, so we have to extract data from the Padded Batch object
        # Bert Forward Pass
        bert_output = self.hparams.bert_model(input_ids.data, input_masks.data)

        # Caculate the real lengths of the inputs
        bert_token_lens = torch.tensor([sum(t) for t in input_masks.data])
        bert_max_length = bert_token_lens.max().item()
        bert_token_lens = bert_token_lens.float()
        bert_token_lens = bert_token_lens / bert_max_length

        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, bert_output, bert_token_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            return p_seq, bert_token_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(
                bert_output, bert_token_lens
            )
            return p_seq, bert_token_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""

        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, bert_token_lens = predictions
        else:
            p_seq, bert_token_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # (No ctc loss)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_semantics = [
                tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            for i in range(len(target_semantics)):
                print(" ".join(predicted_semantics[i]))
                print(" ".join(target_semantics[i]))
                print("")

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.bert_optimizer.step()
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.bert_optimizer.zero_grad()

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
            stage_stats["SER"] = self.wer_metric.summarize("SER")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["SER"])
            old_lr_bert, new_lr_bert = self.hparams.lr_annealing_bert(
                stage_stats["SER"]
            )

            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.bert_optimizer, new_lr_bert
            )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "bert_lr": old_lr_bert,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"SER": stage_stats["SER"]}, min_keys=["SER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the bert optimizer and model optimizer"
        self.bert_optimizer = self.hparams.bert_opt_class(
            self.modules.bert_model.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("bert_opt", self.bert_optimizer)
            self.checkpointer.add_recoverable("optimizer", self.optimizer)


def data_io_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    # validation path of data
    valid_path = hparams["csv_dev_real"]

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_path, replacements={"data_root": data_folder},
    )

    test_real_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test_real"],
        replacements={"data_root": data_folder},
    )

    datasets = [
        train_data,
        valid_data,
        test_real_data,
    ]

    bert_tokenizer = hparams["bert_tokenizer"]

    # 2. Define input pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("input_ids", "input_masks")
    def transcript_pipeline(transcript):
        input_ids, input_masks = bert_tokenizer.encode(transcript)
        input_ids = torch.LongTensor(input_ids)
        yield input_ids
        input_masks = torch.LongTensor(input_masks)
        yield input_masks

    sb.dataio.dataset.add_dynamic_item(datasets, transcript_pipeline)

    tokenizer = hparams["tokenizer"]

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "transcript",
            "semantics",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "input_ids",
            "input_masks",
        ],
    )
    return (
        train_data,
        valid_data,
        test_real_data,
        tokenizer,
    )


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # plots results every N iterations

    hparams["bert_model"].cuda()

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing TAS)
    from prepare_nlu import prepare_TAS  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_TAS,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_real_set, tokenizer,) = data_io_prepare(hparams)

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
    slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test (real data)
    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
    slu_brain.evaluate(
        test_real_set,
        test_loader_kwargs=hparams["dataloader_opts"],
        min_key="SER",
    )
