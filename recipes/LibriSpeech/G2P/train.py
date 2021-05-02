#!/usr/bin/env/python3
"""Recipe for training a grapheme-to-phoneme system with librispeech lexicon.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs an LSTM encoder.
The decoder is based on a standard  GRU. The neural network is trained with
negative-log.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders,  and many other possible variations.


Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the char batches to the output probabilities."""
        batch = batch.to(self.device)
        chars, char_lens = batch.grapheme_encoded
        phn_bos, phn_lens = batch.phn_encoded_bos

        emb_char = self.hparams.encoder_emb(chars)
        x, _ = self.modules.enc(emb_char)

        # Prepend bos token at the beginning
        e_in = self.modules.emb(phn_bos)
        h, w = self.modules.dec(e_in, x, char_lens)
        logits = self.modules.lin(h)
        p_seq = self.hparams.log_softmax(logits)

        if stage != sb.Stage.TRAIN:
            hyps, scores = self.hparams.beam_searcher(x, char_lens)
            return p_seq, char_lens, hyps

        return p_seq, char_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        if stage == sb.Stage.TRAIN:
            p_seq, char_lens = predictions
        else:
            p_seq, char_lens, hyps = predictions

        ids = batch.id
        phns_eos, phn_lens_eos = batch.phn_encoded_eos
        phns, phn_lens = batch.phn_encoded

        loss = self.hparams.seq_cost(p_seq, phns_eos, phn_lens_eos)

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.seq_metrics.append(ids, p_seq, phns_eos, phn_lens)
            self.per_metrics.append(
                ids,
                hyps,
                phns,
                None,
                phn_lens,
                self.phoneme_encoder.decode_ndim,
            )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "PER": per,
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "seq2seq, and PER stats written to file",
                    self.hparams.wer_file,
                )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={"data_root": data_folder},
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
        csv_path=hparams["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    phoneme_encoder = sb.dataio.encoder.TextEncoder()
    grapheme_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define grapheme pipeline:
    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides(
        "grapheme_list", "grapheme_encoded_list", "grapheme_encoded"
    )
    def grapheme_pipeline(char):
        grapheme_list = char.strip().split(" ")
        yield grapheme_list
        grapheme_encoded_list = grapheme_encoder.encode_sequence(grapheme_list)
        yield grapheme_encoded_list
        grapheme_encoded = torch.LongTensor(grapheme_encoded_list)
        yield grapheme_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, grapheme_pipeline)

    # 3. Define phoneme pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def phoneme_pipeline(phn):
        phn_list = phn.strip().split(" ")
        yield phn_list
        phn_encoded_list = phoneme_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            phoneme_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            phoneme_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, phoneme_pipeline)

    # 3. Fit encoder:
    grapheme_encoder.update_from_didataset(
        train_data, output_key="grapheme_list"
    )
    phoneme_encoder.update_from_didataset(train_data, output_key="phn_list")

    if hparams["bos_index"] == hparams["eos_index"]:
        phoneme_encoder.insert_bos_eos(
            bos_label="<eos-bos>",
            eos_label="<eos-bos>",
            bos_index=hparams["bos_index"],
        )
    else:
        phoneme_encoder.insert_bos_eos(
            bos_label="<bos>",
            eos_label="<eos>",
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
        )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "grapheme_encoded",
            "phn_encoded",
            "phn_encoded_eos",
            "phn_encoded_bos",
        ],
    )

    return train_data, valid_data, test_data, phoneme_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "create_lexicon": True,
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, phoneme_encoder = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.phoneme_encoder = phoneme_encoder

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data, min_key="PER", test_loader_kwargs=hparams["dataloader_opts"],
    )
