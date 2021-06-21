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
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained.training import PretrainedModelMixin
from speechbrain.lobes.models.g2p.attnrnn.dataio import (
    grapheme_pipeline,
    phoneme_pipeline,
)


# Define training procedure
class G2PBrain(sb.Brain, PretrainedModelMixin):
    def __init__(self, train_step_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_step_name = train_step_name
        self.train_step = next(
            step for step in self.hparams.train_steps
            if step['name'] == train_step_name)
        self.epoch_counter = self.train_step["epoch_counter"]

    def compute_forward(self, batch, stage):
        """Forward computations from the char batches to the output probabilities."""
        batch = batch.to(self.device)

        p_seq, char_lens, encoder_out = self.hparams.model(
            grapheme_encoded=batch.grapheme_encoded,
            phn_encoded=batch.phn_encoded_bos,
        )

        if stage != sb.Stage.TRAIN:
            hyps, scores = self.hparams.beam_searcher(encoder_out, char_lens)
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
                stats_meta={"Epoch loaded": self.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            wer_file = self.train_step['wer_file']
            with open(wer_file, "w") as w:
                w.write("\nseq2seq loss stats:\n")
                self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "seq2seq, and PER stats written to file",
                    wer_file
                )


def sort_data(data, hparams):
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        data = data.filtered_sorted(sort_key="duration")

    elif hparams["sorting"] == "descending":
        data = data.filtered_sorted(
            sort_key="duration", reverse=True
        )

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    return data


def dataio_prep(hparams, train_step=None):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    if not train_step:
        train_step = hparams
    data_folder = hparams["data_folder"]
    data_load = hparams["data_load"]
    # 1. Declarations:
    train_data = data_load(
        train_step["train_data"], replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    train_data = sort_data(train_data, hparams)

    valid_data = data_load(
        train_step["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = sort_data(valid_data, hparams)

    test_data = data_load(
        train_step["test_data"], replacements={"data_root": data_folder},
    )
    test_data = sort_data(test_data, hparams)


    datasets = [train_data, valid_data, test_data]

    phoneme_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define grapheme pipeline:
    sb.dataio.dataset.add_dynamic_item(
        datasets,
        grapheme_pipeline(
            graphemes=hparams["graphemes"],
            space_separated=hparams['graphemes_space_separated']),
    )

    # 3. Define phoneme pipeline:
    sb.dataio.dataset.add_dynamic_item(
        datasets,
        phoneme_pipeline(
            phonemes=hparams["phonemes"],
            phoneme_encoder=phoneme_encoder,
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
            space_separated=hparams["phonemes_space_separated"]
        ),
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
    if hparams['build_lexicon']:
        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["save_folder"],
                "create_lexicon": True,
                "skip_prep": hparams["skip_prep"],
                "select_n_sentences": hparams.get("select_n_sentences"),
            },
        )
    for train_step in hparams['train_steps']:
        print(f"Running training step: {train_step['name']}")
        # Dataset IO prep: creating Dataset objects and proper encodings for phones
        train_data, valid_data, test_data, phoneme_encoder = dataio_prep(hparams, train_step)

        # Trainer initialization
        g2p_brain = G2PBrain(
            train_step_name=train_step["name"],
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        g2p_brain.phoneme_encoder = phoneme_encoder

        dataloader_opts = hparams["dataloader_opts"]
        # NOTE: This gets modified after the first run and causes a double
        # agument issue
        if 'ckpt_prefix' in dataloader_opts and dataloader_opts['ckpt_prefix'] is None:
            del dataloader_opts['ckpt_prefix']

        # Training/validation loop
        g2p_brain.fit(
            train_step['epoch_counter'],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

        # Test
        g2p_brain.evaluate(
            test_data, min_key="PER", test_loader_kwargs=hparams["dataloader_opts"],
        )

        if hparams.get("save_for_pretrained"):
            g2p_brain.save_for_pretrained()
