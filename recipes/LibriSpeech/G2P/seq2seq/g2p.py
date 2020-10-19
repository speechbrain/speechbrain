#!/usr/bin/env/python3
"""Recipe for training a graphene-to-phoneme system with librispeech lexicon.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch.

To run this recipe, do the following:
> python g2p.py hyperparams.yaml

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

import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding


class G2P(sb.Brain):
    def compute_forward(self, x, y, stage):
        """Forward computations from the chars to the phoneme probabilities."""
        id, chars, char_lens = x
        id, phns, phn_lens = y

        chars, char_lens = chars.to(self.device), char_lens.to(self.device)
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        emb_char = self.hparams.encoder_emb(chars)
        x, _ = self.modules.enc(emb_char)

        # Prepend bos token at the beginning
        y_in = sb.data_io.prepend_bos_token(phns, self.hparams.bos)
        e_in = self.modules.emb(y_in)

        h, w = self.modules.dec(e_in, x, char_lens)
        logits = self.modules.lin(h)
        outputs = self.hparams.softmax(logits)

        if stage != sb.Stage.TRAIN:
            seq, scores = self.hparams.beam_searcher(x, char_lens)
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, targets, stage):
        """Computes the NLL loss given predictions and targets."""
        if stage == sb.Stage.TRAIN:
            outputs = predictions
        else:
            outputs, seq = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = sb.data_io.append_eos_token(
            phns, length=abs_length, eos_index=self.hparams.eos
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns_with_eos.shape[1]
        loss = self.hparams.compute_cost(
            outputs, phns_with_eos, length=rel_length
        )

        if stage != sb.Stage.TRAIN:
            # Convert indices to words
            phns = undo_padding(phns, phn_lens)
            phns = sb.data_io.convert_index_to_lab(phns, self.hparams.ind2lab)
            seq = sb.data_io.convert_index_to_lab(seq, self.hparams.ind2lab)
            self.per_metrics.append(ids, seq, phns)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        inputs, targets = batch
        preds = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        """Computations needed for validation/test batches"""
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, stage)
        loss = self.compute_objectives(out, targets, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["PER"] = self.per_metrics.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["PER"])
            sb.nnet.update_learning_rate(self.optimizer, new_lr)
            if self.root_process:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": old_lr},
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
                self.checkpointer.save_and_keep_only(
                    meta={"PER": stage_stats["PER"]}, min_keys=["PER"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.per_metrics.write_stats(w)


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from librispeech_prepare import prepare_librispeech  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare LibriSpeech lexicon
    prepare_librispeech(
        data_folder=hparams["data_folder"],
        splits=[],
        save_folder=hparams["data_folder"],
        create_lexicon=True,
    )

    # Load index2label dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    test_set = hparams["test_loader"]()
    hparams["ind2lab"] = hparams["test_loader"].label_dict["phonemes"][
        "index2lab"
    ]

    # Brain class initialization
    g2p_brain = G2P(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    g2p_brain.fit(g2p_brain.hparams.epoch_counter, train_set, valid_set)

    # Test
    g2p_brain.evaluate(test_set, min_key="PER")
