#!/usr/bin/env/python3
"""

Recipe for Timers and Such LM.

Run using:
> python train.py hparams/train.yaml

Authors
 * Loren Lugosch 2020
"""

import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


# Define training procedure
class LM(sb.Brain):
    def compute_forward(self, x, stage):
        ids, transcripts, transcript_lens = x

        # Forward pass
        # Tokenize transcript using ASR model's tokenizer
        (tokens, tokens_lens,) = self.hparams.asr_model.mod.tokenizer(
            transcripts, transcript_lens, hparams["ind2lab"], task="encode",
        )
        tokens, tokens_lens = (
            tokens.to(self.device),
            tokens_lens.to(self.device),
        )
        tokens_with_bos = sb.dataio.dataio.prepend_bos_token(
            tokens, bos_index=self.hparams.asr_model.hparams["bos_index"]
        )
        logits = self.hparams.net(tokens_with_bos)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        return p_seq, tokens, tokens_lens

    def compute_objectives(self, predictions, stage):
        """Computes the loss (NLL)."""
        p_seq, tokens, tokens_lens = predictions

        # Add char_lens by one for eos token
        abs_length = torch.round(tokens_lens * tokens.shape[1])

        # Append eos token at the end of the label sequences
        target_tokens_with_eos = sb.dataio.dataio.append_eos_token(
            tokens,
            length=abs_length,
            eos_index=self.hparams.asr_model.hparams[
                "eos_index"
            ],  # self.hparams.eos_index
        )

        # Convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / target_tokens_with_eos.shape[1]
        loss = self.hparams.seq_cost(
            p_seq, target_tokens_with_eos, length=rel_length
        )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        (inputs,) = batch
        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        (inputs,) = batch
        predictions = self.compute_forward(inputs, stage=stage)
        loss = self.compute_objectives(predictions, stage=stage)
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

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from prepare import prepare_TAS

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_TAS(
        data_folder=hparams["data_folder"],
        type="decoupled",
        train_splits=hparams["train_splits"],
    )

    # Load index2label dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    test_real_set = hparams["test_real_loader"]()
    test_synth_set = hparams["test_synth_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["transcript"][
        "index2lab"
    ]  # ugh

    # Brain class initialization
    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    lm_brain.fit(lm_brain.hparams.epoch_counter, train_set, valid_set)

    # Test
    lm_brain.evaluate(test_real_set)
    lm_brain.evaluate(test_synth_set)
