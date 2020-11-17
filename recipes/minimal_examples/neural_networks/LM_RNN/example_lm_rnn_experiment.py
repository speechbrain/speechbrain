#!/usr/bin/env python3
import os
import math
import torch
import speechbrain as sb


class LMBrain(sb.Brain):
    def compute_forward(self, y, stage):
        ids, phns, phn_lens = y
        y_in = sb.data_io.data_io.prepend_bos_token(
            phns, self.hparams.bos_index
        )
        logits = self.modules.model(y_in)
        pout = self.hparams.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, targets, stage):
        pout = predictions
        ids, phns, phn_lens = targets

        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = sb.data_io.data_io.append_eos_token(
            phns, length=abs_length, eos_index=self.hparams.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns_with_eos.shape[1]
        loss = self.hparams.compute_cost(pout, phns_with_eos, length=rel_length)

        return loss

    def fit_batch(self, batch):
        inputs = batch[0]
        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, inputs, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        inputs = batch[0]
        out = self.compute_forward(inputs, stage=stage)
        loss = self.compute_objectives(out, inputs, stage=stage)
        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            perplexity = math.e ** stage_loss
            print(stage, "perplexity: %.2f" % perplexity)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    lm_brain = LMBrain(hparams["modules"], hparams["opt_class"], hparams)
    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        hparams["train_loader"](),
        hparams["valid_loader"](),
    )
    lm_brain.evaluate(hparams["test_loader"]())

    # Check that model overfits for an integration test
    assert lm_brain.train_loss < 0.15


if __name__ == "__main__":
    main()


def test_error():
    main()
