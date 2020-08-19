#!/usr/bin/env python3
import os
import math
import torch
import speechbrain as sb

from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token


class LMBrain(sb.Brain):
    def compute_forward(self, y, stage=sb.Stage.TRAIN, init_params=False):
        ids, phns, phn_lens = y
        y_in = prepend_bos_token(phns, bos_index=self.bos_index)
        logits = self.model(y_in, init_params=init_params)
        pout = self.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, targets, stage=sb.Stage.TRAIN):
        pout = predictions
        ids, phns, phn_lens = targets

        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = append_eos_token(
            phns, length=abs_length, eos_index=self.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns_with_eos.shape[1]
        loss = self.compute_cost(pout, phns_with_eos, length=rel_length)

        return loss

    def fit_batch(self, batch):
        for optimizer in self.optimizers.values():
            inputs = batch[0]
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    train_set = hyperparams.train_loader()
    valid_set = hyperparams.valid_loader()
    first_y = next(iter(train_set))

    lm_brain = LMBrain(
        modules=hyperparams.modules,
        optimizers={"model": hyperparams.optimizer},
        device="cpu",
        first_inputs=first_y,
    )

    lm_brain.fit(hyperparams.epoch_counter, train_set, valid_set)

    lm_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for an integration test
    assert lm_brain.train_loss < 0.15


if __name__ == "__main__":
    main()


def test_error():
    main()
