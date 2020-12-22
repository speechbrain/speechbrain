#!/usr/bin/env python3
import os
import math
import speechbrain as sb


class LMBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        phns, phn_lens = batch.phn_encoded_bos
        logits = self.modules.model(phns)
        pout = self.hparams.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, batch, stage):
        phns, phn_lens = batch.phn_encoded_eos
        loss = self.hparams.compute_cost(predictions, phns, length=phn_lens)

        return loss

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

    # Update label encoder:
    label_encoder = hparams["label_encoder"]
    label_encoder.update_from_didataset(
        hparams["train_data"], output_key="phn_list", sequence_input=True
    )
    label_encoder.update_from_didataset(
        hparams["valid_data"], output_key="phn_list", sequence_input=True
    )
    label_encoder.insert_bos_eos(bos_index=hparams["eos_bos_index"])

    lm_brain = LMBrain(hparams["modules"], hparams["opt_class"], hparams)
    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        hparams["train_data"],
        hparams["valid_data"],
    )
    lm_brain.evaluate(hparams["valid_data"])

    # Check that model overfits for an integration test
    assert lm_brain.train_loss < 0.15


if __name__ == "__main__":
    main()


def test_error():
    main()
