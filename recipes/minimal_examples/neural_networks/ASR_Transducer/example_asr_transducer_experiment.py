#!/usr/bin/python
import os
import speechbrain as sb
import pytest


class TransducerBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Transcription network: input-output dependency
        TN_output = self.modules.enc(feats)
        TN_output = self.modules.enc_lin(TN_output)

        # Prediction network: output-output dependency
        targets, target_lens = batch.phn_encoded_bos
        PN_output = self.modules.emb(targets)
        PN_output, _ = self.modules.dec(PN_output)
        PN_output = self.modules.dec_lin(PN_output)

        # Joint the networks
        joint = self.modules.Tjoint(
            TN_output.unsqueeze(2), PN_output.unsqueeze(1),
        )
        outputs = self.modules.output(joint)
        outputs = self.hparams.log_softmax(outputs)
        if stage == sb.Stage.TRAIN:
            return outputs, lens
        else:
            hyps, scores, _, _ = self.hparams.searcher(TN_output)
            return outputs, lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN:
            predictions, lens = predictions
        else:
            predictions, lens, seq = predictions
            self.per_metrics.append(batch.id, seq, phns, target_len=phn_lens)

        loss = self.hparams.compute_cost(
            predictions,
            phns.to(self.device).long(),
            lens,
            phn_lens.to(self.device),
        )
        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))


def main():
    pytest.importorskip("numba")
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
    label_encoder.insert_bos_eos(bos_index=hparams["blank_id"])

    transducer_brain = TransducerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
    )
    transducer_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_data"],
        hparams["valid_data"],
    )
    transducer_brain.evaluate(hparams["valid_data"])

    # Integration test: check that the model overfits the training data
    assert transducer_brain.train_loss <= 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
