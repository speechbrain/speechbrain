#!/usr/bin/python
import os
import speechbrain as sb


class AlignBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        wavs, lens = batch.wav
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        predictions, lens = predictions
        phns, phn_lens = batch.phn_enc

        prev_alignments = self.hparams.aligner.get_prev_alignments(
            batch.id, predictions, lens, phns, phn_lens
        )
        loss = self.hparams.compute_cost(predictions, prev_alignments)

        if stage != sb.Stage.TRAIN:
            viterbi_scores, alignments = self.hparams.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )
            self.hparams.aligner.store_alignments(batch.id, alignments)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
            print("Valid loss: %.2f" % stage_loss)
            print("Recalculating and recording alignments...")
            self.evaluate(self.hparams.train_data)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    # Label encoder:
    encoder = hparams["label_encoder"]
    dsets = [hparams["train_data"], hparams["valid_data"], hparams["test_data"]]
    for dset in dsets:
        encoder.update_from_didataset(dset, "phn", sequence_input=True)
    for dset in dsets:
        dset.add_dynamic_item("phn_enc", encoder.encode_sequence_torch, "phn")
        dset.set_output_keys(["id", "wav", "phn_enc"])

    align_brain = AlignBrain(hparams["modules"], hparams["opt_class"], hparams)
    align_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_data"],
        hparams["valid_data"],
        batch_size=hparams["batch_size"],
    )
    align_brain.evaluate(hparams["test_data"], batch_size=hparams["batch_size"])

    # Check that model overfits for integration test
    assert align_brain.train_loss < 2.0


if __name__ == "__main__":
    main()


def test_error():
    main()
