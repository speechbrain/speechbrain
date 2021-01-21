#!/usr/bin/env/python3
"""This minimal example trains a speaker identification system based on
x-vectors. The encoder is based on TDNNs. The classifier is a MLP.
"""

import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


# Trains xvector model
class XvectorBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the speaker probabilities."
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x_vect = self.modules.xvector_model(feats)
        outputs = self.modules.classifier(x_vect)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CE loss."
        predictions, lens = predictions
        spkid, spkid_lens = batch.spk_id_encoded
        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(
                stage, "error: %.2f" % self.error_metrics.summarize("average")
            )


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "train.json",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "dev.json",
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.update_from_didataset(train_data, output_key="spk_id")
    label_encoder.update_from_didataset(valid_data, output_key="spk_id")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data


def main():
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    xvect_brain = XvectorBrain(
        hparams["modules"], hparams["opt_class"], hparams
    )

    # Training/validation loop
    xvect_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    xvect_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    assert xvect_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
