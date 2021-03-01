#!/usr/bin/env/python3
"""This minimal example trains a HMM-DNN speech recognition model given the
alignments over phoneme units in input. The network is MLP composed of a
single layer. A greedy search is used on top of the output probabilities.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).
"""

import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class ASRBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        x = self.modules.linear1(feats)
        x = self.hparams.activation(x)
        x = self.modules.linear2(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        outputs, lens = predictions
        ids = batch.id
        ali, ali_lens = batch.alignments
        loss = self.hparams.compute_cost(outputs, ali, lens)

        if stage != sb.Stage.TRAIN:
            self.err_metrics.append(ids, outputs, ali, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.err_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)

        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "error: %.2f" % self.err_metrics.summarize("average"))


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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("ali")
    @sb.utils.data_pipeline.provides("alignments")
    def alignment_pipeline(ali):
        alignments = sb.dataio.dataio.load_pickle(ali)
        alignments = sb.dataio.dataio.to_longTensor(alignments)
        return alignments

    sb.dataio.dataset.add_dynamic_item(datasets, alignment_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "alignments"])

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
    asr_brain = ASRBrain(hparams["modules"], hparams["opt_class"], hparams)

    # Training/validation loop
    asr_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    asr_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    assert asr_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
