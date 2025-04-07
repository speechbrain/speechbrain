#!/usr/bin/env/python3
"""This minimal example trains an HMM-based aligner with the Viterbi algorithm.
The encoder is based on a combination of convolutional, recurrent, and
feed-forward networks (CRDNN) that predict phoneme states.
Given the tiny dataset, the expected behavior is to overfit the training data
(with a validation performance that stays high).
"""

import pathlib

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb


class AlignBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Given an input batch it computes the output probabilities."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Given the network predictions and targets computed the forward loss."""
        predictions, lens = predictions
        phns, phn_lens = batch.phn_encoded

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
        """Gets called when a stage (either training, validation, test) starts."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
            print("Valid loss: %.2f" % stage_loss)
            print("Recalculating and recording alignments...")
            self.evaluate(self.hparams.train_data)


def data_prep(data_folder, hparams):
    """Creates the datasets and their data processing pipelines."""
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "../annotation/ASR_train.json",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "../annotation/ASR_dev.json",
        replacements={"data_root": data_folder},
    )

    # The evaluate method of the brain class, needs to align over training data
    hparams["train_data"] = train_data

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    label_encoder.expect_len(hparams["num_labels"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    label_encoder.update_from_didataset(valid_data, output_key="phn_list")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

    return train_data, valid_data


def main(device="cpu"):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = "../../samples/ASR/"
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    ali_brain = AlignBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
    )

    # Training/validation loop
    ali_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    ali_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    assert ali_brain.train_loss < 2.0


if __name__ == "__main__":
    main()


def test_error(device):
    main(device)
