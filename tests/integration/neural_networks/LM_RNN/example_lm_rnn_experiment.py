#!/usr/bin/env/python3

"""This minimal example trains a character-level language model that predicts
the next characters given the previous ones. The system uses a standard
attention-based encoder-decoder pipeline. The encoder is based on a simple LSTM.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).
"""
import math
import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class LMBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input chars it computes the next-char probability."
        chars, char_lens = batch.char_encoded_bos
        logits = self.modules.model(chars)
        pout = self.hparams.log_softmax(logits)
        return pout

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        chars, char_lens = batch.char_encoded_eos
        loss = self.hparams.compute_cost(predictions, chars, length=char_lens)
        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            perplexity = math.e ** stage_loss
            print(stage, "perplexity: %.2f" % perplexity)


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
    char_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define char pipeline:
    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides(
        "char_list", "char_encoded_bos", "char_encoded_eos"
    )
    def char_pipeline(char):
        char_list = char.strip().split()
        yield char_list
        char_encoded = char_encoder.encode_sequence_torch(char_list)
        char_encoded_bos = char_encoder.prepend_bos_index(char_encoded).long()
        yield char_encoded_bos
        char_encoded_eos = char_encoder.append_eos_index(char_encoded).long()
        yield char_encoded_eos

    sb.dataio.dataset.add_dynamic_item(datasets, char_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    char_encoder.insert_bos_eos(bos_index=hparams["bos_index"])
    char_encoder.update_from_didataset(train_data, output_key="char_list")
    char_encoder.update_from_didataset(valid_data, output_key="char_list")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "char_encoded_bos", "char_encoded_eos"]
    )
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
    lm_brain = LMBrain(hparams["modules"], hparams["opt_class"], hparams)

    # Training/validation loop
    lm_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    lm_brain.evaluate(valid_data)

    # Check that model overfits for integration test
    assert lm_brain.train_loss < 0.15


if __name__ == "__main__":
    main()


def test_error():
    main()
