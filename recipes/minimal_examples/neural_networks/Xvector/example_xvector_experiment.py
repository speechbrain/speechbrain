#!/usr/bin/python
import os
import torch
import speechbrain as sb


# Trains xvector model
class XvectorBrain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x_vect = self.modules.xvector_model(feats)
        outputs = self.modules.classifier(x_vect)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
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


# Extracts xvector given data and truncated model
class Extractor(torch.nn.Module):
    def __init__(self, model, feats, norm):
        super().__init__()
        self.model = model
        self.feats = feats
        self.norm = norm

    def get_emb(self, feats):

        emb = self.model(feats)

        return emb

    def extract(self, x):
        id, wavs, lens = x

        feats = self.feats(wavs)
        feats = self.norm(feats, lens)

        emb = self.get_emb(feats)
        emb = emb.detach()

        return emb


def main():
    # Load hparams file
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../../samples/voxceleb_samples/wav/"
    data_folder = os.path.abspath(experiment_dir + data_folder)

    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    # Data loaders

    # Object initialization for training xvector model
    xvect_brain = XvectorBrain(
        hparams["modules"], hparams["opt_class"], hparams
    )

    # Train the Xvector model
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    xvect_brain.fit(range(hparams["number_of_epochs"]), train_set, valid_set)
    print("Xvector model training completed!")

    # Instantiate extractor obj
    ext_brain = Extractor(
        model=hparams["xvector_model"],
        feats=hparams["compute_features"],
        norm=hparams["mean_var_norm"],
    )

    # Extract xvectors from a validation sample
    valid_x, valid_y = next(iter(valid_set.get_dataloader()))
    print("Extracting Xvector from a sample validation batch!")
    xvectors = ext_brain.extract(valid_x)
    print("Extracted Xvector.Shape: ", xvectors.shape)

    # Check that the model overfits for an integration test
    assert xvect_brain.train_loss < 0.1


if __name__ == "__main__":
    main()


def test_error():
    main()
