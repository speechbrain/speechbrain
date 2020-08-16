#!/usr/bin/python
import speechbrain as sb
from speechbrain.nnet.containers import Sequential


# Trains xvector model
class XvectorBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x

        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)
        x_vect = self.xvector_model(feats, init_params=init_params)
        outputs = self.classifier(x_vect, init_params)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        loss = self.compute_cost(predictions, spkid, lens)

        if stage != "train":
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != "train":
            self.error_metrics = self.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            self.train_loss = stage_loss
        if stage == "valid":
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != "train":
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "error: %.2f" % self.error_metrics.summarize())


# Extracts xvector given data and truncated model
class Extractor(Sequential):
    def __init__(self, model, feats, norm):
        super().__init__()
        self.model = model
        self.feats = feats
        self.norm = norm

    def get_emb(self, feats, lens):

        emb = self.model(feats, lens)

        return emb

    def extract(self, x):
        id, wavs, lens = x

        feats = self.feats(wavs, init_params=False)
        feats = self.norm(feats, lens)

        emb = self.get_emb(feats, lens)
        emb = emb.detach()

        return emb
