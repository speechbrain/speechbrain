#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.nnet.containers import Sequential
from speechbrain.utils.train_logger import summarize_average

# Load params file
experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../../samples/voxceleb_samples/wav/"
data_folder = os.path.abspath(experiment_dir + data_folder)

with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})


# Trains xvector model
class XvectorBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x

        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        x = params.model(feats, init_params)
        x = params.out_linear(x, init_params)

        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        loss = params.compute_cost(predictions, spkid, lens)

        stats = {}

        if stage != "train":
            stats["error"] = params.compute_error(predictions, spkid, lens)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))


# Extracts xvector given data and truncated model
class Extractor(Sequential):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_emb(self, feats):

        emb = self.model(feats)

        return emb

    def extract(self, x):
        id, wavs, lens = x

        feats = params.compute_features(wavs, init_params=False)
        feats = params.mean_var_norm(feats, lens)

        emb = self.get_emb(feats)
        emb = emb.detach()

        return emb


# Data loaders
train_set = params.train_loader()
valid_set = params.valid_loader()

# Xvector Model
modules = [params.model, params.out_linear]
first_x, first_y = next(iter(train_set))

# Object initialization for training xvector model
xvect_brain = XvectorBrain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Train the Xvector model
xvect_brain.fit(
    range(params.number_of_epochs), train_set=train_set, valid_set=valid_set,
)
print("Xvector model training completed!")

# Truncate model and keep till layer emb a
model_a = Sequential(*xvect_brain.modules[0].layers[0:17],)
print("Model has been truncated!")

# Instantiate extractor obj
ext_brain = Extractor(model=model_a)

# Extract xvectors from a validation sample
valid_x, valid_y = next(iter(valid_set))
print(
    "Extracting Xvector from a sample validation batch using truncated model!"
)
xvectors = ext_brain.extract(valid_x)
print("Extracted Xvector.Shape: ", xvectors.shape)


# Integration test: Ensure we overfit the training data
def test_error():
    assert xvect_brain.avg_train_loss < 0.1
