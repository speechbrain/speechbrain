#!/usr/bin/python
import os
import speechbrain as sb
from xvector_brain import XvectorBrain, Extractor

# Load hyperparams file
experiment_dir = os.path.dirname(os.path.abspath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../../samples/voxceleb_samples/wav/"
data_folder = os.path.abspath(experiment_dir + data_folder)

with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

# Data loaders
train_set = hyperparams.train_loader()
valid_set = hyperparams.valid_loader()

# Xvector Model
first_x, first_y = next(iter(train_set))

# Object initialization for training xvector model
xvect_brain = XvectorBrain(
    modules=hyperparams.modules,
    optimizers={("xvector_model", "classifier"): hyperparams.optimizer},
    device="cpu",
    first_inputs=[first_x],
)

# Train the Xvector model
xvect_brain.fit(
    range(hyperparams.number_of_epochs),
    train_set=train_set,
    valid_set=valid_set,
)
print("Xvector model training completed!")

# Instantiate extractor obj
ext_brain = Extractor(
    model=hyperparams.modules["xvector_model"],
    feats=hyperparams.modules["compute_features"],
    norm=hyperparams.modules["mean_var_norm"],
)

# Extract xvectors from a validation sample
valid_x, valid_y = next(iter(valid_set))
print("Extracting Xvector from a sample validation batch!")
xvectors = ext_brain.extract(valid_x)
print("Extracted Xvector.Shape: ", xvectors.shape)


# Integration test: Ensure we overfit the training data
def test_error():
    assert xvect_brain.train_loss < 0.1
