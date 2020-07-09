#!/usr/bin/python
import os
import sys
import torch  # noqa F401
import speechbrain as sb

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from ami_prepare import prepare_ami  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

# Prepare data for AMI
prepare_ami(
    data_folder=params.data_folder,
    save_folder=params.save_folder,
    splits=["train", "dev", "test"],
    split_ratio=[90, 10],
    seg_dur=300,
    vad=False,
    rand_seed=params.seed,
)
