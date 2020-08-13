#!/usr/bin/env python3
import os
import sys
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from asr_brain import ASR_Brain

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import prepare_timit  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)

# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(iter(train_set))

# Create brain object for training
params.modules["ind2lab"] = params.train_loader.label_dict["phn"]["index2lab"]
asr_brain = ASR_Brain(
    modules=params.modules,
    optimizers={("model", "output"): params.optimizer},
    device=params.device,
    first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.modules["checkpointer"].recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.modules["checkpointer"].recover_if_possible(min_key="PER")
test_stats = asr_brain.evaluate(params.test_loader())
params.modules["train_logger"].log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
