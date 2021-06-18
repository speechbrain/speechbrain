# TIMIT ASR with transformer model.
This folder contains the scripts to train a Transformer-based system(s) using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1

# How to run
python train.py hparams/transformer.yaml

# Results
The following are the results of each model after 140 epochs

| hyperparams file | Val. PER | Test PER | Validation Loss | GPUs | Training Time per epoch | Results at Epoch |
|:---------------------------:| -----:| -----:| --------:|:-----------:|:-----------:|:-----------:|
| transformer_adam.yaml |  19.59 | 20.92 | 26.38 | 1xP100 16GB | 47 sec| 107 |
| performer.yaml | 34.42 | 36.35 | 57.7 | 1xP100 16GB | 2 min 25 sec | 110 |
