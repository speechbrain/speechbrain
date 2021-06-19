# TIMIT ASR with transformer model.
This folder contains the scripts to train a Transformer-based system(s) using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1

# How to run
python train.py hparams/transformer.yaml

# Results
The following are the results of each model after 140 epochs

| hyperparams file | Val. PER | Test PER | Validation Loss | GPUs | Training Time per epoch | Results at Epoch |
|:---------------------------:| -----:| -----:| --------:|:-----------:|:-----------:|:-----------:|
| transformer.yaml |  19.59 | 20.92 | 26.38 | 1xP100 16GB | 47 sec| 107 |
| powernorm_transformer.yaml | 19.80 | 21.05 | 29.9 | 1xP100 16GB | 49 sec | 110 |