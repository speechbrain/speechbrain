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
| transformer_madgrad.yaml |  15.32 | 17.10 | 20.77 | 1xP100 16GB | 47 sec| 140 |
| powernorm_madgrad.yaml |  15.36 | 17.52 | 21.13 | 1xP100 16GB | 1 min 3 sec| 130 |
| performer.yaml | 34.42 | 36.35 | 57.7 | 1xP100 16GB | 2 min 25 sec | 110 |
| longformer_madgrad.yaml |  16.50 | 18.48 | 21.62 | 1xP100 16GB | 59 sec| 120 |
