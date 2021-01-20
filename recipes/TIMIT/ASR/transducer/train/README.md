# TIMIT ASR with Transducer models.
This folder contains the scripts to train a RNNT system using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1


# Extra-Dependencies
Before running this recipe, make sure numba is installed. Otherwise, run: 
pip install numba

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Val. PER | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 2020-05-22 | train.yaml |  12.8 | 14.2 | Not Available | 1xV100 32GB |

# Training Time
About 3 min and 20 sec for each epoch with a  TESLA V100.

