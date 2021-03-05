# TIMIT ASR with CTC models.
This folder contains the scripts to train a CTC system using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1

# How to run
python train.py train/train.yaml

# Results
| Release | hyperparams file | Val. PER | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml |  12.80 | 14.78 | https://drive.google.com/drive/folders/1OhBOTfC34PaOuiLIUjEBP1JmmlBTxJ8D?usp=sharing | 1xV100 16GB |

# Training Time
About 1 min and 30 sec for each epoch with a TESLA V100.

