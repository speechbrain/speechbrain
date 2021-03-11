# Phoneme alignment using TIMIT.
This folder contains the scripts to train an HMM-DNN based alignment system.
It supports Viterbi, Forward, and CTC training.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test Accuracy | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train_BPE_1000.yaml | 79.55 | https://drive.google.com/drive/folders/1fXu7JAVUYxZLosH05iBTEPrJyVSCjNRi?usp=sharing  | 1xV100 32GB |


# Training Time
About 2 minutes for each epoch with a  TESLA V100.
