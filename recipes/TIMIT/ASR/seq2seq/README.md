# TIMIT ASR with seq2seq models.
This folder contains the scripts to train a seq2seq RNNN-based system using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Val. PER | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml |  12.50 | 14.07 | https://drive.google.com/drive/folders/1OOieZsNJiLSUSjxidmXg0ywYDJCw0dfm?usp=sharing | 1xV100 16GB |
| 21-04-08 | train_with_wav2vec2.yaml |  7.11 | 8.04 | https://drive.google.com/drive/folders/1-IbO7hldwrRh4rwz9xAYzKeeMe57YIiq?usp=sharing | 1xV100 32GB |
