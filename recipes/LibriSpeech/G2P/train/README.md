# Grapheme-to-phoneme (G2P).
This folder contains the scripts to train a grapheme-to-phoneme system
that converts characters in input to phonemes in output. It used the 
lexicon of the LibriSpeech dataset

You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test WER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml | 3.08 | Not Available | 1xV100 32GB |


# Training Time
About N for each epoch with a TESLA V100.
