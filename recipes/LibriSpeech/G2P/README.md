# Grapheme-to-phoneme (G2P).
This folder contains the scripts to train a grapheme-to-phoneme system
that converts characters in input to phonemes in output. It used the 
lexicon of the LibriSpeech dataset

You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml | 7.28% | https://drive.google.com/drive/folders/1nk9ms8cQ5N07wOG4oTi9h5a1dmiPmvnv?usp=sharing | 1xV100 32GB |


# Training Time
About 2 minutes for each epoch with a TESLA V100.
