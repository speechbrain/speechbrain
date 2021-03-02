# LibriSpeech ASR with Trasformers.
This folder contains the scripts to train a Transformer-based speech recognizer
using LibriSpeech.

You can download LibriSpeech at http://www.openslr.org/12


# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test WER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:|
| 20-05-22 | transformer.yaml | -.-- | Not Available | 1xV100 32GB |
| 20-05-22 | conformer.yaml | -.-- | Not Available | 1xV100 32GB |


# Training Time
About N for each epoch with 4 TESLA V100.
