# Language Model with LibriSpeech
This folder contains recipes for training language models for the LibriSpeech Dataset.
It supports both an RNN-based LM and a Transformer-based LM. 
The scripts rely on the HuggingFace dataset, which manages data reading and loading from
large text corpora. 

You can download LibriSpeech at http://www.openslr.org/12

# Extra Dependency:
Make sure you have the HuggingFace dataset installed. If not, type:
pip install datasets

# How to run:
python train.py hparams/RNNLM.yaml
python train.py hparams/transformer.yaml

| Release | hyperparams file | Test PP | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | RNNLM.yaml | --.-- | Not Available | 1xV100 32GB |
| 20-05-22 | transformer.yaml | --.-- | Not Available | 1xV100 32GB |


# Training Time
About N for each epoch with 4 TESLA V100.

