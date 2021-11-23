# Language Model
This folder contains a recipe for training language models.
It supports both an RNN-based LM and a Transformer-based LM.
The scripts rely on the HuggingFace dataset, which manages data reading and loading from large text corpora.
Training an LM might on large text corpora might take weeks (or months) even on modern GPUs. In this template, for simplicity, we only use the training transcriptions of the mini-librispeech dataset.  In the recipes, we assume you
already ran the tokenizer training (see ../Tokenizer).

# Extra Dependency:
Make sure you have the HuggingFace dataset installed. If not, type:
pip install datasets

# How to run:
python train.py RNNLM.yaml
