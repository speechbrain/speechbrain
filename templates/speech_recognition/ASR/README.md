This folder contains a template for implementing speech recognition systems based on seq2seq (+ CTC) models. 
We assume you already trained the tokenizer (see ../Tokenizer) and the training model (../LM).
Training is done with the mini-librispeech dataset using a CRDNN model for encoding and a GRU for decoding. 
We pre-train with a larger model to ensure convergence (mini-librispeech is too small for training an e2e model
from scratch). In a real case, you can skip pre-training and train from scratch on a larger datasets.

# How to run
python train.py train.yaml

