This folder contains a template for implementing speech recognition systems based on seq2seq (+ CTC) models. 
We assume you already trained the tokenizer (see ../Tokenizer) and the training model (../LM).
Training is done with the mini-librispeech dataset using simple RNNs for encoding and decoding. 
In a real case, we can achieve much better performance using a bigger model (e.g, speechbrian.lobes.models.CRDNN) on larger datasets.

# How to run
python train.py train.yaml

