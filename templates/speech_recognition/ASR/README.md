# Template for Speech Recognition
This folder provides a working, well-documented example for training
a  seq2seq (+ CTC) speech recognizer model from scratch, based on a few hours of data.

There are three files here:

* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.

To train the speech recognition model, just execute the following on the command-line:

```bash
python train.py train.yaml
```

We assume you already trained the tokenizer (see ../Tokenizer) and the language model (../LM).
Training is done with the mini-librispeech dataset using a CRDNN model for encoding and a GRU for decoding.
We pre-train with a larger model to ensure convergence (mini-librispeech is too small for training an e2e model from scratch).
In a real case, you can skip pre-training and train from scratch on a larger dataset.

