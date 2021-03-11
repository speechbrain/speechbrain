# SLU recipes for Fluent Speech Commands
This folder contains recipes for spoken language understanding (SLU) with [Fluent Speech Commands](fluent.ai/research/fluent-speech-commands/).

### Tokenizer recipe
(You don't need to run this because the other recipes download a tokenizer, but you can run this if you want to train a new tokenizer for Fluent Speech Commands.)

Run this to train the tokenizer:

```
cd Tokenizer
python train.py hparams/tokenizer_bpe51.yaml
```

### Direct recipe
The "direct" recipe maps the input speech to directly to semantics using a seq2seq model.
The encoder is pre-trained using the LibriSpeech seq2seq recipe.

```
cd direct
python train.py hparams/train.yaml
```

# Results

| Release | hyperparams file | Test Acc | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:|
| 21-02-07 | train.yaml | 99.42% | https://drive.google.com/drive/folders/13t2PYdedrPQoNYo_QSf6s04WXu2_vAb-?usp=sharing | 1xV100 32GB |


# Training Time
About 10 minutes for each epoch with a TESLA V100.

