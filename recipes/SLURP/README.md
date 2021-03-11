# SLU recipes for SLURP
This folder contains recipes for spoken language understanding (SLU) with [SLURP](https://www.aclweb.org/anthology/2020.emnlp-main.588.pdf).

### Tokenizer recipe
(You don't need to run this because the direct recipe downloads a tokenizer, but you can if you'd like to train a new tokenizer for SLURP.)

Run this to train the tokenizer:

```
cd Tokenizer
python train.py hparams/tokenizer_bpe51.yaml
```

### NLU recipe
The "NLU" recipe takes the true transcript as input rather than speech and trains a seq2seq model to map the transcript to the semantics.

```
cd NLU
python train.py hparams/train.yaml
```


### Direct recipe
The "direct" recipe maps the input speech to directly to semantics using a seq2seq model.
The encoder is pre-trained using the LibriSpeech seq2seq recipe.

```
cd direct
python train.py hparams/train.yaml
```

# Results

Note: SLURP comes with a tool for measuring SLU-F1 and other metrics.
The recipes here dump the model outputs to a file called `predictions.jsonl` in the `results` folder.
You can compute the metrics by feeding `predictions.jsonl` into the [SLURP evaluation tool](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).

The following results were obtained on a 48 GB RTX 8000 (the recipe has also been successfully tested on a 12 GB Tesla K80):

| Model	| scenario (accuracy) | action (accuracy) | intent (accuracy) | Word-F1 | Char-F1 | SLU-F1 | Training time |
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Direct | 81.73 | 77.11 | 75.05 | 61.24 | 65.42 | 63.26 | 1 hour per epoch |

| Model	| scenario (accuracy) | action (accuracy) | intent (accuracy) | Training time |
|:---:|:-----:|:-----:|:-----:|:-----:|
| NLU | 90.81 | 88.29 | 87.28 | 40 min per epoch |


