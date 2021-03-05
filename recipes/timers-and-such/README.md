# SLU recipes for Timers and Such v0.1
This folder contains recipes for spoken language understanding (SLU) with [Timers and Such v0.1](https://zenodo.org/record/4110812), an SLU dataset with a large (train/dev/test) set of synthetic speech and a small (train/dev/test) set of real speech.

### LM recipe
This recipe trains a language model (LM) on Timers and Such transcripts. (It is not necessary to run this before running the other recipes, as they download a trained checkpoint.)

### Decoupled recipe
The "decoupled" recipe uses an ASR model (using the LibriSpeech seq2seq recipe) to map speech to text and a separate NLU model, trained on the true transcripts rather than the ASR output, to map text to semantics using an attention-based seq2seq model.
The ASR model uses either the LibriSpeech LM (`train_LS_LM.yaml`) or the Timers and Such LM (`train_TAS_LM.yaml`).

```
cd decoupled
python train.py hparams/{train_LS_LM, train_TAS_LM}.yaml
```

### Multistage recipe
The "multistage" recipe is similar to the decoupled recipe, but instead of using the true transcripts to train the NLU model, we use transcripts predicted by the ASR model, again using either the LibriSpeech LM (`train_LS_LM.yaml`) or the Timers and Such LM (`train_TAS_LM.yaml`).

```
cd multistage
python train.py hparams/{train_LS_LM, train_TAS_LM}.yaml
```

### Direct recipe
The "direct" maps the input speech to directly to semantics using a seq2seq model.
The encoder is pre-trained using the LibriSpeech seq2seq recipe.

```
cd direct
python train.py hparams/train.yaml
```

# Performance summary

[Sentence accuracy on Timers and Such v0.1, measured using 5 random seeds.]
| System | Synthetic | Real |
|----------------- | ------------ | ------|
| Decoupled (LibriSpeech LM) | 18.7% ± 5.1% | 23.6% ± 7.3% |
| Decoupled (Timers and Such LM) | 31.9% ± 3.9% | 44.4% ± 6.9% |
| Multistage (LibriSpeech LM) | 69.9% ± 2.5% | 69.8% ± 3.5% |
| Multistage (Timers and Such LM) | 73.1% ± 8.7% | 75.3% ± 4.2% |
| Direct | 96.1% ± 0.2% | 74.5% ± 6.9% |

You can find the output folder (model, logs, etc) here:
https://drive.google.com/drive/folders/1kSwdBT8kDhnmTLzrOPDL77LX_Eq-3Tzl?usp=sharing
