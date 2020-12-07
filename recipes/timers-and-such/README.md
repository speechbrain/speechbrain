# SLU recipes for Timers and Such v0.1
This folder contains recipes for spoken language understanding (SLU) with [Timers and Such v0.1](https://zenodo.org/record/4110812), an SLU dataset with a large (train/dev/test) set of synthetic speech and a small (train/dev/test) set of real speech.

### Decoupled recipe
The "decoupled" recipe uses an ASR model (using the LibriSpeech seq2seq recipe) to map speech to text and a separate NLU model, trained on the true transcripts rather than the ASR output, to map text to semantics using an attention-based seq2seq model.

```
cd decoupled
python train.py hparams/train.yaml
```

### Multistage recipe
The "multistage" recipe is similar to the decoupled recipe, but instead of using the true transcripts to train the NLU model, we use transcripts predicted by the ASR model.

```
cd multistage
python train.py hparams/train.yaml
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
| Decoupled (Timers and Such LM) | ? % | ? % |
| Multistage (LibriSpeech LM) | 69.9% ± 2.5% | 69.8% ± 3.5% |
| Multistage (Timers and Such LM) | ? % | ? % |
| Direct | 96.1% ± 0.2% | 74.5% ± 6.9% |
