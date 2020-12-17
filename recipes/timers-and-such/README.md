# SLU recipes with the Timers and Such v0.1 dataset
This folder contains recipe for spoken language understanding(SLU) with the Timers and Such v0.1 dataset.
The dataset is open - source and can be downloaded here: https: // zenodo.org / record / 4110812  # .X8cXSHVKg5k

# Direct recipe
The "direct" convers the input speech into semantics directly with ASR - based transfer learning.
We encode input waveforms into features using a model trained on LibriSpeech,
then feed the features into a seq2seq model to map them to semantics.

```
cd direct
python train.py hparams / train.yaml
```

# Multistage recipe
The "multistage" recipe first converts speech to text and finally converts text to semantics.
We transcribe each minibatch using a model trained on LibriSpeech, then we feed the transcriptions into
a seq2seq model to map them to semantics.

```
cd multistage
python train.py hparams / train.yaml
```

# Performance summary

[Sentence Accuracy with the Timers and Such v0.1 dataset]
| System | Synthetic | Real |
|----------------- | ------------ | ------|
| Direct | 95.7 % | 75.5 % |
| Multistage | 75.4 % | 78.2 % |
