# SLU recipes for SLURM
This folder contains recipes for spoken language understanding (SLU) with [SLURM](https://zenodo.org/record/4274930#.YEFCYHVKg5k).

### Direct recipe
The "direct" maps the input speech directly to semantics using a seq2seq model.
The encoder is pre-trained using the LibriSpeech seq2seq recipe.

```
cd direct
python train.py hparams/train.yaml
```

### NLU recipe
This text-only recipe maps the gold transcriptions into the semantics directly.

# Performance summary

[Sentence accuracy on SLURM]

| System | Synthetic | Real | Link |
|----------------- | ------------ | ------|--------|
| Direct | -  | -  |--------|
| NLU | -  | -  |--------|


