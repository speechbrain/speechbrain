# Recipes for ASR with AISHELL-1
This folder contains recipes for tokenization and speech recognition with [AISHELL-1](https://www.openslr.org/33/), a 150-hour Chinese ASR dataset.

### Base recipe
The recipe uses an attention-based CRDNN sequence-to-sequence model with an auxiliary CTC loss.
Outputs are tokenized using a 5000-token SentencePiece tokenizer.
This recipe is based on the LibriSpeech recipe, but uses no external language model, which is the convention for this dataset.

```
cd ASR/train
python train.py hparams/train.yaml
```

# Performance summary
Results are reported in terms of Character Error Rate (CER). It is not clear from published results whether spaces are kept or removed when computing CER; we therefore report results for both settings. (This can be controlled using `python train.py hparams/train.yaml --remove_spaces={True,False}`.)

[Test Character Error Rate (CER).]
| System | Test CER |
|----------------- | ------|
| Base (remove spaces) | 7.76 |
| Base (keep spaces) | 7.51 |
