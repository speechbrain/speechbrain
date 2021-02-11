# VoiceBank ASR with CTC models

This folder contains the scripts to train a CTC acoustic model using Voicebank.

Use the `download_vctk()` function in the `voicebank_prepare.py` file to
download and resample the dataset.

## How to run

```bash
python train.py hparams/train.yaml
```

## Results

| Release  | hyperparams file | input type  | Test PER | Model link    | GPUs        |
|:--------:|:----------------:|:-----------:|:--------:|:-------------:|:-----------:|
| 21-02-09 | train.yaml       | `clean_wav` | 10.12    | Not Available | 1xV100 32GB |

## Training Time

About 4 mins for each epoch with a TESLA V100.
