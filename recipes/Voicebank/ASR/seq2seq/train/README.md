# VoiceBank ASR with a seq2seq model

This folder contains the scripts to train a RNN speech recognizer based on CTC+ATT using Voicebank.

Use the `download_vctk()` function in the `voicebank_prepare.py` file to
download and resample the dataset.

This model does not converge without pretraining. By default, a model pre-trained
on LibriSpeech 1000h that achieves nearly 3.0% WER is downloaded and fine-tuned.

## How to run

```bash
python train.py hparams/train.yaml
```

## Results

| Release  | hyperparams file | input type  | Test WER | Model link    | GPUs        |
|:--------:|:----------------:|:-----------:|:--------:|:-------------:|:-----------:|
| 21-02-09 | train.yaml       | `clean_wav` | 2.83     | Not Available | 1xV100 32GB |
| 21-02-09 | train.yaml       | `noisy_wav` | 4.36     | Not Available | 1xV100 32GB |

## Training Time

About 5 mins for each epoch with a TESLA V100.
