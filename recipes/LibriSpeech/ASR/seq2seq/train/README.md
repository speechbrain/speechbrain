# LibriSpeech ASR with seq2seq models (CTC + attention).
This folder contains the scripts to train a seq2seq CNN-RNN-based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train.py hparams/file.yaml

# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 01-03-21 | train_BPE_1000.yaml | 3.08 | [HuggingFace](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) | Not Available| 1xV100 32GB |
| 01-03-21 | train_BPE_5000.yaml | 2.89 | [HuggingFace](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) | Not Available | 1xV100 32GB |
