# LibriSpeech ASR with Transformers.
This folder contains the scripts to train a Transformer-based speech recognizer
using LibriSpeech.

You can download LibriSpeech at http://www.openslr.org/12


# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 20-05-22 | transformer.yaml | 2.46 | [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | [GoogleDrive](https://drive.google.com/drive/folders/1ZudxqMWb8VNCJKvY2Ws5oNY3WI1To0I7?usp=sharing) | 2xRTX8000 42GB |
| 20-05-22 | conformer.yaml | -.-- | Not Available | Not Available | 4xV100 32GB |
