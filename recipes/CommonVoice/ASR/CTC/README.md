# CommonVoice ASR with CTC based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset and CTC:
- French
- Kinyarwanda
- English

# Results

| Language | CommonVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| English | 2020-12-11 | train_en_with_wav2vec.yaml | No | 5.01 | 12.57 | 7.32 | 15.58 | Not Avail. | [model]() | 2xV100 32GB |
| French | 2020-12-11 | train_fr_with_wav2vec.yaml | No | 3.70 | 12.98 | 4.42 | 14.44 | Not Avail. | [model]() | 2xV100 32GB |
| French | 2020-12-11 | train_it_with_wav2vec.yaml | No | 2.81 | 9.81 | 3.21 | 10.93 | Not Avail. | [model]() | 2xV100 32GB |
| Kinyarwanda | 2020-12-11 | train_rw_with_wav2vec.yaml | No | 6.20 | 20.07 | 8.25 | 23.12 | Not Avail. | [model]() | 2xV100 32GB |


## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for CommonVoice.
