# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset:
- French
- Kinyarwanda
- Italian
- English

# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| French | 2020-06-22 | train_fr.yaml | No | 5.36 | 15.87 | 6.54 | 17.70 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-fr) | [model](https://drive.google.com/drive/folders/13i7rdgVX7-qZ94Rtj6OdUgU-S6BbKKvw?usp=sharing) | 2xV100 16GB |
| French | 2021-04-23 | train_fr_with_wav2vec.yaml | No | 6.09 | 12.47 | 9.62 | 13.93 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr) | [model](https://drive.google.com/drive/folders/1tjz6IZmVRkuRE97E7h1cXFoGTer7pT73?usp=sharing) | 2xV100 32GB |
| Kinyarwanda | 2020-06-22 | train_rw.yaml | No | 7.30 | 21.36 | 9.55 | 24.27 | Not Avail. | [model](https://drive.google.com/drive/folders/122efLUMYoc1LGoK7O6LIWkSklmjKVGxM?usp=sharing) | 2xV100 16GB |
| Italian | 2020-06-22 | train_it.yaml | No | 5.14 | 15.59 | 5.40 | 16.61 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-it) | [model](https://drive.google.com/drive/folders/1asxPsY1EBGHIpIFhBtUi9oiyR6C7gC0g?usp=sharing) | 2xV100 16GB |
| English | 2020-06-22 | train_en.yaml | No | 8.66 | 20.16 | 12.93 | 24.89 | Not Avail. | [model](https://drive.google.com/drive/folders/1FAKRhfu_1gLnkshYGKp-6G9ZVMIUlv9n?usp=sharing) | 2xV100 16GB |

## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for CommonVoice.
