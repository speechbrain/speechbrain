# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/fr)

# Data preparation
Fortunately, SpeechBrain provides a script that automatically generates the training, validation and testing files. However, it is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Therefore the `commonvoice_prepare.py` script converts these files to .wav and 16KHz.

# Languages covered
Here is a list of the different languages that we tested within the CommonVoice dataset:
- French

# Reported results

| Language | hyperparams file | LM ? | Val. CER | Val. WER | Test CER | Test WER | Model link |
| ------------- |:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:|
| French | CRDNN_fr_best.yaml | No | 7.20 | 16.96 | 8.05 | 18.78 | [Not Available](https://commonvoice.mozilla.org/fr) |
