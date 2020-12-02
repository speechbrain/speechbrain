# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/fr)

# Data preparation
Fortunately, SpeechBrain provides a script that automatically generates the training, validation and testing files. However, it is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Therefore the `commonvoice_prepare.py` script converts these files to .wav and 16KHz.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset:
- French
- Kinyarwanda
- Italian

# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| French | 2020-06-22 | CRDNN_fr.yaml | No | 6.70 | 16.35 | 7.61 | 18.22 | [Download](https://drive.google.com/file/d/1a7J80SIB5rCgZ4PtA_ydwqE5G-FzShdc/view?usp=sharing) | 2xV100 16GB |
| Kinyarwanda | 2020-06-22 | CRDNN_rw.yaml | No | 11.09 | 30.91 | 16.53 | 38.78 | [Download](https://drive.google.com/file/d/11tRQrl7aEktiiwbSM3eKQ_mIYGFHMBCH/view?usp=sharing) | 2xV100 16GB |
| Italian | 2020-06-22 | CRDNN_it.yaml | No | 9.34 | 25.42 | 10.04 | 26.78 | [Download](https://drive.google.com/file/d/1zm7tJKfs5E6-65uKuLJHS6GjObbLugz0/view?usp=sharing) | 2xV100 16GB |
