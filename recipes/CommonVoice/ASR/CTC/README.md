# CommonVoice ASR with CTC based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.yaml

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset and CTC:
- English
- German
- French
- Italian
- Kinyarwanda

# Results
| Language | CommonVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| English | 2020-12-11 | train_en_with_wav2vec.yaml | No | 5.01 | 12.57 | 7.32 | 15.58 | Not Avail. | [model](https://www.dropbox.com/sh/o3q43r4wdovbmnd/AADXcVomQr549NdAgCpI7OQHa?dl=0) | 2xV100 32GB |
| German | 2022-08-16 | train_de_with_wav2vec.yaml | No | 1.90 | 8.02 | 2.40 | 9.54 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-de) | [model](https://www.dropbox.com/sh/vdz7apt16nbq94g/AADI5o23Ll_NmjiPlg9bzPjta?dl=0) | 1xRTXA6000 48GB |
| French | 2020-12-11 | train_fr_with_wav2vec.yaml | No | 2.60 | 8.59 | 3.19 | 9.96 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr) | [model](https://www.dropbox.com/sh/wytlbeddrt8oe4n/AAAY59qMsDlWy5F017bmBeVua?dl=0) | 2xV100 32GB |
| Italian | 2020-12-11 | train_it_with_wav2vec.yaml | No | 2.77 | 9.83 | 3.16 | 10.85 | Not Avail. | [model](https://www.dropbox.com/sh/0v2o2hmrv1j33p6/AAA3xUiqKbSKsX88fWfptPmFa?dl=0) | 2xV100 32GB |
| Kinyarwanda | 2020-12-11 | train_rw_with_wav2vec.yaml | No | 6.20 | 20.07 | 8.25 | 23.12 | Not Avail. | [model](https://www.dropbox.com/sh/ccgirbq9r8uzubi/AAAynCvEV8EjEpMavFRPp87Ta?dl=0) | 2xV100 32GB |

*For German, it takes around 5.5 hrs an epoch.* <br>
The output folders with checkpoints and logs can be found [here](https://www.dropbox.com/sh/852eq7pbt6d65ai/AACv4wAzk1pWbDo4fjVKLICYa?dl=0).

## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for CommonVoice.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
Footer
