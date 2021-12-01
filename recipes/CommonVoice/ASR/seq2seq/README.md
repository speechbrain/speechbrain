# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

Make sure you have "transformers" installed if you use the wav2vec2 fine-tuning model.

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset:
- French
- Kinyarwanda
- Italian
- English

# Results

| Language | CommonVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| French | 2020-12-11 | train_fr.yaml | No | 5.22 | 13.92 | 6.43 | 15.99 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-fr) | [model](https://drive.google.com/drive/folders/1GShpLaX9AOLklwBAOLB9B9lLTn0HsWwS?usp=sharing) | 2xV100 16GB |
| French | 2020-12-11 | train_fr_with_wav2vec.yaml | No | 6.13 | 11.82 | 9.78 | 13.34 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr) | [model](https://drive.google.com/drive/folders/1tjz6IZmVRkuRE97E7h1cXFoGTer7pT73?usp=sharing) | 2xV100 32GB |
| Kinyarwanda | 2020-12-11 | train_rw.yaml | No | 7.30 | 21.36 | 9.55 | 24.27 | Not Avail. | [model](https://drive.google.com/drive/folders/122efLUMYoc1LGoK7O6LIWkSklmjKVGxM?usp=sharing) | 2xV100 32GB |
| Kinyarwanda | 2020-12-11 | train_rw_with_wav2vec.yaml | No | 5.08 | 15.88 | 8.33 | 18.91 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-rw) | [model](https://drive.google.com/drive/folders/1ceHxyNojY0wXmXyPoyn9xUiH_5B5qgE4?usp=sharing) | 2xV100 16GB |
| English | 2020-12-11 | train_en.yaml | No | 8.66 | 20.16 | 12.93 | 24.89 | Not Avail. | [model](https://drive.google.com/drive/folders/1FAKRhfu_1gLnkshYGKp-6G9ZVMIUlv9n?usp=sharing) | 2xV100 16GB |
| English | 2020-12-11 | train_en_with_wav2vec.yaml | No | 14.50 | 13.21 | 24.65 | 15.69 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) | [model](https://drive.google.com/drive/folders/1EfIZiJi8ch53mil9K4tn46OrmTJq5WYj?usp=sharing) | 2xV100 32GB |
| Italian | 2020-12-11 | train_it.yaml | No | 5.14 | 15.59 | 15.40 | 16.61 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-it) | [model](https://drive.google.com/drive/folders/1asxPsY1EBGHIpIFhBtUi9oiyR6C7gC0g?usp=sharing) | 2xV100 16GB |
| Italian | 2020-12-11 | train_it_with_wav2vec.yaml | No | 3.11 | 8.30 | 5.75 | 9.86 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-it) | [model](https://drive.google.com/drive/folders/1LKA50Qsr1fM1E3t4PHMWUjlBMS2QGFHj?usp=sharing) | 2xV100 16GB |
| German | 2021-10-28 | train_de.yaml | No | 4.32 | 13.99 | 4.93 | 15.37 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-de) | -- | 1x V100 16GB |

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
