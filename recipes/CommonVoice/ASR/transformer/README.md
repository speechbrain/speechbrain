# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

## For Whisper finetuning:

python train_with_whisper.py hparams/train_<locale>_hf_whisper.yaml e.g. train_<locale>_hf_whisper

Note: When using whisper large model, to improve memory usage during model recovery. You could use (see https://github.com/speechbrain/speechbrain/pull/1743)

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset
with our transformers:
- French

For Whisper-large-v2 finetuning, here is list of the different language that we tested  within the CommonVoice.10_0 dataset:
- Hindi
- Arabic
- Persian
- Serbian
- Mongolian
- French


# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| French | 2020-06-22 | train_fr.yaml | No | 5.15 | 17.80 | 6.01 | 19.21 | [model](https://drive.google.com/drive/folders/12ny6daoz1Ze1MmgLrsqf352AXvhwob6d?usp=sharing) | 1xV100 16GB |

## Whisper Finetuning Result:
Following table contains whisper-finetuning results for 1 epoch using whisper_large_v2 model, freezing encoder and finetuning decoder.
| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| Arabic | 2023-01-10 | train_ar_hf_whisper.yaml | No | 4.02 | 12.47 | 5.20 | 16.96 | [model](https://drive.google.com/drive/folders/10mYPYfj9NpDNAa0nO16Zd_K1bIEUOIpx?usp=sharing) | 1xV100 16GB |
| Persian | 2023-01-10 | train_fa_hf_whisper.yaml | No | 6.91 | 25.30 | 9.38 | 31.75 | [model](https://drive.google.com/drive/folders/1nzMMYmB5SxMKsFUk-rM9_ijcqzia8pX7?usp=sharing) | 1xV100 16GB |
| Mongolian | 2023-01-10 | train_mn_hf_whisper.yaml | No | 24.05 | 62.37 | 25.73 | 64.92 | [model](https://drive.google.com/drive/folders/10E2xclgNx_6BFxNmv9i1HorBNnsMveP_?usp=sharing) | 1xV100 16GB |
| Hindi | 2023-01-10 | train_hi_hf_whisper.yaml | No | 4.54 | 10.46 | 7.00 | 15.27 | [model](https://drive.google.com/drive/folders/11PKCsyIE703mmDv6n6n_UnD0bUgMPbg_?usp=sharing) | 1xV100 16GB |
| Serbian | 2023-01-10 | train_sr_hf_whisper.yaml | No | 8.92 | 27.12 |  7.60 | 23.63 | [model](https://drive.google.com/drive/folders/1QG67qoekEB29jBd9knt8stLJD4T_xgG7?usp=sharing) | 1xV100 16GB |
| French | 2023-01-10 | train_fr_hf_whisper.yaml | No | 3.00 | 8.95 | 3.83 | 10.62 | [model](https://drive.google.com/drive/folders/1_iI_G-pMYNeyLsvmHPgNR6gPi8zazkF4?usp=sharing) | 1xV100 16GB |

The output folders with checkpoints and logs can be found [here](https://drive.google.com/drive/folders/11NMzY0zV-NqJmPMyZfC3RtT64bYe-G_O?usp=sharing).

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
