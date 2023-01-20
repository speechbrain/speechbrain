# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

## For Whisper finetuning:

python train_with_whisper.py hparams/train_<locale>_hf_whisper.yaml e.g. train_<locale>_hf_whisper

Note: When using whisper large model, to improve memory usage during model recovery especiall. you could use (see https://github.com/speechbrain/speechbrain/pull/1743)

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset
with our transformers:
- French

For Whisper finetuning, here is list of the different language that we tested  within the CommonVoice dataset:
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
Following tabl econtains whisper-finetuning results for 1 epoch using whisper_large model when freezing encoder and only fine-tune whisper-decoder.
| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER |Zero-shot Test WER  | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| Arabic | 2023-01-10 | train_ar_hf_whisper.yaml | No | 4.65, | 14.91 | 6.39 | 20.40 | 56.0 | 1xV100 16GB |
| Persian | 2023-01-10 | train_fa_hf_whisper.yaml | No | 8.09 | 28.57 | 10.49 | 34.97 | 44.8 | 1xV100 16GB |
| Mongolian | 2023-01-10 | train_mn_hf_whisper.yaml | No | 25.23 | 65.01 | 27.38 | 67.12 | 117.4 | 1xV100 16GB |
| Hindi | 2023-01-10 | train_hi_hf_whisper.yaml | No | 5.07 | 11.59 | 8.09 | 17.20 | 25.0 | 1xV100 16GB |
| Serbian | 2023-01-10 | train_sr_hf_whisper.yaml | No | 14.55 | 35.65 | 13.33 | 32.54 | 87.4 | 1xV100 16GB |
| French | 2023-01-10 | train_fr_hf_whisper.yaml | No | 3.5 | 10.36 | 4.42 | 12.34 | 14.7 | 1xV100 16GB |

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
