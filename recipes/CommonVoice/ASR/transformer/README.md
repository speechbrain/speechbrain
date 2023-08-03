# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
```shell
python train.py hparams/{hparam_file}.py
```

# How to run on test sets only
```shell
python train.py hparams/{hparam_file}.py --test_only
```
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
| French | 2020-06-22 | train_fr.yaml | No | 5.15 | 17.80 | 6.01 | 19.21 | [model](https://www.dropbox.com/sh/o3bhc0n8h3t9tdg/AADxwA2j-z9a5Y9Um2E6rUwha?dl=0) | 1xV100 16GB |

## Whisper Finetuning Result:
Following table contains whisper-finetuning results for 1 epoch using whisper_large_v2 model, freezing encoder and finetuning decoder.
| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| Arabic | 2023-01-10 | train_ar_hf_whisper.yaml | No | 4.02 | 12.47 | 5.20 | 16.96 | [model](https://www.dropbox.com/sh/45o3xkxdheksdfi/AAAs1zxCw76mcAbudYEonzg0a?dl=0) | 1xV100 16GB |
| Persian | 2023-01-10 | train_fa_hf_whisper.yaml | No | 6.91 | 25.30 | 9.38 | 31.75 | [model](https://www.dropbox.com/sh/a2vd6nn0icybdcz/AAC7z41jcheW1R9aNNK4-lHha?dl=0) | 1xV100 16GB |
| Mongolian | 2023-01-10 | train_mn_hf_whisper.yaml | No | 24.05 | 62.37 | 25.73 | 64.92 | [model](https://www.dropbox.com/sh/2t0srpb2nt2wst5/AACRJQCwooRaLxPoLkmTvKq8a?dl=0) | 1xV100 16GB |
| Hindi | 2023-01-10 | train_hi_hf_whisper.yaml | No | 4.54 | 10.46 | 7.00 | 15.27 | [model](https://www.dropbox.com/sh/qkcm86bzzb1y4sj/AABjA_ckw_hPwJCBzUiXLWrBa?dl=0) | 1xV100 16GB |
| Serbian | 2023-01-10 | train_sr_hf_whisper.yaml | No | 8.92 | 27.12 |  7.60 | 23.63 | [model](https://www.dropbox.com/sh/a798gw3k2ezerp5/AADz7UxvQRQDOH4DnCJ4J4dja?dl=0) | 1xV100 16GB |
| French | 2023-01-10 | train_fr_hf_whisper.yaml | No | 3.00 | 8.95 | 3.83 | 10.62 | [model](https://www.dropbox.com/sh/8c2lpa7m5amasjz/AAD5AZlD6OslhFc0W81D3nosa?dl=0) | 1xV100 16GB |

The output folders with checkpoints and logs can be found [here](https://www.dropbox.com/sh/852eq7pbt6d65ai/AACv4wAzk1pWbDo4fjVKLICYa?dl=0).

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
