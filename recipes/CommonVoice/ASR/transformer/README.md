# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice 14.0 dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/) and pytorch 2.0
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
- Italian
- French

For Whisper-large-v2 and medium finetuning, here is list of the different language that we tested  within the CommonVoice.14_0 dataset:
- Hindi
- Arabic
- Persian
- Serbian
- Mongolian
- French
- Italian


# Results
## Transformer
| Language | CV version | hyperparams file |  LM | Val. CER | Val. WER | Test CER | Test WER | Hugging Face link |  Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:|:-----------:| :-----------:| :-----------:|
| English | 16.1 | mwmha_transformer_large.yaml | No | 4.72 | 10.97 | 6.68 | 13.69 | - | [model](https://1drv.ms/f/c/039f8ffe91e06416/Et7KEbSlWNdJhkjLIi7_vGQBMVhGwRRBzCSljh6aA4sJSw?e=dXeuiY) | 1xL40 48GB |
| English | 16.1 | conformer_large.yaml | No | 4.48 | 10.48 | 6.42 | 13.39 | - | [model](https://www.dropbox.com/scl/fo/3w24pxln0fjyofl6xbfv1/AJJqzWfCtGFFTRLwM3DeZG8?rlkey=wpzzhizreedptts64d2m9jq4u&st=xu5g9an8&dl=0) | 4xA40 46GB |
| Italian | 14.0 | conformer_large.yaml | No | 2.91 | 9.79 | 2.68 | 9.27 | - | [model](https://www.dropbox.com/scl/fo/tf44itp8f4icf2z5qlxpm/AIOYS_CMov5ss5Q9AonFEno?rlkey=xek5ikbhqoovcao31iniqimrr&dl=0) | 2xV100 32GB |
| French | 14.0 | conformer_large.yaml | No | 2.64 | 7.62 | 3.55 | 9.48 | - | [model](https://www.dropbox.com/scl/fo/y862nl95zoe4sj3347095/ACxmT3_uw1ScLoYs0DSbGRM?rlkey=q66dk13w5nu1lkphtdinnnigm&dl=0) | 2xV100 32GB |

### **About MW-MHA Transformer**
Multi-Window Multi-Head Attention (MW-MHA) is a new Multi-Head attention module where the constituent individual attention heads operate on different local sizes of the input sequence, capturing local-global dependencies more effectively. The method was proposed in the paper "Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners" by Yadav et al. (2024), where it was shown to capture better local-global dependencies when learning general-purpose audio representations.

Here, we simply replaced the standard MHA in the transformer encoder with MW-MHA, achieving performance quite close to that of a Conformer model with no additional parameters or modifications. You can learn more about MW-MHA through the following links:

- Paper: https://openreview.net/forum?id=Q53QLftNkA
- Code: https://github.com/SarthakYadav/mwmae-jax-official

If you use MW-MHA in your work, please cite the following paper:

```bibtex
@inproceedings{
  yadav2024masked,
  title={Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners},
  author={Sarthak Yadav and Sergios Theodoridis and Lars Kai Hansen and Zheng-Hua Tan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Q53QLftNkA}
  }
```

## Whisper Finetuning
Following table contains whisper-finetuning results for 1 epoch using Whisper model, freezing encoder and finetuning decoder.
| Language | Release | Model | commit hash | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:| -----:|-----:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------: |:-----------:| :-----------:|
| French | 2024-03-28 | large-v3 | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 2.31% | 7.38% | 3.11% | 9.09% | x | [DropBox](https://www.dropbox.com/scl/fo/erwh83bg2jbzf3bf8v6ur/AHmQ5i8uWRaieXCOe5DSRUk?rlkey=kjivz2hx3o1pi7wbzadjznpid&dl=0) | 2xV100 32GB |
| Italian | 2024-03-28 | large-v3 | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 1.27% | 4.85% | 1.62% | 5.47% | x | [DropBox](https://www.dropbox.com/scl/fo/gtfo3qoz1ceg4xg0dfq1d/AIabz2J9NxkNAEbGF7rHCHU?rlkey=eokq2a2z07ke48scazqnn5v73&dl=0) | 2xV100 32GB |
| French | 2024-03-28 | medium | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 2.92% | 8.90% | 4.02% | 11.07% | x | [DropBox](https://www.dropbox.com/scl/fo/72aiaflc9w6168rk9jv6u/AGIVW5ml74wZYED7HUFjX-U?rlkey=nz7eo6i6gbze7rwv8la6sxobx&dl=0) | 2xV100 32GB |
| Italian | 2024-03-28 | medium | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 2.05% | 7.17% | 2.31% | 7.79% | x | [DropBox](https://www.dropbox.com/scl/fo/sso9k4n6hma9cub44oi2p/AKINkGK0XMCYND-JrMQh4LQ?rlkey=gywsgxle4k473z9c7tf4l1m7n&dl=0) | 2xV100 32GB |
| French | 2024-03-28 | small | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 4.34% | 12.57% | 5.89% | 15.46% | x | [DropBox](https://www.dropbox.com/scl/fo/h8idsgzp8xz5vsupqv0q8/ACS13H9awYU2G7DeTcyxiV0?rlkey=bbqpx0lbf5aify6ib029g2gn0&dl=0) | 2xV100 32GB |
| Italian | 2024-03-28 | small | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 3.20% | 11.40% | 3.71% | 12.25% | x | [DropBox](https://www.dropbox.com/scl/fo/o4objjm5c65c5hzy1vvk4/ABXA2V1Gy1GCg7FGS6Ty9yc?rlkey=4kbjmmljdznvureyxfip5tw8q&dl=0) | 2xV100 32GB |
| Arabic | 2023-08-15 | large-v2 | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) | train_ar_hf_whisper.yaml | No | 4.02 | 12.47 | 5.20 | 16.96 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-ar) | [model](https://www.dropbox.com/sh/45o3xkxdheksdfi/AAAs1zxCw76mcAbudYEonzg0a?dl=0) | 1xV100 16GB |
| Persian | 2023-08-15 | large-v2 | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_fa_hf_whisper.yaml | No | 6.91 | 25.30 | 9.38 | 31.75 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fa) | [model](https://www.dropbox.com/sh/a2vd6nn0icybdcz/AAC7z41jcheW1R9aNNK4-lHha?dl=0) | 1xV100 16GB |
| Mongolian | 2023-08-15 | large-v2 | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_mn_hf_whisper.yaml | No | 24.05 | 62.37 | 25.73 | 64.92 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-mn) | [model](https://www.dropbox.com/sh/2t0srpb2nt2wst5/AACRJQCwooRaLxPoLkmTvKq8a?dl=0) | 1xV100 16GB |
| Hindi | 2023-08-15 | large-v2 | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_hi_hf_whisper.yaml | No | 4.54 | 10.46 | 7.00 | 15.27 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-hi) | [model](https://www.dropbox.com/sh/qkcm86bzzb1y4sj/AABjA_ckw_hPwJCBzUiXLWrBa?dl=0) | 1xV100 16GB |
| Serbian | 2023-08-15 | large-v2 | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_sr_hf_whisper.yaml | No | 8.92 | 27.12 |  7.60 | 23.63 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-sr) | [model](https://www.dropbox.com/sh/a798gw3k2ezerp5/AADz7UxvQRQDOH4DnCJ4J4dja?dl=0) | 1xV100 16GB |
| Arabic | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_ar_hf_whisper.yaml | No | 4.95 | 14.82 | 6.51 | 20.24 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-ar) | [model](https://www.dropbox.com/sh/0e4vtvbg6hf2e13/AAD-tfzCZGUrh85aeAeJj8I9a?dl=0) | 1xV100 16GB |
| Persian | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_fa_hf_whisper.yaml | No | 8.58 | 35.48 | 11.27 | 35.48 |[model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-fa) | [model](https://www.dropbox.com/sh/w1urihacmtoulmi/AADMtK3qeAF5mLYk5LMHyiOra?dl=0) | 1xV100 16GB |
| Mongolian | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_mn_hf_whisper.yaml | No |  27.08 |  67.41 | 27.69 | 67.84 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-mn) | [model](https://www.dropbox.com/sh/6fbhmey7q1udykf/AAAiGObWTTe2cdXHt2Uv2VQXa?dl=0) | 1xV100 16GB |
| Hindi | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_hi_hf_whisper.yaml | No | 5.82 | 12.51 | 8.16 | 17.04 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-hi) | [model](https://www.dropbox.com/sh/z9vriyy3i6xqvif/AAB7ql-40yWTjKEQJiuhYUr5a?dl=0) | 1xV100 16GB |
| Serbian | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_sr_hf_whisper.yaml | No | 8.63 | 25.10 |  7.25 | 22.29 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-sr) | [model](https://www.dropbox.com/sh/5lhk230q45sd97z/AAD-U9b_Ws_vFPs-cazsbOY0a?dl=0) | 1xV100 16GB |
| French | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_fr_hf_whisper.yaml | No | 3.26 | 9.65 | 4.30 | 11.79 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-fr) | [model](https://www.dropbox.com/sh/7zlk07yxnslk4yy/AAANcI3EaG0ZFy6UrKk1Mm2Ga?dl=0) | 1xV100 16GB |
| Italian | 2023-08-15 | Medium | [b112860](https://github.com/speechbrain/speechbrain/pull/2254/commits/b1128604e040d43e80e9a3214c5116f34d5806db) |train_it_hf_whisper.yaml | No | 2.42 | 8.26 | 3.03 | 9.63 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-it) | [model](https://www.dropbox.com/sh/u5tex3nvzzs5pex/AAD-J7cOBE_fNfBono8waTKCa?dl=0) | 1xV100 16GB |

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
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
