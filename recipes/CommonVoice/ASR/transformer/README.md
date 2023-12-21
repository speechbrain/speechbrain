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
- French
- Italian
- German

For Whisper-large-v2 and medium finetuning, here is list of the different language that we tested  within the CommonVoice.14_0 dataset:
- Hindi
- Arabic
- Persian
- Serbian
- Mongolian
- French
- Italian


# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Hugging Face link |  Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:|:-----------:| :-----------:| :-----------:|
| French | 2023-08-15 | train_fr.yaml | No | 5.41 | 16.00 | 5.41 | 17.61 | - | [model](https://www.dropbox.com/sh/zvu9h9pctksnuvp/AAD1kyS3-N0YtmcoMgjM-_Tba?dl=0) | 1xV100 32GB |
| Italian | 2023-08-15 | train_it.yaml | No | 3.72 | 16.31 | 4.01 | 16.80 | - | [model](https://www.dropbox.com/sh/yy8du12jgbkm3qe/AACBHhTCM-cU-oGvAKJ9kTtaa?dl=0) | 1xV100 32GB |
| German | 2023-08-15 | train_de.yaml | No | 3.60 | 15.33 | 4.22 | 16.76 |- | [model](https://www.dropbox.com/sh/umfq986o3d9o1px/AAARNF2BFYELOWx3xhIOEoZka?dl=0) | 1xV100 32GB |

## Whisper Finetuning Result:
Following table contains whisper-finetuning results for 1 epoch using whisper_medium model, freezing encoder and finetuning decoder.
| Language | Release | Model | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:| -----:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------: |:-----------:| :-----------:|
| Arabic | 2023-08-15 | large-v2 | train_ar_hf_whisper.yaml | No | 4.02 | 12.47 | 5.20 | 16.96 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-ar) | [model](https://www.dropbox.com/sh/45o3xkxdheksdfi/AAAs1zxCw76mcAbudYEonzg0a?dl=0) | 1xV100 16GB |
| Persian | 2023-08-15 | large-v2 | train_fa_hf_whisper.yaml | No | 6.91 | 25.30 | 9.38 | 31.75 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fa) | [model](https://www.dropbox.com/sh/a2vd6nn0icybdcz/AAC7z41jcheW1R9aNNK4-lHha?dl=0) | 1xV100 16GB |
| Mongolian | 2023-08-15 | large-v2 | train_mn_hf_whisper.yaml | No | 24.05 | 62.37 | 25.73 | 64.92 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-mn) | [model](https://www.dropbox.com/sh/2t0srpb2nt2wst5/AACRJQCwooRaLxPoLkmTvKq8a?dl=0) | 1xV100 16GB |
| Hindi | 2023-08-15 | large-v2 | train_hi_hf_whisper.yaml | No | 4.54 | 10.46 | 7.00 | 15.27 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-hi) | [model](https://www.dropbox.com/sh/qkcm86bzzb1y4sj/AABjA_ckw_hPwJCBzUiXLWrBa?dl=0) | 1xV100 16GB |
| Serbian | 2023-08-15 | large-v2 | train_sr_hf_whisper.yaml | No | 8.92 | 27.12 |  7.60 | 23.63 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-sr) | [model](https://www.dropbox.com/sh/a798gw3k2ezerp5/AADz7UxvQRQDOH4DnCJ4J4dja?dl=0) | 1xV100 16GB |
| French | 2023-08-15 | large-v2 | train_fr_hf_whisper.yaml | No | 3.00 | 8.95 | 3.83 | 10.62 | [model](https://huggingface.co/speechbrain/asr-whisper-large-v2-commonvoice-fr) | [model](https://www.dropbox.com/sh/8c2lpa7m5amasjz/AAD5AZlD6OslhFc0W81D3nosa?dl=0) | 1xV100 16GB |
| Arabic | 2023-08-15 | Medium | train_ar_hf_whisper.yaml | No | 4.95 | 14.82 | 6.51 | 20.24 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-ar) | [model](https://www.dropbox.com/sh/0e4vtvbg6hf2e13/AAD-tfzCZGUrh85aeAeJj8I9a?dl=0) | 1xV100 16GB |
| Persian | 2023-08-15 | Medium | train_fa_hf_whisper.yaml | No | 8.58 | 35.48 | 11.27 | 35.48 |[model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-fa) | [model](https://www.dropbox.com/sh/w1urihacmtoulmi/AADMtK3qeAF5mLYk5LMHyiOra?dl=0) | 1xV100 16GB |
| Mongolian | 2023-08-15 | Medium | train_mn_hf_whisper.yaml | No |  27.08 |  67.41 | 27.69 | 67.84 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-mn) | [model](https://www.dropbox.com/sh/6fbhmey7q1udykf/AAAiGObWTTe2cdXHt2Uv2VQXa?dl=0) | 1xV100 16GB |
| Hindi | 2023-08-15 | Medium | train_hi_hf_whisper.yaml | No | 5.82 | 12.51 | 8.16 | 17.04 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-hi) | [model](https://www.dropbox.com/sh/z9vriyy3i6xqvif/AAB7ql-40yWTjKEQJiuhYUr5a?dl=0) | 1xV100 16GB |
| Serbian | 2023-08-15 | Medium | train_sr_hf_whisper.yaml | No | 8.63 | 25.10 |  7.25 | 22.29 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-sr) | [model](https://www.dropbox.com/sh/5lhk230q45sd97z/AAD-U9b_Ws_vFPs-cazsbOY0a?dl=0) | 1xV100 16GB |
| French | 2023-08-15 | Medium | train_fr_hf_whisper.yaml | No | 3.26 | 9.65 | 4.30 | 11.79 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-fr) | [model](https://www.dropbox.com/sh/7zlk07yxnslk4yy/AAANcI3EaG0ZFy6UrKk1Mm2Ga?dl=0) | 1xV100 16GB |
| Italian | 2023-08-15 | Medium | train_it_hf_whisper.yaml | No | 2.42 | 8.26 | 3.03 | 9.63 | [model](https://huggingface.co/speechbrain/asr-whisper-medium-commonvoice-it) | [model](https://www.dropbox.com/sh/u5tex3nvzzs5pex/AAD-J7cOBE_fNfBono8waTKCa?dl=0) | 1xV100 16GB |

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
