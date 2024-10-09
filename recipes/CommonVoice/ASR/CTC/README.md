# CommonVoice ASR with CTC based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice 14.0 dataset

# How to run
python train.py hparams/{hparam_file}.yaml

To use an n-gram Language Model (LM) for decoding, follow these steps:
1. Uncomment the line `kenlm_model_path: none` in the `test_beam_search` entry in the yaml file.
2. Set a path to an ARPA or bin file containing the n-gram LM.

For training an n-gram LM in ARPA (or bin) format, refer to the LM recipe in recipes/CommonVoice/LM.
Alternatively, you can download a pre-trained n-gram LM from our Dropbox repository at this link: [Pretrained n-gram LMs](https://www.dropbox.com/scl/fo/zw505t10kesqpvkt6m3tu/h?rlkey=6626h1h665tvlo1mtekop9rx5&dl=0).

These models are trained on the Commonvoice audio transcriptions available in the training set.

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset and CTC:
- English
- German
- French
- Italian
- Kinyarwanda
- Arabic
- Spanish
- Portuguese
- Chinese(china)

>>Note:
>In our experiments,  we use CTC beam search and also boost the performance using the 5-gram model previously trained
on the transcription of the training data.(Refer to LM recipe: recipes/CommonVoice/LM).

>>Note:
> For Chinese the concept of word is not well-defined, hence, we consider the character error rate instead of the word error rate. For the same reason,  we don't also employ 5-gram.

# Results
| Language | CommonVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| English | 2024-01-05 | train_en_with_wav2vec.yaml | 5-gram | 3.79 | 10.79 | 4.96 | 11.37 |  | [model](https://www.dropbox.com/scl/fo/gx0szpbectig2r6r6p9vk/APdoN_wWWq_wP4My7w6SvMo?rlkey=v8fhd887bn947yjb45i99wm8p&st=6muft51b&dl=0) | 4xA40 46GB |
| German | 2023-08-15 | train_de_with_wav2vec.yaml | No | 1.74 | 7.40 | 2.18 | 8.39 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-de) | [model](https://www.dropbox.com/sh/dn7plq4wfsujsi1/AABS1kqB_uqLJVkg-bFkyPpVa?dl=0) | 1xV100 32GB |
| French | 2023-08-15 | train_fr_with_wav2vec.yaml | No | 2.59 | 8.47 | 3.36 | 9.71 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-fr) | [model](https://www.dropbox.com/sh/0i7esfa8jp3rxpp/AAArdi8IuCRmob2WAS7lg6M4a?dl=0) | 1xV100 32GB |
| Italian | 2023-08-15 | train_it_with_wav2vec.yaml | No | 2.10 | 7.77 |  2.30 | 7.99 |[model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-it) | [model](https://www.dropbox.com/sh/hthxqzh5boq15rn/AACftSab_FM6EFWWPgHpKw82a?dl=0) | 1xV100 32GB |
| Kinyarwanda | 2023-08-15 | train_rw_with_wav2vec.yaml | No | 5.47 | 19.58 | 7.30 | 22.52 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-rw) | [model](https://www.dropbox.com/sh/4iax0l4yfry37gn/AABuQ31JY-Sbyi1VlOJfV7haa?dl=0) | 1xV100 32GB |
| Arabic | 2023-08-15 | train_ar_with_wav2vec.yaml | No | 6.45 | 20.80 | 9.65 | 28.53 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-ar) | [model](https://www.dropbox.com/sh/7tnuqqbr4vy96cc/AAA_5_R0RmqFIiyR0o1nVS4Ia?dl=0) | 1xV100 32GB |
| Spanish | 2023-08-15 | train_es_with_wav2vec.yaml | No | 3.36 | 12.61 | 3.67 | 12.67 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-es) | [model](https://www.dropbox.com/sh/ejvzgl3d3g8g9su/AACYtbSWbDHvBr06lAb7A4mVa?dl=0) | 1xV100 32GB |
| Portuguese | 2023-08-15 | train_pt_with_wav2vec.yaml | No | 6.26 | 21.05 | 6.63 | 21.69 | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-pt) | [model](https://www.dropbox.com/sh/80wucrvijdvao2a/AAD6-SZ2_ZZXmlAjOTw6fVloa?dl=0) | 1xV100 32GB |
| Chinese(china) | 2023-08-15 | train_zh-CN_with_wav2vec.yaml | No | 25.03 | - | 23.17 | - | [model](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-zh-CN) | [model](https://www.dropbox.com/sh/2bikr81vgufoglf/AABMpD0rLIaZBxjtwBHgrNpga?dl=0) | 1xV100 32GB |


## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for CommonVoice.

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
Footer
