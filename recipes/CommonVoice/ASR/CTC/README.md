# CommonVoice ASR with CTC based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# How to run
python train.py hparams/{hparam_file}.py

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset and CTC:
- French
- Kinyarwanda
- English

# Results

| Language | CommonVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| English | 2020-12-11 | train_en_with_wav2vec.yaml | No | 5.01 | 12.57 | 7.32 | 15.58 | Not Avail. | [model](https://drive.google.com/drive/folders/1tYO__An68xrM5pR1UIXzEkwzvKX2Tz2o?usp=sharing) | 2xV100 32GB |
| French | 2020-12-11 | train_fr_with_wav2vec.yaml | No | 2.60 | 8.59 | 3.19 | 9.96 | Not Avail. | [model](https://drive.google.com/drive/folders/1T9DfdZwcNI9CURxhLCi8GA5JVz8adiY8?usp=sharing) | 2xV100 32GB |
| Italian | 2020-12-11 | train_it_with_wav2vec.yaml | No | 2.77 | 9.83 | 3.16 | 10.85 | Not Avail. | [model](https://drive.google.com/drive/folders/1JhlxeA04tWg_vKcNChOoXSnjBe4luRby?usp=sharing) | 2xV100 32GB |
| Kinyarwanda | 2020-12-11 | train_rw_with_wav2vec.yaml | No | 6.20 | 20.07 | 8.25 | 23.12 | Not Avail. | [model](https://drive.google.com/drive/folders/12_BDenvOqEERDZLAN-KdiAHklvuo35tx?usp=sharing) | 2xV100 32GB |


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
