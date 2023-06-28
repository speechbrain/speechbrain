# Speech Emotion Diarization (SED)

Speech Emotion Diarization ([arXiv link](https://arxiv.org/pdf/2211.01910.pdf)) aims to predict the correct emotions and their temporal boundaries within an utterance.

## Dependencies

First, please install the extra dependencies, do  `pip install -r requirements.txt`


## Datasets

### Test Set
The test set is Zaion Emotion Dataset (ZED), which can be downloaded [here](https://zaion.ai/en/resources/zaion-lab-blog/zaion-emotion-dataset/).

### Training Set
1. [RAVDESS](https://zenodo.org/record/1188976)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/i954d87qqyl055e/RAVDESS.zip`

   <!-- Unzip and rename the folder as "RAVDESS". -->

2. [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/d1xn0l1abroqmkj/ESD.zip`
   <!-- Unzip and rename the folder as "ESD". -->

3. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)


4. [JL-CORPUS](https://www.kaggle.com/datasets/tli725/jl-corpus?resource=download)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/vb7anwzamhkp1yj/JL_corpus.zip`

5. [EmoV-DB](https://openslr.org/115/)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/c7uwbh5iu9vfotf/EmoV-DB.zip`, where only `Amused, Neutral, Angry` emotions are kept.


## Run the code

First download the train/test datasets and unzip them.

To run the code, do `python train_with_wav2vec.py hparams/train_with_wav2vec.yaml --zed_folder /path/to/ZED --emovdb_folder /path/to/EmoV-DB --esd_folder /path/to/ESD --iemocap_folder /path/to/IEMOCAP --jlcorpus_folder /path/to/JL_corpus --ravdess_folder /path/to/RAVDESS`.

The frame-wise classification result for each utterance can be found in `results/eder.txt`.


## Results

The EDER (Emotion Diarization Error Rate) reported here was averaged on 5 different seeds

| model | EDER |
|:-------------:|:---------------------------:|
| Wav2vec2.0-large | 36.2 ± 1.14 |
| HuBERT-large | 34.5 ± 0.81 |
| WavLM-large | 30.2 ± 1.60 |

## Inference

The pretrained models and a easy-inference interface can be found on [HuggingFace](to be added).



# **About Speech Emotion Diarization/Zaion Emotion Dataset**

```bibtex
@article{wang2023speech,
  title={Speech Emotion Diarization: Which Emotion Appears When?},
  author={Wang, Yingzhi and Ravanelli, Mirco and Nfissi, Alaa and Yacoubi, Alya},
  journal={arXiv preprint arXiv:2306.12991},
  year={2023}
}
```

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/

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