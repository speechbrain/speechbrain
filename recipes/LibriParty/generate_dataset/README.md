## LibriParty synthetic dataset.
A synthetic, **cocktail-party/meeting scenario dataset** useful for fast experimentation derived
from [LibriSpeech](https://openslr.org/12/) [1].

It is an enhanced version of what has been used in [2] for End-to-End Diarization ([EEND](https://github.com/hitachi-speech/EEND)),
with addition of "impulsive" noises and Room Impulsive Responses from [https://openslr.org/28/](https://openslr.org/28/)
(reference article [3]), and,
**optionally**, background noises from [QUT-NOISE-TIMIT](https://github.com/qutsaivt/QUT-NOISE) [4]. Also we allow for finer control
over Signal-To-Noise Ratio (SNR) and other parameters.

It is suitable for a wide range of tasks including:

- Voice Activity Detection
- Overlapped Speech Detection
- Automatic Speech Recognition
- Diarization

---
### Instructions:
This recipe offers **"custom" dataset creation** or **"official" dataset creation**, by downloading pre-created,
**"official" metadata files** from [link](https://www.dropbox.com/s/0u6x6ndyedb4rl7/LibriParty_metadata.zip?dl=1) and creating the dataset from this metadata.

#### From official metadata:
"Official" dataset uses LibriSpeech train-clean-100 for training, dev-clean for development and test-clean for test.
It also requires background QUT-TIMIT noises. The metadata are downloaded from the web to make sure the data generation is replicable.

- Step 1:  **use provided script** for downloading and resampling sources:
    - run *download_required_data.py* to download all required data automatically.

- Step 1 (Alternative):  **manual download**:
    - download LibriSpeech train-clean-100, dev-clean and test-clean from [https://openslr.org/12/](https://openslr.org/12/).
    - download noises and impulse responses from [link](https://openslr.org/28/).
    - download background QUT noise from [link](https://github.com/qutsaivt/QUT-NOISE).
    - resample QUT noise sources to 16 kHz using *local/resample_folder.py*.
    - download "official" metadata from [link](https://www.dropbox.com/s/0u6x6ndyedb4rl7/LibriParty_metadata.zip?dl=1).

- step 2: change the paths accordingly in *dataset.yaml*.
        You need to specify *metadata_folder*, *out_folder* and paths to downloaded source datasets:
        Librispeech, noises and impulse responses and QUT noise.

- step 3: run *get_dataset_from_metadata.py*

#### Custom:
Follow the next steps to create a novel LibriParty datasets.

- Step 1, **manual download**:
    - download LibriSpeech train-clean-100, dev-clean and test-clean from [https://openslr.org/12/](https://openslr.org/12/).
    - download noises and impulse responses from [here](https://openslr.org/28/).
    - download background QUT noise from [here](https://github.com/qutsaivt/QUT-NOISE).
    - resample QUT noise sources to 16 kHz using *local/resample_folder.py*.

- Step 1 (Alternative), **use provided script** for downloading and resampling sources:
    - run *download_required_data.py* to download all required data automatically.

- step 2: change the paths accordingly in *dataset.yaml*.
           In particular path to downloaded Librispeech dataset,
           rirs and noises dataset and background noise dataset.

- step 3: run *create_custom_dataset.py* passing dataset.yaml as an argument.

---
References:

[1] Panayotov, Vassil, et al. "Librispeech: an asr corpus based on public domain audio books."
2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

[2] Fujita, Yusuke, et al. "End-to-end neural speaker diarization with self-attention."
2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2019.

[3]  Ko, Tom, et al. "A study on data augmentation of reverberant speech for robust speech recognition."
 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017.

[4] D. Dean, S. Sridharan, R. Vogt, M. Mason (2010) "The QUT-NOISE-TIMIT corpus for the evaluation of voice activity detection algorithms",
in Proceedings of Interspeech 2010, Makuhari Messe International Convention Complex, Makuhari, Japan


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
