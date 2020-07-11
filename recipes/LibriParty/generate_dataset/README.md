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
**"official" metadata files** from [temp-onedrive-link]() and creating the dataset from this metadata.

#### from official metadata:
"Official" dataset uses LibriSpeech train-clean-100 for training, dev-clean for development and test-clean for test.
It also requires background TIMIT-QUT noises.  

- Step 1, **manual download**:
    - download LibriSpeech train-clean-100, dev-clean and test-clean from [https://openslr.org/12/](https://openslr.org/12/). 
    - download noises and impulse responses from [here](). 
    - download from [here]().
    - resample QUT-TIMIT noise sources to 16 kHz using *local/resample_folder.py*.

- Step 1, **use provided script** for downloading and resampling sources:
    - run *get_dataset_from_metadata.py*. 

- step 2: download "official" metadata from [temp-onedrive-link](). 
- step 3: change the paths accordingly in *dataset.yml*
- step 4: run *create_dataset_from_metadata.py*
 
#### custom


- step 4: change the paths accordingly in dataset.yml and optionally the dataset parameters. 
- step 5: run create_mixtures_metadata.py to create the dataset metadata. 
- step 6: run create_mixtures_from_metadata.py to create the actual audio mixtures from the metadata.

**NOTE**: Steps 4 to 6 only create data for one split (e.g. train), if one wants to have train, dev and test it must run the 
script multiple times with different sources. 


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