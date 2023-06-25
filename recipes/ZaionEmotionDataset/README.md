# Speech Emotion Diarization (SED)

Speech Emotion Diarization ([arXiv link](to be added)) aims to predict the correct emotions and their temporal boundaries with in an utterance.

## Dependencies

First, please install the extra dependencies, do  `pip install -r requirements.txt`


## Datasets

### Test Set
The test set is Zaion Emotion Database (ZED), which can be downloaded [here](https://zaion.ai/en/resources/zaion-lab-blog/zaion-emotion-dataset/).

### Training Set
1. [RAVDESS](https://zenodo.org/record/1188976)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/s2cvu9bc7e13z1u/RAVDESS.zip`

   <!-- Unzip and rename the folder as "RAVDESS". -->

2. [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/o9fitanahhiq4bi/ESD.zip`

   <!-- Unzip and rename the folder as "ESD". -->

3. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)


4. [JL-CORPUS](https://www.kaggle.com/datasets/tli725/jl-corpus?resource=download)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/fma9dashyivlskx/JL_corpus.zip`


5. [EmoV-DB](https://openslr.org/115/)

   A fast download can be done by `wget https://dl.dropboxusercontent.com/s/5fnqr26qmqrm99k/EmoV-DB.zip`, where only `Amused, Neutral, Angry` emotions are kept.


## Run the code

First download the train/test datasets and unzip them.

To run the code, do `python train_with_wav2vec.py hparams/train_with_wav2vec.yaml --zed_folder /path/to/ZED --emovdb_folder /path/to/EmoV-DB --esd_folder /path/to/ESD --iemocap_folder /path/to/IEMOCAP --jlcorpus_folder /path/to/JL_corpus --ravdess_folder /path/to/RAVDESS`.

The frame-wise classification result for each utterance can be found in `results/eder.txt`.



## Inference

The pretrained models and a easy-inference interface can be found on [HuggingFace](to be added).



## Citation

to be added