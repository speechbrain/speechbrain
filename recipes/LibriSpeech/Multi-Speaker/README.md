# Multi-Speaker
Here we create overlapping speaker data from LibriSpeech single speaker samples by randomly sampling n single speaker audio tracks from LibriSpeech. We then adapt the train and librispeech preparation scripts to be able to model the maximum number of concurrent speakers in a given sample using both Xvector and ECAPA TDNN modeling architectures.
## 1. Overlapping LibriSpeech Data Creation

Here, we combine a variable number of 5s unique speaker LibriSpeech samples and create overlapping speech samples to be used in speaker count applications. 

### Usage

Before running you need to have downloaded dev-clean, test-clean and train-clean-100 datasets from the openSLR library. (You can modify the list of sample sets, but these will work out the box.)

Running Data Creation: 
```bash
python data_creation.py hparams/data_params.yaml --original_data_folder <link-to-folder> --new_data_folder <link-to-destination-folder>
```
## 2. Maximum Concurrent Speaker Modeling

We have added support for XVector and ECAPA TDNN model architectures for modelling the max number of simultaneous speakers in a given sample. 

### Usage

XVector Modeling of Speaker Count: 
```bash
python train.py hparams/train_xvector_params.yaml --data_folder <link-to-created-data-folder>
```

ECAPA TDNN Modeling of Speaker Count: 
```bash
python train.py hparams/train_ecapa_tdnn.yaml --data_folder <link-to-created-data-folder>
```