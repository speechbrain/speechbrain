# Wav2vec2-FR for RECOLA

## Installation


```bash
conda create --prefix [conda_path]/recola_sb python=3.8 # create the environment
conda activate [conda_path]/recola_sb # activate the environment
conda install -c anaconda pandas==1.4.2 # used to store predictions of a trained model
conda install -c conda-forge ffmpeg # preprocessing is done by ffmpeg
pip install speechbrain==0.5.13 # used as the main experimental framework
pip install transformers==4.21.3 # used for loading wav2vec2 models
```

# Experiments and Results

|        Feature        |                             Run                              | Dev. Arousal | Test Arousal | Dev. Valence | Test Valence |
| :-------------------: | :----------------------------------------------------------: | ------------ | ------------ | -----------: | -----------: |
|          MFB          | python train.py settings_MFB.yaml --emotion_dimension=[arousal/valence] --experiment_folder="./Results" --data_path=[RECOLA_2016 path] | .520         | .615         |         .373 |         .425 |
|  wav2vec2-FR-1K-base  | python train.py settings.yaml --emotion_dimension=[arousal/valence] --feat_size=768 --w2v2_hub=LeBenchmark/wav2vec2-FR-1K-base --experiment_folder="./Results" --data_path=[RECOLA_2016 path] | .584         | .431         |         .121 |         .222 |
| wav2vec2-FR-2.6K-base | python train.py settings.yaml --emotion_dimension=[arousal/valence] --feat_size=768 --w2v2_hub=LeBenchmark/wav2vec2-FR-2.6K-base --experiment_folder="./Results" --data_path=[RECOLA_2016 path] | **.719**     | **.682**     |     **.498** |     **.463** |
|  wav2vec2-FR-3K-base  | python train.py settings.yaml --emotion_dimension=[arousal/valence] --feat_size=768 --w2v2_hub=LeBenchmark/wav2vec2-FR-3K-base --experiment_folder="./Results" --data_path=[RECOLA_2016 path] | .188         | .237         |         .001 |         .099 |
|  wav2vec2-FR-7K-base  | python train.py settings.yaml --emotion_dimension=[arousal/valence] --feat_size=768 --w2v2_hub=LeBenchmark/wav2vec2-FR-7K-base --experiment_folder="./Results" --data_path=[RECOLA_2016 path] | .515         | .454         |         .096 |         .015 |

Note: if you want to specify the device, it is possible to add it as an argument, e.g. --device="cpu". Also, it is possible to overwrite any of the parameters in the settings.yaml file by passing it as an argument.