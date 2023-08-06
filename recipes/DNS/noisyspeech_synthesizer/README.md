# **DNS: Noisy speech synthesizer**
This folder contains scripts to synthesize noisy audio for training.
Scripts have been taken from [official GitHub repo](https://github.com/microsoft/DNS-Challenge).


## **Usage**
1. Modify parameters like `sampling_rate`, `audio_length` , `total_hours` etc in the YAML file as per your requirement.
2. To create noisy dataset, run
```
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name read_speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name german_speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name french_speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name italian_speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name russian_speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name spanish_speech
```
It's recommended to execute these commands in parallel for quicker synthesis.

**Time** : It takes about 140 HRS to synthesize a dataset of 500 HRS. This calls the need for dynamic mixing.
