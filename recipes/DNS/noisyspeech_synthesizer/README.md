# **DNS: Noisy speech synthesizer**
This folder contains scripts to synthesize noisy audio for training.
Scripts have been taken from [official GitHub repo](https://github.com/microsoft/DNS-Challenge).


## **Usage**
1.  First download RIRs table from [Google Drive](https://drive.google.com/drive/folders/1P2QvpwsK-xxI2ahX0LQQDmc7azK5OPc_?usp=sharing)
or can also be downloaded from [official github repo](https://github.com/microsoft/DNS-Challenge/blob/0443a12f5e6e7bec310f453cf0d9637ca28e0eea/datasets/acoustic_params/RIR_table_simple.csv), but then have to be adapted.

2. Modify parameters like `sampling_rate`, `audio_length` etc in the YAML file as per your requirement.

3. To create noisy dataset, run
```
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --split_name read_speech

python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --split_name german_speech

python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --split_name french_speech

python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --split_name italian_speech

python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --split_name russian_speech

python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --split_name spanish_speech

```
It's recommended to execute these commands in parallel for quicker synthesis.

**Time** : It takes about 140 HRS to synthesize a dataset of 500 HRS. This calls the need for dynamic mixing.