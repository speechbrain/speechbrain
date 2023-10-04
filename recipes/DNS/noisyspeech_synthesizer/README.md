# **DNS: Noisy speech synthesizer**
This folder contains scripts to synthesize noisy audio for training.
Scripts have been taken from [official GitHub repo](https://github.com/microsoft/DNS-Challenge).

Modify parameters like `sampling_rate`, `audio_length` , `total_hours` etc in the YAML file as per your requirement.

## Synthesize clean-noisy data and create the Webdataset shards
Synthesize clean-noisy data and create WebDataset shards.

### **Usage**
To create noisy dataset, run
```
## synthesize read speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --input_shards_dir ../DNS-shards --split_name read_speech --synthesized_data_dir synthesized_data_shards

## synthesize German speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --input_shards_dir ../DNS-shards --split_name german_speech --synthesized_data_dir synthesized_data_shards

## synthesize Italian speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --input_shards_dir ../DNS-shards --split_name italian_speech --synthesized_data_dir synthesized_data_shards

## synthesize French speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --input_shards_dir ../DNS-shards --split_name french_speech --synthesized_data_dir synthesized_data_shards

## synthesize Spanish speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --input_shards_dir ../DNS-shards --split_name spanish_speech --synthesized_data_dir synthesized_data_shards

## synthesize Russian speech
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --input_shards_dir ../DNS-shards --split_name russian_speech --synthesized_data_dir synthesized_data_shards
```

It's recommended to execute these commands in parallel for quicker synthesis.

**Time** : It takes about 140 HRS to synthesize a dataset of 500 HRS. This calls the need for dynamic mixing.
