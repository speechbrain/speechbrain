# **DNS: Noisy speech synthesizer**
This folder contains scripts to synthesize noisy audio for training.
Scripts have been taken from [official GitHub repo](https://github.com/microsoft/DNS-Challenge).

Modify parameters like `sampling_rate`, `audio_length` , `total_hours` etc in the YAML file as per your requirement.

## Option: 1- Synthesize clean-noisy data and create the Webdataset shards
Synthesize clean-noisy data and create WebDataset shards.
### **Usage**
To create noisy dataset, run
```
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name read_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data_shards
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name german_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data_shards
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name french_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data_shards
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name italian_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data_shards
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name russian_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data_shards
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name spanish_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data_shards
```

## Option: 2- Synthesize clean-noisy data and store them as wav files
Synthesize clean-noisy data and save them as wavs.

### **Usage**
To create noisy dataset, run
```
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name read_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data --sharding False
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name german_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data --sharding False
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name french_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data --sharding False
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name italian_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data --sharding False
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name russian_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data --sharding False
python noisyspeech_synthesizer_singleprocess.py noisyspeech_synthesizer.yaml --uncompressed_path ../DNS-dataset/datasets_fullband/ --split_name spanish_speech --input_shards_dir ../DNS-shards --synthesized_data_dir synthesized_data --sharding False
```

It's recommended to execute these commands in parallel for quicker synthesis.

**Time** : It takes about 140 HRS to synthesize a dataset of 500 HRS. This calls the need for dynamic mixing.
