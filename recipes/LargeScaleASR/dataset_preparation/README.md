# Preparing the data of The LargeScaleASR Set

This folder contains the necessary SpeechBrain recipe to create the LargeScaleASR
Set. In short:
- **generate.py** is the main SpeechBrain recipe file and must be called with *hparams/data_prep.yaml*. This is the file to call for each individual step.
- **hparams/data_prep.yaml** is the yaml configuration file containing all paths and parameters of the dataset creation. You must edit this file according to your system.
- **{dataset}_prepare.py** are preparation functionalities relative to each dataset. In practice, no need to touch these files unless ones want to do some customisation.
- **merge_csv_manifests.py** is an ensemble of utility functions to merge individual dataset manifests into the LargeScaleASR Set ones. In theory, no need to touch this file as it is called from *generate.py*.

## Step 1: Download datasets and prepare the parameters

**6-10TB of storage is needed to prepare The LargeScaleASR Set**

Each individual dataset including Libriheavy (and librilight large), voxpopuli, yodas (en000, en001 and en003), the people's speech (clean), commonvoice 18 and librispeech must be downloaded to disk. We do not provide any utility for this as they all are different and may vary dependign on your system. These datasets are, however, fairly easy to download.

Once downloaded, *hparams/data_prep.yaml* must be edited with the path to each downloaded (and extracted) dataset.

## Step 2: Prepare each individual dataset manifest.

Each individual dataset will need to be prepared. This include:
1. Text normalisation
2. Normalisation of the audio and copy of the wav/flac into the LargeScaleASR Set folder.
3. Generation of the csv.
This is done by simply calling:

```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=dataset_name```

Examples:

```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=commonvoice```
```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=libriheavy```

## Step 3: The LargeScaleASR Set manifests

Once each individual manifest has been generated for each dataset, they can be combined:

```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=set_name```

Examples:

```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=large_set```
```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=medium_set```
```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=val_test_sets```

Now, you are left with a typical SpeechBrain csv based dataset, you can already use it as is.
However, we recommend continuing with the HuggingFace interface.

First, you will need to put a README.md file with the following metadata:

```
---
dataset_info:
configs:
- config_name: large
  features:
  - name: ID
    dtype: string
  - name: duration
    dtype: float32
  - name: wav
    dtype: string
  - name: spk_id
    dtype: string
  - name: sex
    dtype: string
  - name: text
    dtype: string
  default: true
  data_files:
  - split: train
    path:
    - "manifests/largescaleasr_large_train.csv"
  - split: dev
    path:
    - "manifests/largescaleasr_dev.csv"
  - split: test
    path:
    - "manifests/largescaleasr_test.csv"
- config_name: medium
  features:
  - name: ID
    dtype: string
  - name: duration
    dtype: float32
  - name: wav
    dtype: string
  - name: spk_id
    dtype: string
  - name: sex
    dtype: string
  - name: text
    dtype: string
  data_files:
  - split: train
    path:
    - "manifests/largescaleasr_medium_train.csv"
  - split: dev
    path:
    - "manifests/largescaleasr_dev.csv"
  - split: test
    path:
    - "manifests/largescaleasr_test.csv"
- config_name: small
  features:
  - name: ID
    dtype: string
  - name: duration
    dtype: float32
  - name: wav
    dtype: string
  - name: spk_id
    dtype: string
  - name: sex
    dtype: string
  - name: text
    dtype: string
  data_files:
  - split: train
    path:
    - "manifests/largescaleasr_small_train.csv"
  - split: dev
    path:
    - "manifests/largescaleasr_dev.csv"
  - split: test
    path:
    - "manifests/largescaleasr_test.csv"
- config_name: clean
  features:
  - name: ID
    dtype: string
  - name: duration
    dtype: float32
  - name: wav
    dtype: string
  - name: spk_id
    dtype: string
  - name: sex
    dtype: string
  - name: text
    dtype: string
  data_files:
  - split: train
    path:
    - "manifests/largescaleasr_clean_train.csv"
  - split: dev
    path:
    - "manifests/largescaleasr_dev.csv"
  - split: test
    path:
    - "manifests/largescaleasr_test.csv"
---
```

After this stage, the dataset is already ready to be loaded by HuggingFace. For instance:

```python
from datasets import load_dataset
ds = load_dataset('path_to_folder_with_the_above_readme', 'medium')
print(ds['train'])
```
But another step is necessary if one wants to
shard the dataset instead of relying on individual files.

## Step 4: Sharding with .parquet files and HuggingFace

Audio files and supervision labels can be embedded into bigger shards with:

```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=export_to_parquet```

Each set "**large, medium, small, clean, val, test** must be exported individually. This is done by overriding the yaml variable, such as:

```python generate.py hparams/data_prep.yaml --ACTION_TO_PERFORM=export_to_parquet --PARQUET_SUBSET=large --PARQUET_SPLIT=train  --MAX_SHARD_SIZE=500MB --PARQUET_OUTPUT_FOLDER=/path/ThelargescaleasrSet_sharded/large --PARQUET_ORIG_CSV=/ThelargescaleasrSet/manifests/thelargescaleasrset_small_train.csv --HF_DATASET_ROOT=/ThelargescaleasrSet```

This step will also copy the original csv to the parquet folder. wav path are simplified to only the filename to prevent any security issue. This is necessary for two reasons:
1. The user may want to inspect the dataset, and a csv is much easier to view than a parquet file.
2. The tokenizer may need to access all the text to be trained, and this is also easier from a csv file.
