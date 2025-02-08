# LargeScale ASR with CTC + Attention Transformers

This repository contains the necessary scripts to run an Automatic Speech Recognition (ASR) experiment using the [LargeScaleASR dataset](https://huggingface.co/datasets/speechbrain/LargeScaleASR). The experiment combines both Connectionist Temporal Classification (CTC) and attention-based transformer models.

## Dataset Description

The LargeScaleASR dataset is a curated mix of five permissively licensed datasets. The following table summarizes the composition:

| Dataset                         | Amount Taken (large/medium/small/dev/test) | License   |
| ------------------------------- | ------------------------------------------ | --------- |
| VoxPopuli                       | 550 / 500 / 50 / 5 / 7                       | CC0       |
| LibriHeavy                      | 11,000 / 500 / 50 / 0 / 0                    | CC BY 4.0 |
| Librispeech (dev-/test-other)    | 0 / 0 / 0 / 5 / 7                           | CC BY 4.0 |
| YODAS                           | 6,100 / 500 / 50 / 1.5 / 1.5                 | CC BY 3.0 |
| People's Speech                 | 5,900 / 500 / 50 / 1.5 / 1.5                 | CC-BY 4.0 |
| CommonVoice 18.0                | 1,660 / 500 / 50 / 5 / 7                     | CC0       |

For the **development** and **test** splits, only data from the corresponding `dev` and `test` sets of each dataset are used (i.e., data is not extracted from the training set, except for YODAS). For YODAS, data is extracted from the `en003` split and manually verified for audio/transcription quality to form the `dev`/`test` partitions.

## Downloading the Data

**Note:** Downloading and extracting the large subset of the LargeScaleASR dataset requires approximately **4 TB** of storage.

As with other SpeechBrain projects, this recipe does not include a script to download the data. Please refer to the [HuggingFace webpage](https://huggingface.co/datasets/speechbrain/LargeScaleASR) for instructions. You can typically download the dataset using the `datasets` library in just a couple of lines of Python.

Alternatively, you can use the `huggingface-cli` command-line tool to download the dataset:

```bash
huggingface-cli download speechbrain/LargeScaleASR --local-dir /path/to/local/dir --repo-type dataset
```

## Obtaining the CSV Files for Tokenization

The tokeniser requires CSV files (available [in the repository](https://huggingface.co/datasets/speechbrain/LargeScaleASR/tree/main)) for training. You have two options:

1. **Clone the entire repository:**

    ```bash
    git clone https://huggingface.co/datasets/speechbrain/LargeScaleASR
    ```

2. **Download specific CSV files:**

    ```bash
    wget https://huggingface.co/datasets/speechbrain/LargeScaleASR/resolve/main/largescaleasr_large_train.csv?download=true
    wget https://huggingface.co/datasets/speechbrain/LargeScaleASR/resolve/main/largescaleasr_medium_train.csv?download=true
    wget https://huggingface.co/datasets/speechbrain/LargeScaleASR/resolve/main/largescaleasr_small_train.csv?download=true
    ```

Alternatively, you can download only the CSV files using `huggingface-cli`:

```bash
huggingface-cli download speechbrain/LargeScaleASR --include="*.csv" --local-dir /path/to/local/dir --repo-type dataset
```

## How to Run

Execute the training script using the following command:

```bash
torchrun --nproc_per_node=[number_of_gpus] train.py hparams/{hparam_file}.py \
  --data_folder=/path/to/HF/downloaded_folder \
  --hf_caching_dir=/path/to/hf/cache/dir \
  --tls_subset=[large|medium|small] \
  --train_csv=/path/to/downloaded/train.csv
```

**Note:** The `hf_caching_dir` usually corresponds to the environment variable `$HF_HUB_CACHE`. If this variable is not set, you may need to locate your HuggingFace cache directory (commonly in the `.cache` folder).

## Results

Below is a summary of experimental results:

| Hyperparameters File    | # Parameters | Split             | Validation WER | Test WER | GPUs          | HuggingFace Link                                                             |
| ----------------------- | ------------ | ----------------- | -------------- | -------- | ------------- | ---------------------------------------------------------------------------- |
| `conformer_base.yaml`   | 100M         | Small (250h)      | 22.3           | 22.7     | 4xV100 32GB   | N/A                                                                          |
| `conformer_large.yaml`  | 250M         | Medium (2,500h)   | 10.7           | 11.9     | 4xV100 32GB   | N/A                                                                          |
| `conformer_large.yaml`  | 250M         | Large (25,000h)   | 7.9            | 8.8      | 8xV100 32GB   | N/A                                                                          |
| `conformer_xlarge.yaml` | 480M         | Large (25,000h)   | 6.8            | 7.5      | 8xV100 32GB   | [Model](https://huggingface.co/speechbrain/asr-conformer-largescaleasr) |
