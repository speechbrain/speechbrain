# LargeScale ASR with CTC + Attention Transformers

This repository provides the scripts needed to run an Automatic Speech Recognition (ASR) experiment using the [LargeScaleASR dataset](https://huggingface.co/datasets/speechbrain/LargeScaleASR). The experiment leverages a combination of Connectionist Temporal Classification (CTC) and attention-based transformer models.

## Table of Contents

- [How to Run](#how-to-run)
- [Results](#results)
- [LargeScaleASR Dataset](#largescaleasr-dataset)
  - [Description](#description)
  - [Downloading the Data](#downloading-the-data)
  - [Obtaining the CSV Files for Tokenization](#obtaining-the-csv-files-for-tokenization)
- [About SpeechBrain](#about-speechbrain)
- [Citing SpeechBrain](#citing-speechbrain)

## How to Run

Execute the training script using the command below. Be sure to replace the placeholder values (e.g., `[number_of_gpus]`, `{hparam_file}`, and file paths) with your specific configuration:

```bash
torchrun --nproc_per_node=[number_of_gpus] train.py hparams/{hparam_file}.py \
  --hf_caching_dir=/path/to/hf/cache/dir \
  --tls_subset=[large|medium|small] \
  --train_csv=/path/to/downloaded/train.csv
```

**Note:**
The `hf_caching_dir` typically corresponds to the `$HF_HUB_CACHE` environment variable. If this variable isn’t set, locate your HuggingFace cache directory (commonly in the `.cache` folder) and provide its path.

## Results

Below is a summary of experimental results:

| Hyperparameters File    | # Parameters | Split             | Validation WER | Test WER | GPUs          | HuggingFace Link                                                             |
| ----------------------- | ------------ | ----------------- | -------------- | -------- | ------------- | ---------------------------------------------------------------------------- |
| `conformer_base.yaml`   | 100M         | Small (250h)      | 22.3           | 22.7     | 4xV100 32GB   | N/A                                                                          |
| `conformer_large.yaml`  | 250M         | Medium (2,500h)   | 10.7           | 11.9     | 4xV100 32GB   | N/A                                                                          |
| `conformer_large.yaml`  | 250M         | Large (25,000h)   | 7.9            | 8.8      | 8xV100 32GB   | N/A                                                                          |
| `conformer_xlarge.yaml` | 480M         | Large (25,000h)   | 6.8            | 7.5      | 8xV100 32GB   | [Model](https://huggingface.co/speechbrain/asr-conformer-largescaleasr)        |

## LargeScaleASR Dataset

### Description

The LargeScaleASR dataset is a curated blend of five permissively licensed datasets. The table below summarizes its composition:

| Dataset                      | Amount Taken (large/medium/small/dev/test) | License   |
| ---------------------------- | ------------------------------------------ | --------- |
| VoxPopuli                    | 550 / 500 / 50 / 5 / 7                       | CC0       |
| LibriHeavy                   | 11,000 / 500 / 50 / 0 / 0                    | CC BY 4.0 |
| Librispeech (dev-/test-other) | 0 / 0 / 0 / 5 / 7                           | CC BY 4.0 |
| YODAS                        | 6,100 / 500 / 50 / 1.5 / 1.5                 | CC BY 3.0 |
| People's Speech              | 5,900 / 500 / 50 / 1.5 / 1.5                 | CC-BY 4.0 |
| CommonVoice 18.0             | 1,660 / 500 / 50 / 5 / 7                     | CC0       |

For the **development** and **test** splits, only data from the respective `dev` and `test` sets of each dataset are used (i.e., no data is extracted from the training set, except for YODAS). For YODAS, data is drawn from the `en003` split and manually verified for audio and transcription quality to form the `dev`/`test` partitions.

### Downloading the Data

**Note:** Downloading and extracting the large subset of the LargeScaleASR dataset requires approximately **4 TB** of storage.

As with other SpeechBrain projects, this recipe does not include a data download script. Please refer to the [HuggingFace webpage](https://huggingface.co/datasets/speechbrain/LargeScaleASR) for instructions on downloading the dataset using the `datasets` library or `huggingface-cli`.

### Obtaining the CSV Files for Tokenization

The tokenizer requires CSV files (available [in the repository](https://huggingface.co/datasets/speechbrain/LargeScaleASR/tree/main)) for training. You have two options:

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

Alternatively, download only the CSV files using `huggingface-cli`:

```bash
huggingface-cli download speechbrain/LargeScaleASR --include="*.csv" --repo-type dataset
```

## About SpeechBrain

- Website: [https://speechbrain.github.io/](https://speechbrain.github.io/)
- Code: [https://github.com/speechbrain/speechbrain/](https://github.com/speechbrain/speechbrain/)
- HuggingFace: [https://huggingface.co/speechbrain/](https://huggingface.co/speechbrain/)

## Citing SpeechBrain

If you use SpeechBrain for your research or business, please cite it:

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```