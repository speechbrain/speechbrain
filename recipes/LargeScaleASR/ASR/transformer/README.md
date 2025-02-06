# The LargeScale ASR with CTC + Attention transformers


This folder contains scripts necessary to run an ASR experiment with the LargeScaleASR dataset: [HuggingFace webpage](https://huggingface.co/datasets/speechbrain/LargeScaleASR) and pytorch 2.0.

# Downloading the data

**We note that 4TB of storage are necessary to download and extract the large subset of the LargeScaleASR dataset.**

## Get the bulk of the data
Much like any SpeechBrain code, this recipe does not provide any script to download the data.
We invite you to refer to the [HuggingFace webpage](https://huggingface.co/datasets/speechbrain/LargeScaleASR) to first download the dataset. This can usually be done with the datasets library
in as little as 2 lines of python code in a prompt.

**We very much encourage you to set the HF_HUB_CACHE global variable before downloading the dataset. This will change where HF puts the data (by default it's in some hidden .cache place).**

## Get the csv files for tokenisation

It is also important to download the .csv files [from the repository](https://huggingface.co/datasets/speechbrain/LargeScaleASR/tree/main). This is because the tokeniser needs them to be trained.

To achieve this, two possibilities exist:
```shell
git clone https://huggingface.co/datasets/speechbrain/LargeScaleASR
```
or for a more finegrained choice ... (you only need the csv of the subset that you are interested in):
```shell
wget https://huggingface.co/datasets/speechbrain/LargeScaleASR/resolve/main/largescaleasr_large_train.csv?download=true
wget https://huggingface.co/datasets/speechbrain/LargeScaleASR/resolve/main/largescaleasr_medium_train.csv?download=true
wget https://huggingface.co/datasets/speechbrain/LargeScaleASR/resolve/main/largescaleasr_small_train.csv?download=true
```

# How to run

```shell
torchrun --nproc_per_node=[nb_of_gpu] train.py hparams/{hparam_file}.py --data_folder=/path/to/the/HF/downloaded_folder --hf_caching_dir=/path/to/hf/cache/dir --tls_subset=[large||medium||small] --train_csv=/path/to/the/downloaded/train.csv
```

We note that *hf_caching_dir* usually is equal to $HF_HUB_CACHE. If this variable is not set in your
environment then you will need to find out where HF stored it! (Have a look around .cache). We unfortunately don't have a cleaner way to manage this as it is handled by HuggingFace.

# Results

| hyperparams file | #params  | validation WER | test WER |  GPUs | HuggingFace Link |
|:-------------:|:-------------:|:-------------:| :-----:| :-----:| :-----:|
| conformer_large.yaml | 250M | 7.9 | 8.8 | 8xV100 32GB | N/A |
| conformer_xlarge.yaml | 480M | 6.8 | 7.5 | 8xV100 32GB | [model](https://huggingface.co/speechbrain/asr-conformer-largescaleasr) |



