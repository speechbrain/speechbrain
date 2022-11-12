# Speech separation with WSJ0-Mix
This folder contains some popular recipes for the WSJ0-Mix task (2/3 sources).

* This recipe supports train with several source separation models on WSJ0-2Mix, including [Sepformer](https://arxiv.org/abs/2010.13154), [RE-SepFormer](https://arxiv.org/abs/2206.09507), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

**Web Demo** Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo Speech Seperation: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/speechbrain-speech-seperation)

Additional dependency:
```
pip install mir_eval
```

To run it:

```
python train.py hyperparams/sepformer.yaml --data_folder yourpath/wsj0-mix/2speakers
```
Note that during training we print the negative SI-SNR (as we treat this value as the loss).


# WSJ0-2mix and WSJ0-3mix dataset creation
* The best way to create the datasets is using the original matlab script. This script and the associated meta data can be obtained through the following [link](https://www.dropbox.com/s/gg524noqvfm1t7e/create_mixtures_wsj023mix.zip?dl=1).
* The dataset creation script assumes that the original WSJ0 files in the sphere format are already converted to .wav .


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training. For this you need to have the WSJ0 dataset (available though LDC at `https://catalog.ldc.upenn.edu/LDC93S6A`).


# Results

* You can listen to example results on the test set of WSJ0-2/3Mix with SepFormer through this [page](https://sourceseparationresearch.com/static/sepformer_example_results/sepformer_results.html).

* Here are the SI - SNRi results (in dB) on the test set of WSJ0-2/3 Mix with SepFormer:

| | SepFormer, WSJ0-2Mix |
|--- | --- |
|NoAugment | 20.4 |
|DynamicMixing | 22.4 |

| | SepFormer, WSJ0-3Mix |
|--- | --- |
|NoAugment | 17.6 |
|DynamicMixing | 19.8 |

| | RE-SepFormer, WSJ0-2Mix |
| --- | --- |
|DynamicMixing | 18.6 |

| | SkiM, WSJ0-2Mix |
| --- | --- |
|DynamicMixing | 18.1 |


# Training Time
Each epoch takes about 2 hours for WSJ0-2Mix and WSJ0-3Mix (DynamicMixing ) on a NVIDIA V100 (32GB).

# Pretrained Models:
Pretrained models for SepFormer on WSJ0-2Mix, WSJ0-3Mix, and WHAM! datasets can be found through huggingface:
* https://huggingface.co/speechbrain/sepformer-wsj02mix
* https://huggingface.co/speechbrain/sepformer-wsj03mix
* https://huggingface.co/speechbrain/resepformer-wsj02mix

* The output folder (with logs and checkpoints) for SepFormer (hparams/sepformer.yaml) can be found [here](https://drive.google.com/drive/folders/11ulM8NqLYle6vNNZb3NvPRPHR5Rrl-FF?usp=sharing).
* The output folder (with logs and checkpoints) for RE-SepFormer (hparams/resepformer.yaml) can be found [here](https://drive.google.com/drive/folders/1rXOyPQ7OZZMUzg7wrP1Zsa_fjFKMqaeu?usp=sharing).
* The output folder (with logs and checkpoints) for convtasnet (hparams/convtasnet.yaml) can be found [here](https://drive.google.com/drive/folders/12_Df4zsRW18YvD4hPAJAT9y_mVWnNyBm?usp=sharing).
* The output folder (with logs and checkpoints) for dual-path RNN (hparams/dprnn.yaml) can be found [here](https://drive.google.com/drive/folders/1Olq2077mXKqtqHluxECn1lMKIbo7xPFu?usp=sharing).
* The output folder (with logs and checkpoints) for SkiM (hparams/skim.yaml) can be found [here](https://drive.google.com/drive/folders/12HqVPpMXY-OOMsZ3xTAtkN7kk5TZ2YaL?usp=sharing).
* The output folder (with logs and checkpoints) for Sepformer with conformer block as intra model (hparams/sepformer-conformerintra.yaml) can be found [here](https://drive.google.com/drive/folders/1NcB7pKj7qWzDaI3ScDOyQwJLvRdW9rfl).




# Example calls for running the training scripts


* WSJ0-2Mix training without dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/2speakers`

* WSJ0-2Mix training with dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/2speakers --base_folder_dm yourpath/wsj0/si_tr_s --dynamic_mixing True`

* WSJ0-3Mix training without dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/3speakers`--num_spks 3

* WSJ0-3Mix training with dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/3speakers`--num_spks 3 --base_folder_dm yourpath/wsj0/si_tr_s --dynamic_mixing True`


# Multi-GPU training

You can run the following command to train the model using Distributed Data Parallel (DDP) with 2 GPUs:

```
 python -m torch.distributed.launch --nproc_per_node=2 train.py hparams/sepformer.yaml --data_folder /yourdatapath --distributed_launch --distributed_backend='nccl'
```
You can add the other runtime options as appropriate. For more complete information on multi-GPU usage, take a look at this [tutorial](https://colab.research.google.com/drive/13pBUacPiotw1IvyffvGZ-HrtBr9T6l15?usp=sharing).




# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
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
