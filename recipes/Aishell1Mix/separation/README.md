# Speech separation with Aishell1Mix
This folder contains some popular recipes for the Aishell1Mix dataset similar to [LibriMix Dataset](https://arxiv.org/pdf/2005.11262.pdf) (2/3 sources).

* This recipe supports train with several source separation models on Aishell1Mix, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

Additional dependencies:
```
pip install mir_eval
pip install pyloudnorm
```

To run it:

```
python train.py hparams/sepformer-aishell1mix2.yaml --data_folder youraishell1mixpath/aishell1mix/Aishell1Mix2 --base_folder_dm /youraishell1path/aishell1/data_aishell/wav/train
python train.py hparams/sepformer-aishell1mix3.yaml --data_folder youraishell1mixpath/aishell1mix/Aishell1Mix3 --base_folder_dm /youraishell1path/aishell1/data_aishell/wav/train

```
Note that during training we print the negative SI-SNR (as we treat this value as the loss).


# Aishell1Mix2/3
* The Dataset can be created using the scripts at `https://github.com/huangzj421/Aishell1Mix`.


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training.

# Results

Here are the SI - SNRi results (in dB) on the test set of Aishell1Mix dataset with SepFormer:

| | SepFormer. Aishell1Mix2 |
| --- | --- |
|SpeedAugment | |
|DynamicMixing | 13.4|


| | SepFormer. Aishell1Mix3 |
| --- | --- |
|SpeedAugment | |
|DynamicMixing | 11.2|

# Pretrained Models:
Pretrained models for SepFormer on Aishell1Mix2, Aishell1Mix3, and WHAM! datasets can be found through huggingface:
* https://huggingface.co/speechbrain/sepformer-aishell1mix2
* https://huggingface.co/speechbrain/sepformer-aishell1mix3


You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/sepformer-wsj02mix).

# Example calls for running the training scripts

* Aishell1Mix2 with dynamic mixing `python train.py hparams/sepformer-aishell1mix2.yaml --data_folder youraishell1mixpath/aishell1mix/Aishell1Mix2 --base_folder_dm /youraishell1path/aishell1/data_aishell/wav/train --dynamic_mixing True`

* Aishell1Mix3 with dynamic mixing `python train.py hparams/sepformer-aishell1mix3.yaml --data_folder youraishell1mixpath/aishell1mix/Aishell1Mix3 --base_folder_dm /youraishell1path/aishell1/data_aishell/wav/train --dynamic_mixing True`

* Aishell1Mix2 with dynamic mixing with WHAM! noise in the mixtures `python train.py hparams/sepformer-aishell1mix2.yaml --data_folder youraishell1mixpath/aishell1mix/Aishell1Mix2 --base_folder_dm /youraishell1path/aishell1/data_aishell/wav/train --dynamic_mixing True --use_wham_noise True`

* Aishell1Mix3 with dynamic mixing with WHAM! noise in the mixtures `python train.py hparams/sepformer-aishell1mix3.yaml --data_folder youraishell1mixpath/aishell1mix/Aishell1Mix3 --base_folder_dm /youraishell1path/aishell1/data_aishell/wav/train --dynamic_mixing True --use_wham_noise True`

# Multi-GPU training

You can run the following command to train the model using Distributed Data Parallel (DDP) with 2 GPUs:

```
 python -m torch.distributed.launch --nproc_per_node=2 train.py hparams/sepformer.yaml --data_folder /yourdatapath --distributed_launch --distributed_backend='nccl'
```
You can add the other runtime options as appropriate. For more complete information on multi-GPU usage, take a look at this [tutorial](https://colab.research.google.com/drive/13pBUacPiotw1IvyffvGZ-HrtBr9T6l15?usp=sharing).


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


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
