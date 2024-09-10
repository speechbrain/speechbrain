# Spectral-Mask Recipe for dereverberation

This recipe implements the Spectral-Mask baseline for dereverberation as described in the paper
[MetricGAN-U: Unsupervised speech enhancement/ dereverberation based only on noisy/ reverberated speech](https://arxiv.org/abs/2110.05866)

Notes:
1- By default we use srmr as a default target metric. This requires you to install SRMRpy (see extra-dependencies.txt)
2- To use dnsmos as a target metric, you have to ask the key from the DNS organizer first: dns_challenge@microsoft.com

# Dataset
Please "Manually" Download VoiceBank-SLR dataset from [here](https://bio-asplab.citi.sinica.edu.tw/Opensource.html#VB-SLR):

# How to run
To run an experiment, execute the following command in
the current folder:

```bash
python train.py hparams/train_dereverb.yaml --data_folder /path/to/data_folder
```

## Results
Experiment Date | Hyperparams file | PESQ | STOI |
-|-|-|-|
2021-10-31 | train.yaml | 2.35 | 0.886 |

You can find the full experiment folder (i.e., checkpoints, logs, etc) [here](https://www.dropbox.com/sh/pw8aer8gcsrdbx7/AADknh7plHF5GBeTRK9VkIKga?dl=0).

## Citation
If you find the code useful in your research, please cite:

	@article{fu2021metricgan,
	  title={MetricGAN-U: Unsupervised speech enhancement/dereverberation based only on noisy/reverberated speech},
	  author={Fu, Szu-Wei and Yu, Cheng and Hung, Kuo-Hsuan and Ravanelli, Mirco and Tsao, Yu},
	  journal={arXiv preprint arXiv:2110.05866},
	  year={2021}
	}

    @inproceedings{fu2019metricGAN,
      title     = {MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement},
      author    = {Fu, Szu-Wei and Liao, Chien-Feng and Tsao, Yu and Lin, Shou-De},
      booktitle = {International Conference on Machine Learning (ICML)},
      year      = {2019}
    }


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

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
