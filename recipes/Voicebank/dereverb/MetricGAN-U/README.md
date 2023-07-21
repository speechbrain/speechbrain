# MetricGAN-U Recipe for dereverberation

This recipe implements MetricGAN-U recipe for dereverberation as described in the paper
[MetricGAN-U: Unsupervised speech enhancement/ dereverberation based only on noisy/ reverberated speech](https://arxiv.org/abs/2110.05866)

Notes:
1- By default we use srmr as a default target metric. This requires you to install SRMRpy (see extra-dependecies.txt)
2- To use dnsmos as a target metric, you have to ask the key from the DNS organizer first: dns_challenge@microsoft.com

# Dataset
Please "Manually" Download VoiceBank-SLR dataset from [here](https://bio-asplab.citi.sinica.edu.tw/Opensource.html#VB-SLR):

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# How to run
To run an experiment, execute the following command in
the current folder:

```bash
python train.py hparams/train_dereverb.yaml --data_folder /path/to/data_folder
```

## Results
Experiment Date | Hyperparams file | PESQ | SRMR |
-|-|-|-|
2021-10-31 | train_dereverb.yaml | 2.07 | 8.265 |

You can find the full experiment folder (i.e., checkpoints, logs, etc) [here](https://www.dropbox.com/sh/r94qn1f5lq9r3p7/AAAZfisBhhkS8cwpzy1O5ADUa?dl=0).


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
