# MetricGAN+ Recipe for Enhancement

This recipe implements MetricGAN+ recipe for enhancement as described in the paper
[MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement](https://arxiv.org/abs/2104.03538)

Use the `download_vctk` function in `voicebank_prepare.py` to download the dataset
and resample it to 16000 Hz. To run an experiment, execute the following command in
the current folder:

```bash
python train.py hparams/train.yaml --data_folder /path/to/data_folder
```

## Results

Experiment Date | PESQ | CSIG | CBAK | COVL
-|-|-|-|-
2021-03-06 | 3.15 | 4.14 | 3.16 | 3.64


# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/metricgan-plus-voicebank

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://drive.google.com/drive/folders/1IV3ohFracK0zLH-ZGb3LTas-l3ZDFDPW?usp=sharing



## Citation

If you find the code useful in your research, please cite:

	@article{fu2021metricgan+,
	  title={MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement},
      author={Fu, Szu-Wei and Yu, Cheng and Hsieh, Tsun-An and Plantinga, Peter and Ravanelli, Mirco and Lu, Xugang and Tsao, Yu},
      journal={arXiv preprint arXiv:2104.03538},
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
