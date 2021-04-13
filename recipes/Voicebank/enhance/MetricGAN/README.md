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
