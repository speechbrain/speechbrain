# MetricGAN Recipe for Enhancement

This recipe implements MetricGAN recipe for enhancement as described in the paper
[MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement](https://arxiv.org/abs/1905.04874)

Use the `download_vctk` function in `voicebank_prepare.py` to download the dataset
and resample it to 16000 Hz. To run an experiment, execute the following command in
the current folder:

```bash
python train.py hparams/train.yaml --data_folder /path/to/data_folder
```

## Results

Experiment Date | PESQ | STOI
-|-|-
2021-03-06 | 3.08 | 93.0

## Citation

If you find the code useful in your research, please cite:

    @inproceedings{fu2019metricGAN,
      title     = {MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement},
      author    = {Fu, Szu-Wei and Liao, Chien-Feng and Tsao, Yu and Lin, Shou-De},
      booktitle = {International Conference on Machine Learning (ICML)},
      year      = {2019}
    }
