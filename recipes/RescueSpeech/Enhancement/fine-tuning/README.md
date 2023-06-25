# Speech enhancement on **RescueSpeech**
This folder contains speech enhancement recipes using [SepFormer](https://arxiv.org/abs/2010.13154) for the [RescueSpeech dataset](https://zenodo.org/record/8077622) (`Task_enhancement.tar.gz`)

Additional dependency:
```
pip install pesq librosa pystoi mir_eval
```

We use SepFormer speech enhancement model first trained on 1300 HRS of Deep Noise Suppression (DNS)Challenge 4 dataset and further fine-tune on **~7h** of RescueSpeech noisy dataset.

## How to run?
```
python train.py hparams/sepformer_16k.yaml
```
## Results
Results on RescueSpeech test set are as follows:

| Metric | Before fine-tuning | After fine-tuning |
| ------ | ------------------ | ----------------- |
| SISNRi | 4.621              | 7.849             |
| SDRi   | 6.046              | 8.414             |
| PESQ   | 1.940              | 2.244             |
| STOI   | 0.764              | 0.812             |
| CSIG   | 2.852              | 3.282             |
| CBAK   | 1.993              | 2.169             |
| COVL   | 2.358              | 2.750             |

- The output folder with the model checkpoints and logs for WHAMR! is available *add full model drive link*.
- It takes around 30 mins per epoch to train on RTXA6000 48 GB GPU.

## Pretrained Models
Pretrained models for SepFormer on DNS-4 dataset can be found through:
- HuggingFace : *add hf link to sepformer_dns_16k*
- Full Model Link: *add drive link to sepformer_dns_16k*


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


**Citing SpeechBrain**
- Please, cite SpeechBrain if you use it for your research or business.

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

**Citing RescueSpeech**
- Dataset
```bibtex
@misc{sagar_sangeet_2023_8077622,
  author       = {Sagar, Sangeet and
                  Kiefer, Bernd and
                  Kruijff Korbayova, Ivana},
  title        = {{RescueSpeech: A German Corpus for Speech
                   Recognition in Search and Rescue Domain}},
  month        = jun,
  year         = 2023,
  note         = {{Our work was supported under the project "A-DRZ:
                   Setting up the German Rescue Robotics Center" and
                   funded by the German Ministry of Education and
                   Research (BMBF), grant No. I3N14856.}},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8077622},
  url          = {https://doi.org/10.5281/zenodo.8077622}
}
```
- Paper
```bibtex
@misc{sagar2023rescuespeech,
    title={RescueSpeech: A German Corpus for Speech Recognition in Search and Rescue Domain},
    author={Sangeet Sagar and Mirco Ravanelli and Bernd Kiefer and Ivana Kruijff Korbayova and Josef van Genabith},
    year={2023},
    eprint={2306.04054},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

**Citing SepFormer**
```bibtex
@inproceedings{subakan2021attention,
      title={Attention is All You Need in Speech Separation},
      author={Cem Subakan and Mirco Ravanelli and Samuele Cornell and Mirko Bronzi and Jianyuan Zhong},
      year={2021},
      booktitle={ICASSP 2021}
}
```
