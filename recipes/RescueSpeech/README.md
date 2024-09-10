# Noise robust speech recognition on RescueSpeech dataset
[RescuSpeech](https://doi.org/10.5281/zenodo.8077622) is a dataset specifically designed for performing noise robust speech recognition in the Search and Rescue domain. In this repository, we provide training recipes and pre-trained models for the best setup that have been developed and evaluated using RescuSpeech data. This aims to enhance the performance of speech recognizers in challenging and noisy environments.

Our [paper](https://arxiv.org/abs/2306.04054) compares ASR models (CRDNN, Wav2vec2, WavLM, Whisper) and speech-enhancement systems (SepFormer). This recipe contains the best-performing model, which is based on a simple combination of a speech enhancement model (**SepFormer**) and an ASR (**Whisper**) model. The models are trained jointly and then combined to tackle noise interference.

- Link to dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8077622.svg)](https://doi.org/10.5281/zenodo.8077622)
- Language: German (DE)


## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

## How to run
```
cd RescueSpeech/ASR/noise-robust
python train.py hparams/robust_asr_16k.yaml --data_folder=<data_folder_path>
```
Here the data path should be the path to **uncompressed `Task_ASR.tar.gz`** downloaded from link above.

# How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
cd RescueSpeech/ASR/noise-robust
python train.py hparams/robust_asr_16k.yaml --data_folder=<data_folder_path> --test_only
```
## Computing power
Please note that running this recipe can be computationally demanding due to the Whisper ASR (`whisper-large-v2`) model with 906.5M parameters (compared to 1.5B parameters in the original model but feature encoder is frozen in our case). When fine-tuning both the Whisper and SepFormer models together, we used an Nvidia A100-80 GB GPU, which took approximately 15 minutes per epoch.

## Results
During training, both speech enhancement and ASR is kept unfrozen- i.e. both ASR and enhance loss are backpropagated and weights are updated.

| Model | SISNRi | SDRi | PESQ   | STOI  | *WER*   |
|------ |--------|-------|-------|-------|----   |
| Whisper (`large-v2`)| 7.482 | 8.011 | 2.083 | 0.854 | **45.29** |


## Fine-tuned models
1. Firstly, the SepFormer enhancement model is trained on the Microsoft-DNS dataset. Subsequently, it undergoes fine-tuning with our RescueSpeech *enhancement* dataset (first row in the table below).
2. The Whisper ASR is fine-tuned on the RescueSpeech *ASR* dataset (second row in the table below).
3. Finally, the fine-tuned SepFormer and Whisper ASR models are jointly fine-tuned using our RescueSpeech *ASR* dataset. This represents the best model reported in the table above, with its pretrained models and logs accessible in the third row of the table below.

|S. No. |  Model        | HuggingFace link                               | Full Model link                                |
|---|----------------|------------------------------------------------|------------------------------------------------|
| 1. | Whisper ASR    | [HuggingFace](https://huggingface.co/speechbrain/whisper_rescuespeech)             | [Dropbox](https://www.dropbox.com/sh/dgmgi0b3bfxlfo4/AAAo3EYPXUEMZRTdRDzhw4lea?dl=)             |
| 2. | Sepformer Enhancement   | [HuggingFace](https://huggingface.co/speechbrain/sepformer_rescuespeech)            | [Dropbox](https://www.dropbox.com/sh/edrna82oarivkzl/AACsiGQXnbAYa_bfTJzjY23qa?dl=0)            |
| 3. | Sepformer +  Whisper ASR  (fine-tuned)  |  [HuggingFace](https://huggingface.co/sangeet2020/noisy-whisper-resucespeech)            | [Dropbox](https://www.dropbox.com/sh/kqs2ld14fm20cxl/AACiobSLdNtXhm-4Y3IIbTeia?dl=0)            |


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
