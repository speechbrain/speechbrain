# Noise robust speech recognition on RescueSpeech dataset
[RescuSpeech](https://doi.org/10.5281/zenodo.8077622) is a dataset specifically designed for performing noise robust speech recognition in the Search and Rescue domain. In this repository, we provide training recipes and pre-trained models for the best setup that have been developed and evaluated using RescuSpeech data. This aim to enhance the performance of speech recognizers in challenging and noisy environments.

Our paper compares ASR models (CRDNN, Wav2vec2, WavLM, Whisper) and speech-enhancement systems (SepFormer). But here we present our best-performing models achieved using the best strategy. See below for pre-trained model details and the full model link.

This recipe supports a simple combination of a speech enhancement model (**SepFormer**) and an ASR (**Whisper**) model. The models are trained jointly and then combined to tackle noise interference.

- Link to dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8077622.svg)](https://doi.org/10.5281/zenodo.8077622)
- Language: German (DE)


## How to run
```
python train.py hparams/robust_asr_16k.yaml
```

## Results
During training, both speech enhancement and ASR is kept unfrozen.

| Model | SISNRi | SDRi | PESQ   | STOI  | *WER*   |
|------ |--------|-------|-------|-------|----   |
| Whisper (`large-v2`)| 7.334 | 7.871 | 2.085 | 0.857 | **24.20** |

## Pretrained Models
We initially perform fine-tuning of both the ASR model and SepFormer model using the CommonVoice dataset and the Microsoft-DNS dataset. Subsequently, we proceed with a second stage of fine-tuning on our RescueSpeech dataset. Here you can find links to the trained models.


| Dataset        | CRDNN                                          | Wav2vec2                                       | wavLM                                          | Whisper                                        |
|----------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| German <br> CommonVoice10.0    | [HuggingFace](link_commonvoice_crdnn_hf)        | [HuggingFace](link_commonvoice_wav2vec2_hf)    | [HuggingFace](link_commonvoice_wavlm_hf)        | [HuggingFace](link_commonvoice_whisper_hf)      |
|                | [Google Drive](link_commonvoice_crdnn_gd)       | [Google Drive](link_commonvoice_wav2vec2_gd)   | [Google Drive](link_commonvoice_wavlm_gd)       | [Google Drive](link_commonvoice_whisper_gd)     |
| RescueSpeech   | [HuggingFace](link_rescuespeech_crdnn_hf)       | [HuggingFace](link_rescuespeech_wav2vec2_hf)   | [HuggingFace](link_rescuespeech_wavlm_hf)       | [HuggingFace](link_rescuespeech_whisper_hf)     |
|                | [Google Drive](link_rescuespeech_crdnn_gd)      | [Google Drive](link_rescuespeech_wav2vec2_gd)  | [Google Drive](link_rescuespeech_wavlm_gd)      | [Google Drive](link_rescuespeech_whisper_gd)    |



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
