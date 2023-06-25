# Noise robust speech recognition on **RescueSpeech** (joint training approach)
This recipe outlines a method for developing a noise robust speech recognition system. The system consists of two components: a speech enhancement model and an Automatic Speech Recognition (ASR) model. The models are trained jointly and then combined to tackle noise interference. The speech enhancement model, based on the SepFormer architecture, enhances input speech by reducing background noise. The ASR system utilizes the WavLM and Whisper models.

During training, both speech enhancement and ASR is kept unfrozen, allowing them to update their weight during training. By doing so, the model can learn from the ASR system's feedback and further refine its noise reduction capabilities based on the specific requirements of speech recognition.


- Link to dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8077622.svg)](https://doi.org/10.5281/zenodo.8077622) (`Task_ASR.tar.gz`)
- Language: German (DE)

## How to run
```
python train.py hparams/robust_asr_wav2vec2_16k.yaml
```

## Results
Since enhancement model is frozen, both WavLM and Whisper share the same results for speech enhancement.

| Model | SISNRi | SDRi | PESQ   | STOI  | *WER*   |
|------ |--------|-------|-------|-------|----   |
| WavLM | 7.140  | 7.694 | 2.064 | 0.854 | 46.04 |
| Whisper| 7.334 | 7.871 | 2.085 | 0.857 | **24.20** |


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
