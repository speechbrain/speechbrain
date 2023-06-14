# **RescueSpeech** ASR with Whisper model
This folder contains the script to fine-tune a Whisper model ([openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2/tree/main)) on the RescueSpeech dataset.
- Link to dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8030657.svg)](https://doi.org/10.5281/zenodo.8030657)
- Language: German (DE)

# How to run
As discussed in the paper we use multiple training strategies to create a robust speech recognition system that operates in the SAR (search and rescue) domain.

1. **Pre-training**: Directly infer on RescueSpeech dataset (clean/noisy inputs) without any fine-tuning.
    ```
    python train_with_whisper.py hparams/train_hf_whisper.yaml --number_of_epochs 0 --input_type clean_wav/noisy_wav
    ```

2. **Clean-training**: Fine-tune the ASR model on RescueSpeech clean dataset.
    ```
    python train_with_whisper.py hparams/train_hf_whisper.yaml --input_type clean_wav
    ```

3. **Multi-condition training**: Fine-tune the ASR model on an equal mix of clean and noisy audio from the RescueSpeech noisy dataset.
    ```
    python train_with_whisper.py hparams/train_hf_whisper.yaml --input_type clean_noisy_mix
    ```

*Note* <br>
To ensure accurate inference on clean and noisy inputs, please modify the `pretrained_whisper_model` parameter in the `train_hf_whisper.yaml` file accordingly. When conducting inference for the clean training strategy, ensure that `pretrained_whisper_model` points to the best saved checkpoint obtained during clean training (similarly for multi-condition training). Additionally, it is recommended to perform inference on both clean and noisy test inputs for comprehensive evaluation.


# Test Results
Here we show test WERs using different training strategies on clean and noisy speech inputs from RescueSpeech dataset.
Clean WER and Noisy WER represent WER on clean and noisy test inputs respectively.

| Release | Type                        |   Clean WER   |   Noisy WER   |   HuggingFace link    | Full Model link |
|:--------|:----------------------------|:--------------|:--------------|:---------------------:|:----------------|
|06-12-23 | Pre-training                |    27.01      |    76.14      |   *link*              | *link*          |
|06-12-23 | Clean training              |    34.65      |    73.07      |   *link*              | *link*          |
|06-12-23 | Multi condition training    |    40.73      |    78.02      |   *link*              | *link*          |


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing**
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
Citing the paper
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
