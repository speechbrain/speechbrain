# **RescueSpeech** ASR with CTC models
This folder contains the script to fine-tune WavLM-large CTC based model on the RescueSpeech dataset.
The WavLM model is first trained on German CommonVoice10.0 corpus, and later fine-tuned on the RescueSpeech data.

- [`microsoft/wavlm-large`](https://huggingface.co/microsoft/wavlm-large) : See *hf link* to use the German CommonVoice fine-tuned model.
- Link to dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8030657.svg)](https://doi.org/10.5281/zenodo.8030657)
- Language: German (DE)


# How to run
We provide hyper-parameter file for WavLM-large. As discussed in the paper we use multiple training strategies  to create a robust speech recognition system that operates in the SAR (search and rescue) domain.

1. **Pre-training**: Use the CommonVoice fine-tuned WavLM model and directly infer on RescueSpeech dataset (clean/noisy inputs) without any fine-tuning.
    ```
    python train.py hparams/train_with_wavlm.yaml --number_of_epochs 0 --input_type clean_wav/noisy_wav
    ```

2. **Clean-training**: Now we fine-tune the ASR model (CommonVoice fine-tuned WavLM) on RescueSpeech clean dataset.
    ```
    python train.py hparams/train_with_wavlm.yaml --input_type clean_wav
    ```

3. **Multi-condition training**: Now we fine-tune the ASR model (CommonVoice fine-tuned WavLM) on an equal mix of clean and noisy audio from the RescueSpeech noisy dataset.
    ```
    python train.py hparams/train_with_wavlm.yaml --input_type clean_noisy_mix
    ```

*Note* <br>
Please modify the `pretrained_wav2vec2_path` parameter in the `train_with_wavlm.yaml` file accordingly. When training, it should point the
model trained on German CommonVoice (available at *wavlm-large-cv-trained hf link*). And when conducting inference for the clean training strategy, ensure that `pretrained_wav2vec2_path` points to the best saved checkpoint obtained during clean training (similarly for multi-condition training). Additionally, it is recommended to perform inference on both clean and noisy test inputs for comprehensive evaluation.


# Test Results
Here we show test WERs on WavLM-large using different training strategies on clean and noisy speech inputs from RescueSpeech dataset.
Clean WER and Noisy WER represent WER on clean and noisy test inputs respectively.

| Release | Type                        |   Clean WER   |   Noisy WER   |   HuggingFace link    | Full Model link |
|:--------|:----------------------------|:--------------|:--------------|:---------------------:|:----------------|
|06-12-23 | Pre-training                |    46.31      |    87.42      |   *link*              | *link*          |
|06-12-23 | Clean training              |    24.77      |    77.89      |   *link*              | *link*          |
|06-12-23 | Multi condition training    |    25.41      |    72.52      |   *link*              | *link*          |


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
