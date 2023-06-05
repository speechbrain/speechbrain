# **RescueSpeech** ASR with CTC models
This folder contains the script to fine-tune Wav2vec2.0 & WavLM-large CTC based model on the RescueSpeech dataset.
The Wav2vec2/WavLM model is first trained on German CommonVoice10.0 corpus, and later fine-tuned on the RescueSpeech data.
- [`facebook/wav2vec2-large-xlsr-53-german`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-german/tree/main) : See *Gdrive link* for the pre-trained models.
- [`microsoft/wavlm-large`](https://huggingface.co/microsoft/wavlm-large) : See *Gdrive link* for the pre-trained models.
- Link to dataset: *put link here*
- Language: German (DE)


# How to run
We provide hyper-parameter file for both Wav2vec2.0 and WavLM-large. Please use them appropriately. As discussed in the paper we use multiple training strategies  to create a robust speech recognition system that operates in the SAR (search and rescue) domain.

1. **Pre-training**: Use the CommonVoice pre-trained Wav2vec2.0 and WavLM model and directly infer on RescueSpeech dataset (clean/noisy inputs) without any fine-tuning.
    ```
    python train.py hparams/train_with_wav2vec.yaml --number_of_epochs 0 --input_type clean_wav/noisy_wav
    ```

2. **Clean-training**: Fine-tune the ASR model on RescueSpeech clean dataset.
    ```
    python train.py hparams/train_with_wav2vec.yaml --input_type clean_wav
    ```

3. **Multi-condition training**: Fine-tune the ASR model on an equal mix of clean and noisy audio from the RescueSpeech noisy dataset.
    ```
    python train.py hparams/train_with_wav2vec.yaml --input_type clean_noisy_mix
    ```

*Note* <br>
To ensure accurate inference on clean and noisy inputs, please modify the `pretrained_wav2vec2_path` parameter in the `*.yaml` file accordingly. When conducting inference for the clean training strategy, ensure that `pretrained_wav2vec2_path` points to the best saved checkpoint obtained during clean training (similarly for multi-condition training). Additionally, it is recommended to perform inference on both clean and noisy test inputs for comprehensive evaluation.


# Test Results
Here we show test WERs on Wav2vec2.0/WavLM-large using different training strategies on clean and noisy speech inputs from RescueSpeech dataset.
Clean WER and Noisy WER represent WER on clean and noisy test inputs respectively.

1. **Model: Wav2vec2**
    | Release | Type                        |   Clean WER   |   Noisy WER   |   HuggingFace link    | Full Model link |
    |:--------|:----------------------------|:--------------|:--------------|:---------------------:|:----------------|
    |06-12-23 | Pre-training                |    47.95      |    89.16      |   *link*              | *link*          |
    |06-12-23 | Clean training              |    26.94      |    88.09      |   *link*              | *link*          |
    |06-12-23 | Multi condition training    |    30.19      |    76.07      |   *link*              | *link*          |

2. **Model: WavLM**
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
Citing the RescueSpeech dataset
```
paper bib goes here
```
