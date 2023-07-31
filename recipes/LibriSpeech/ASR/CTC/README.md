# LibriSpeech ASR with CTC and pre-trained wav2vec2 or whisper models.
This folder contains the scripts to finetune a wav2vec2 or a whisper based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12.

**Supported pre-trained wav2vec2:** [SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/self-supervised-learning/wav2vec2) and [HuggingFace](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice/self-supervised-learning/wav2vec2)

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# How to run
```
python train_with_wav2vec.py hparams/file.yaml
```
```
python train_with_whisper.py hparams/file.yaml
```
To run a fine-tuning of "WavLM" with signal downsampled inputs (for faster training and inferences)

```
python train_with_wav2vec.py hparams/downsampled/train_hf_wavlm_signal_downsampling.yaml --downsampling_factor 2
```

# KenLM n-gram CTC rescoring
To enable n-gram rescoring during the decoding, you can download the LibriSpeech official LM from [here](https://www.openslr.org/11/). Please make sure to install the extra dependencies first. Any KenLM language model may be used with this rescoring technique. Results are reported without rescoring.

# Results

| Release | Hyperparams file | Finetuning Split | Test Clean WER | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :-----:| :--------:|
| 09-09-21 | train_hf_wav2vec.yaml | 960h | 1.90 | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) | [Link](https://www.dropbox.com/sh/qj2ps85g8oiicrj/AAAxlkQw5Pfo0M9EyHMi8iAra?dl=0) | 1xRTX8000 48GB |
| 22-09-22 | train_sb_wav2vec.yaml | 960h | 4.2 | Not Avail. | Not Avail. | 2xTesla V100 32GB |
| 06-12-23 | train_hf_whisper.yaml (small) | 960h | 4.89 | Not Avail. | Not Avail. | 4xRTX 2080 Ti |

# Downsampling inputs for faster fine-tuning and inferences using SSL Models
This repository contains the code allowing to reproduce part of the results obtained in the paper : "Fine-tuning Strategies for Faster Inference using Speech Self-Supervised Models:  A Comparative Study"
The reported experiments are the ones leading to largest inference time reductions while keeping lower error rates, using a downsampling of the input sequences. You can download LibriSpeech at http://www.openslr.org/12.

### Downsampling Results with Librispeech train-clean-100 split
The inference times shown here are for running the whole test-clean LibriSpeech split, and are in seconds. MACs shown here are the mean MACs for a test batch
These results are obtained using WavLM Large finetuned only on the train-clean-100 split of LibriSpeech (100 hours of speech)

| Name  | Factor | WER   | GPU- Inference Time | CPU - Inference Time | WER-LM | GPULM - Inference Time | CPULM - Inference Time | MACs (G) |
|-------|--------|-------|---------------------|----------------------|--------|------------------------|------------------------|----------|
| No SD | 1      |  4.09 |                 134 |                 1121 |   3.31 |                    152 |                   1128 | 386.538  |
| CL2   |      2 | 4.61  |                  84 |                  582 | 3.48   |                     98 |                    600 | 192.97   |
| CL3   |      3 | 5.47  |                  69 |                  414 |   4.12 |                     91 |                    436 | 134.864  |
| AV2   |      2 | 4.93  |                  80 |                  570 | 3.66   |                     98 |                    578 | 192.97   |
| AV3   |      3 |  6.01 |                  64 |                  406 | 4.27   |                     90 |                    422 | 134.864  |
| SD2   |      2 | 4.85  |                  86 |                  569 | 3.58   |                     97 |                    575 | 192.97   |
| SD3   |      3 | 5.83  |                  72 |                  427 |   4.08 |                     89 |                    458 | 134.864  |

CL: Learned convolutional downsampling

SD : Signal downsampling

AV : Averaging window

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
If you use the downsampling approach, please cite :

```bibtex
@article{zaiem2023fine,
  title={Fine-tuning Strategies for Faster Inference using Speech Self-Supervised Models: A Comparative Study},
  author={Zaiem, Salah and Algayres, Robin and Parcollet, Titouan and Essid, Slim and Ravanelli, Mirco},
  journal={arXiv preprint arXiv:2303.06740},
  year={2023}
}
```


