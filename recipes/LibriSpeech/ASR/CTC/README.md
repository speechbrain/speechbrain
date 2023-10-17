# LibriSpeech ASR with CTC and pre-trained wav2vec2 or whisper models.
This folder contains the scripts to finetune a wav2vec2 or a whisper based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12.
The loss function is the CTC loss and it is implemented in two different ways:
- Using the [CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) from PyTorch.
- Using the [CTC implementation](https://github.com/k2-fsa/k2/blob/master/k2/python/k2/ctc_loss.py) from K2 (WFST-based). For an example of such recipe, check the `train_with_wav2vec_k2.py` file.

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

# WFST-based CTC loss
To fine-tune a wav2vec 2.0 model with the WFST-based CTC loss, you can use the `train_with_wav2vec_k2.py` script. This will create a `lang` directory inside your output folder, which will contain the files required to build a lexicon FST. The tokenization method used here is a very basic character-based tokenization (e.g. `hello -> h e l l o`).

To use this script, you will first need to install `k2`. The integration has been tested with `k2==1.24.3` and `torch==1.13.1`, although it should also work with `torch==2.0.1`. You can install `k2` by following the instructions [here](https://k2-fsa.github.io/k2/installation/from_wheels.html#linux-cuda-example).

Using a lexicon FST (L) while training can help guide the model to better predictions. When decoding, you can either use a simple HL decoding graph (where H is the ctc topology), or use an HLG graph (where G is usually a 3-gram language model) to further improve the results. In addition, whole lattice rescoring is also supported. This typically happens with a 4-gram language model. See `hparams/train_with_wav2vec_k2.yaml`` for more details.

If you choose to either use a 3-gram or a 4-gram language model, you will need to provide pre-existing ARPA LMs for both cases. Those can be found in LibriSpeech's official repository: https://www.openslr.org/11/. The 3-gram LM is called `3-gram.pruned.1e-7.arpa.gz`, while the 4-gram LM is called `4-gram.arpa.gz`. You can download them and unzip them in the same folder. Then, you can pass the path to the folder containing the ARPA LMs to the `--lm_dir` argument. This will automatically build the required FSTs for you.

Besides the ARPA files, it is also advised to download the `librispeech-vocab.txt` file under your `data_folder` (i.e. where the LibriSpeech dataset is downloaded). This file contains extra words that will be used in the lexicon creation to make it more robust to OOVs. You can download it from [here](http://www.openslr.org/resources/11/librispeech-vocab.txt).

Example usage:
```
python train_with_wav2vec_k2.py hparams/train_hf_wav2vec_k2.yaml --data_folder=/path/to/LibriSpeech --lm_dir=/path/to/LibriSpeech/LM
```

To use the HLG graph (instead of the default HL), pass `--use_HLG=True`. To use the 4-gram LM for rescoring, pass the `--decoding_method=whole-lattice-rescoring` argument. Note that this will require more memory, as the whole lattice will be kept in memory during the decoding. In this recipe, the `lm_scale` used by default is 0.4. This is the value that gave the best results in our HL-graph experiments after trying scales of `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]`. When rescoring is used alongside the HLG graph, the 4-gram seems to not bring any improvement. The best lm scale in that case was 0.2 (the lowest value we tried).

# KenLM n-gram CTC rescoring
To enable n-gram rescoring during the decoding, you can download the LibriSpeech official LM from [here](https://www.openslr.org/11/). Please make sure to install the extra dependencies first. Any KenLM language model may be used with this rescoring technique. The n-gram can either be a binary or an arpa file, but note that the binary format is faster to load. The following command shows how to use the official LibriSpeech 4-gram LM with SpeechBrain:
```bash
wget https://openslr.elda.org/resources/11/4-gram.arpa.gz
gzip -d 4-gram.arpa.gz
python train_with_wav2vec.py hparams/file.yaml --kenlm_model_path='4-gram.arpa'
```

# Results

| Release | Hyperparams file | Decoding method | Finetuning Split | Test-clean WER | GPI- Test-clean Inference Time | Test-other WER | GPU- Test-other Inference Time |  HuggingFace link | Full model link | Inference GPUs | Training GPUs |
|:-------------:|:---------------------------:|  :----------:|  :-----:| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:| :--------:| :--------:|
| 05-08-23 | train_hf_wav2vec.yaml | GreedySearch | 960h  | 2.12 | 1min30s | 4.31| 1min24s | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) | [Link](https://www.dropbox.com/sh/qj2ps85g8oiicrj/AAAxlkQw5Pfo0M9EyHMi8iAra?dl=0) | 1xRTX3090 24GB | 1xA100 40GB |
| 05-08-23 | train_hf_wav2vec.yaml | GreedySearch  + test batch size = 1| 960h  | 1.95 | 2min09s | 3.97| 2min21s | Not Avail. | [Link](https://www.dropbox.com/sh/8zqufkmegbgpsa8/AACB6MMJ_efbGDvTi5ZhB4pQa?dl=0) | 1xRTX3090 24GB | 1xA100 40GB |
| 05-08-23 | train_hf_wav2vec.yaml | CTCBeamSearch  + test batch size = 1| 960h  | 1.92 | 2min22s | 3.97 | 2min16s | Not Avail. | [Link](https://www.dropbox.com/sh/8zqufkmegbgpsa8/AACB6MMJ_efbGDvTi5ZhB4pQa?dl=0) | 1xRTX3090 24GB | 1xA100 40GB |
| 05-08-23 | train_hf_wav2vec.yaml | CTCPrefixBeamSearch  + test batch size = 1| 960h | 1.92 | 2min45s | 3.97 | 2min21s | Not Avail. | [Link](https://www.dropbox.com/sh/8zqufkmegbgpsa8/AACB6MMJ_efbGDvTi5ZhB4pQa?dl=0) | 1xRTX3090 24GB | 1xA100 40GB |
| 05-08-23 | train_hf_wav2vec.yaml | CTCBeamSearch + 4-gram  + test batch size = 1| 960h  | 1.75  | 2min37s | 3.67 | 2min20s | Not Avail. | [Link](https://www.dropbox.com/sh/8zqufkmegbgpsa8/AACB6MMJ_efbGDvTi5ZhB4pQa?dl=0) | 1xRTX3090 24GB | 1xA100 40GB |
| 05-08-23 | train_hf_wav2vec.yaml | CTCPrefixBeamSearch + 4-gram  + test batch size = 1| 960h  | 1.80 | 2min38s | 3.78 | 2min25s |Not Avail. | [Link](https://www.dropbox.com/sh/8zqufkmegbgpsa8/AACB6MMJ_efbGDvTi5ZhB4pQa?dl=0) | 1xRTX3090 24GB | 1xA100 40GB |
| 22-09-22 | train_sb_wav2vec.yaml | GreedySearch | 960h | 4.2 | Not Avail. | Not Avail. | Not Avail. | Not Avail. | Not Avail. | Not Avail.| 2xTesla V100 32GB |
| 08-12-23 | train_hf_whisper.yaml (small) | CTCBeamSearch  + test batch size = 1 | 960h | 4.72 | 3.08 | 12.66 |3.30 | Not Avail. | [Link](https://www.dropbox.com/sh/zmtp13huxn02fot/AADyKL5q0MwRhEG1-WbSXDWda?dl=0) |  1xRTX3090 24GB | 2xTesla V100 32GB |
| 08-12-23 | train_hf_whisper.yaml (small) | CTCPrefixBeamSearch  + test batch size = 1 | 960h | 4.73 | 3.19 | 12.65 |3.39 | Not Avail. | [Link](https://www.dropbox.com/sh/zmtp13huxn02fot/AADyKL5q0MwRhEG1-WbSXDWda?dl=0) |  1xRTX3090 24GB | 2xTesla V100 32GB |
| 08-12-23 | train_hf_whisper.yaml (small) | CTCBeamSearch + 4-gram  + test batch size = 1 | 960h | 4.37 | 3.16 | 11.76 | 3.43 | Not Avail. | [Link](https://www.dropbox.com/sh/zmtp13huxn02fot/AADyKL5q0MwRhEG1-WbSXDWda?dl=0) |  1xRTX3090 24GB | 2xTesla V100 32GB |
| 08-12-23 | train_hf_whisper.yaml (small) | CTCPrefixBeamSearch + 4-gram  + test batch size = 1 | 960h | 4.44 | 3.30 | 11.89 | 3.47 | Not Avail. | [Link](https://www.dropbox.com/sh/zmtp13huxn02fot/AADyKL5q0MwRhEG1-WbSXDWda?dl=0) |  1xRTX3090 24GB | 2xTesla V100 32GB |
| 10-05-23 | train_hf_wav2vec_k2.yaml | k2CTC + HL graph + test batch size = 1 | 960h | 1.83 | Not Avail. | Not Avail. | Not Avail. | Not Avail. | Not Avail. |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 10-05-23 | train_hf_wav2vec_k2.yaml | k2CTC + HLG graph + test batch size = 1 | 960h | 1.73 | Not Avail. | Not Avail. | Not Avail. | Not Avail. | Not Avail. |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 10-05-23 | train_hf_wav2vec_k2.yaml | k2CTC + HL graph + 4-gram rescoring + test batch size = 1 | 960h | 1.75 | Not Avail. | Not Avail. | Not Avail. | Not Avail. | Not Avail. |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 10-05-23 | train_hf_wav2vec_k2.yaml | k2CTC + HLG graph + 4-gram rescoring + test batch size = 1 | 960h | 2.60 | Not Avail. | Not Avail. | Not Avail. | Not Avail. | Not Avail. |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |

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


