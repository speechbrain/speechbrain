# LibriSpeech ASR with CTC only or pre-trained wav2vec2 or whisper models.
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
python train.py hparams/file.yaml
python train_with_wav2vec.py hparams/file.yaml
python train_with_whisper.py hparams/file.yaml
```
To run a fine-tuning of "WavLM" with signal downsampled inputs (for faster training and inferences)

```
python train_with_wav2vec.py hparams/downsampled/train_hf_wavlm_signal_downsampling.yaml --downsampling_factor 2
```
To train a model from scratch (without any pre-training), please firstly go to the Tokenizer folder to train a tokenizer:

```
cd ../../Tokenizer
python train.py hparams/128_bpe.yaml
```
Then, go back to this directory. You can train a Branchformer CTC model with:

```
python train.py hparams/train_branchformer.yaml
```
or a Conformer CTC model with:

```
python train.py hparams/train_conformer.yaml
```
# WFST-based CTC loss
To fine-tune a wav2vec 2.0 model with the WFST-based CTC loss, you can use the `train_with_wav2vec_k2.py` script. This will create a `lang` directory inside your output folder, which will contain the files required to build a lexicon FST. The tokenization method used here is a very basic character-based tokenization (e.g. `hello -> h e l l o`).

To use this script, you will first need to install `k2`. The integration has been tested with `k2==1.24.4` and `torch==2.0.1`, although it should also work with any `torch` version as long as `k2` supports it (compatibility list [here](https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/index.html)). You can install `k2` by following the instructions [here](https://k2-fsa.github.io/k2/installation/from_wheels.html#linux-cuda-example).

Using a lexicon FST (L) while training can help guide the model to better predictions. When decoding, you can either use a simple HL decoding graph (where H is the ctc topology), or use an HLG graph (where G is usually a 3-gram language model) to further improve the results. In addition, whole lattice rescoring is also supported. This typically happens with a 4-gram language model. See `hparams/train_with_wav2vec_k2.yaml`` for more details.

If you choose to use a 3-gram or a 4-gram language model, you can either supply pre-existing ARPA LMs for both cases, including the option to train your own, or you can specify the name in the YAML docstring for automatic downloading. Comprehensive instructions are provided in `train_hf_wav2vec_k2.yaml`.

For those interested in training their own language model, please consult our recipe at LibriSpeech/LM/train_ngram.py.

Example usage:
```
python train_with_wav2vec_k2.py hparams/train_hf_wav2vec_k2.yaml --data_folder=/path/to/LibriSpeech
```

To use the HLG graph (instead of the default HL), pass `--compose_HL_with_G=True`. To use the 4-gram LM for rescoring, pass the `--decoding_method=whole-lattice-rescoring` argument. Note that this will require more memory, as the whole lattice will be kept in memory during the decoding. In this recipe, the `lm_scale` used by default is 0.4. This is the value that gave the best results in our HL-graph experiments after trying scales of `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]`. When rescoring is used alongside the HLG graph, the 4-gram seems to not bring any improvement. The best lm scale in that case was 0.2 (the lowest value we tried).

# KenLM n-gram CTC rescoring
To enable n-gram rescoring during the decoding, you can download the LibriSpeech official LM from [here](https://www.openslr.org/11/). Please make sure to install the extra dependencies first. Any KenLM language model may be used with this rescoring technique. The n-gram can either be a binary or an arpa file, but note that the binary format is faster to load. The following command shows how to use the official LibriSpeech 4-gram LM with SpeechBrain:
```bash
wget https://openslr.elda.org/resources/11/4-gram.arpa.gz
gzip -d 4-gram.arpa.gz
python train_with_wav2vec.py hparams/file.yaml --kenlm_model_path='4-gram.arpa'
```

# Rescoring with a Neural Language Model
Two yamls do support LM rescoring: `train_hf_wav2vec_rnn_rescoring.yaml` and `train_hf_wav2vec_transformer_rescoring.yaml`. The first one uses a RNN LM, while the second one uses a Transformer LM. Both LMs are already pretrained on LibriSpeech (see [RNNLM](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) and [TransformerLM](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech)). The acoustic model (wav2vec2) generates a list of hypotheses (called n-best), which are then rescored (aka re-ranked) by the LM. The LM rescores by computing the score of each hypothesis by summing the log-probabilities of each tokens with respect to the previous tokens. The LM score is then added to the acoustic model score to obtain the final score. Using this technique, will results in better WERs. For instance, we went from 1.95 to 1.57 of WER. However, note that the inference time will be slower.

Two parameters need to be tuned: `topk` (and `beam_size` to have enough topk) and `lm_weight`. Increasing `topk` will increase the number of hypotheses to be rescored, and ultimately the inference time. Increasing `lm_weight` will increase the importance of the LM score in the final score. The following command shows how to use the RNN LM with SpeechBrain:
```bash
python train_with_wav2vec.py hparams/train_hf_wav2vec_rnn_rescoring.yaml --data_folder=/path/to/LibriSpeech/ --topk=50 --beam_size=50 --lm_weight=0.5
```
Note: by default, `topk` is set to 20 as it gives a good trade-off between WER and inference time.

# Results

| Release | Hyperparams file | Decoding method | Finetuning Split | Test-clean WER | GPU- Test-clean Inference Time | Test-other WER | GPU- Test-other Inference Time |  HuggingFace link | Full model link | Inference GPUs | Training GPUs |
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
| 23-01-24 | train_hf_wav2vec_k2.yaml | k2CTC + HL graph + 1best decoding + test batch size = 1 | 960h | 1.83 | Not Avail. | 3.82 | Not Avail. | Not Avail. | [Link](https://www.dropbox.com/scl/fo/678rj1a44jt4zrxjwaetu/h?rlkey=x0xwz31nkl01qwr3k5ivtywvz&dl=0) |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 23-01-24 | train_hf_wav2vec_k2.yaml | k2CTC + HLG graph + 1best decoding + test batch size = 1 | 960h | 1.69 | Not Avail. | 3.44 | Not Avail. | Not Avail. | [Link](https://www.dropbox.com/scl/fo/c91vqlr8ase90x0m7u3v3/h?rlkey=duh55n0qzlfnfhy4auu0a4f8g&dl=0) |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 23-01-24 | train_hf_wav2vec_k2.yaml | k2CTC + HL graph + whole lattice rescoring + test batch size = 1 | 960h | 1.72 | Not Avail. | 3.51 | Not Avail. | Not Avail. | [Link](https://www.dropbox.com/scl/fo/mx6hd4zc0iyzqvixxre6q/h?rlkey=xxbpb949btmeiecw30be5qwhj&dl=0) |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 23-01-24 | train_hf_wav2vec_k2.yaml | k2CTC + HLG graph + whole lattice rescoring + test batch size = 1 | 960h | 1.81 | Not Avail. | 3.57 | Not Avail. | Not Avail. | [Link](https://www.dropbox.com/scl/fo/kj2ujqj3votq7ue6ydh0l/h?rlkey=mibyoria19zasvuxs0iwx6plt&dl=0) |  1xRTX2080Ti 12GB | 1xRTX2080Ti 12GB |
| 08-12-23 | train_hf_wav2vec.yaml | CTCBeamSearch + RNNLM Rescorer  + test batch size = 1 + topk = 100  | 960h | 1.69 | 26mins15 | 3.55 | 32min44s | Not Avail. | [Link](https://www.dropbox.com/sh/k4ixa211yp5b1tm/AAD85sgYw2CH7NKk_qKMO9Tja?dl=0) |  1x A100 40GB | 2xTesla V100 40GB |
| 08-12-23 | train_hf_wav2vec.yaml | CTCBeamSearch + TransformerLM Rescorer + test batch size = 1 + topk = 100 | 960h | 1.57 | 26mins56s | 3.37 | 32min46 | Not Avail. | [Link](https://www.dropbox.com/sh/ijqalvre7mm08ng/AAD_hsN-8dBneUMMkELsOOxga?dl=0) |  1x A100 40GB | 2xTesla V100 32GB |
| 06-12-23 | train_branchformer.yaml (25.9M) | 960h | 3.6 (no LM) | Not Avail. | Not Avail. | 8xA40 46G |
| 06-12-23 | train_conformer.yaml (28.8M) | 960h | 3.7 (no LM) | Not Avail. | Not Avail. | 8xA40 46G |


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
If you use the downsampling approach, please cite :

```bibtex
@article{zaiem2023fine,
  title={Fine-tuning Strategies for Faster Inference using Speech Self-Supervised Models: A Comparative Study},
  author={Zaiem, Salah and Algayres, Robin and Parcollet, Titouan and Essid, Slim and Ravanelli, Mirco},
  journal={arXiv preprint arXiv:2303.06740},
  year={2023}
}
```


