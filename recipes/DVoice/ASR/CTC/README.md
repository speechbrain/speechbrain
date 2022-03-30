# DVoice ASR with CTC based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the DVoice dataset: [Link](https://zenodo.org/record/6342622)

# Data preparation
[DVoice](https://dvoice.ma) attempts to provide automatic voice processing solutions for African languages and dialects. We use preprocessing techniques including voice augmentation to fill the data gap for each language.

# How to run
python train.py hparams/{hparam_file}.py

# Languages
Here is a list of the different African languages and dialects that we tested:
- Darija
- Swahili
- Wolof
- Fongbe
- Amharic

# Results

| Language | DVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:|
| Darija (Moroccan Arabic) | v2.0 | train_dar_with_wav2vec.yaml | No | 5.51 | 18.46 | 5.85 | 18.28 | [Link](https://huggingface.co/nairaxo/dvoice-darija) | 
| Swahili | v2.0 | train_sw_with_wav2vec.yaml | No | 8.83 | 22.78 | 9.46 | 23.16 | [Link](https://huggingface.co/nairaxo/dvoice-swahili) |
| Multilingual (Darija, Swahili, Wolof, Fongbe, Amharic) | v2.0 | train_multi_with_wav2vec.yaml | No | 7.98 | 24.82 | 7.75 | 24.56 | [Link](https://huggingface.co/nairaxo/dvoice-multilingual) |


## Performances of DVoice Multilingual on each language
Dataset Link | Language | Test WER |
|:---------------------------:| -----:| -----:|
| [DVoice](https://zenodo.org/record/6342622) | Darija | 13.27 |
| [DVoice/VoxLingua107](https://zenodo.org/record/6342622) + [ALFFA](https://github.com/besacier/ALFFA_PUBLIC) | Swahili | 29.31 |
| [ALFFA](https://github.com/besacier/ALFFA_PUBLIC) | Fongbe | 10.26 |
| [ALFFA](https://github.com/besacier/ALFFA_PUBLIC) | Wolof | 21.54 |
| [ALFFA](https://github.com/besacier/ALFFA_PUBLIC) | Amharic | 31.15 |



## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for DVoice.

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
