# Speaker Diarization on AMI corpus
This directory contains the scripts for speaker diarization on the AMI corpus (http://groups.inf.ed.ac.uk/ami/corpus/).

# Extra requirements
The code requires sklearn as an additional dependency. To install it, type:
pip install sklearn

# How to run
python experiment.py hparams/ecapa_tdnn.yaml

# Speaker Diarization using Deep Embedding and Spectral Clustering
The script assumes the pre-trained model. Please refer to speechbrain/recipes/VoxCeleb/SpeakerRec/README.md to know more about the available pre-trained models that can easily be downloaded.
You can also train the speaker embedding model from scratch using instructions in the same file. Use the following command to run diarization on AMI corpus.

`python experiment.py hparams/xvectors.yaml`
`python experiment.py hparams/ecapa_tdnn.yaml`

# Performance Summary using Xvector model trained on VoxCeleb1+VoxCeleb2 dataset
Xvectors : Dev = 4.34 % | Eval = 4.45 %
ECAPA   :  Dev = 2.19 % | Eval = 2.74 %
ECAPA_big: Dev = 2.16 % | Eval = 2.72 %

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