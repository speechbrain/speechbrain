# Language Model with Symbolic Music Datasets
This folder contains recipes for training an autoregressive model for modeling symbolic music. We support JSB Chorales, Muse, Nottingham, Piano-midi and MAESTRO datasets.
It converts Midi or pickle based files into useable CSV files split into train, test and validation sets.

You can download MAESTRO at https://magenta.tensorflow.org/datasets/maestro#v200 (maestro-v2.0.0-midi.zip)
You can download the other 4 pickle files at http://www-etud.iro.umontreal.ca/~boulanni/icml2012 (Piano-roll)

# Extra Dependency:
Make sure you have the MusPy library installed. If not, type:
pip install -r extra_dependencies.txt

Hao-Wen Dong, Ke Chen, Julian McAuley, and Taylor Berg-Kirkpatrick, “MusPy: A Toolkit for Symbolic Music Generation,” in Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR), 2020.

# How to run:
python train.py hparams/RNNLM_{dataset}.yaml
python generate.py hparams/RNNLM_{dataset}.yaml

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
