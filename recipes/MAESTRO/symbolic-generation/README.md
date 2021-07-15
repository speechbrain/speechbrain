# Language Model with Symbolic Music Datasets
This folder contains recipes for training an autoregressive model for modeling symbolic music. We support JSB Chorales, Muse, Nottingham, Piano-midi and MAESTRO datasets.
It converts Midi or pickle based files into useable CSV files split into train, test and validation sets.

The code automatically downloads the datasets. However, you can also do this manually from the links below.

* You can download MAESTRO at `https://magenta.tensorflow.org/datasets/maestro#v200` (maestro-v2.0.0-midi.zip)
* You can download the other 4 pickle files at `http://www-etud.iro.umontreal.ca/~boulanni/icml2012` (Piano-roll)


# Extra Dependency:
Make sure you have the MusPy library installed. If not, type:
```
pip install -r extra_dependencies.txt
```

Hao-Wen Dong, Ke Chen, Julian McAuley, and Taylor Berg-Kirkpatrick, “MusPy: A Toolkit for Symbolic Music Generation,” in Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR), 2020.

# How to train a model:
```
python train.py hparams/RNNLM_{dataset}.yaml --data_path yourpath/dataset
```
For example for JSB chorales dataset, you can call:
```
python train.py hparams/RNNLM_JSB.yaml --data_path yourpath/JSB
```
The code will automatically download the dataset to the specified path.

# How to generate from a trained model

```
python generate.py yourpath/trained_model_CKPT
```
You need to specify the path to the model checkpoint to be able to generate. You need to copy the corresponding `hyperparams.yaml` inside the checkpoint folder to be able to run this script.


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
