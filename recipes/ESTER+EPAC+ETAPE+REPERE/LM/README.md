# Language Model with ESTER+EPAC+ETAPE+REPERE
This folder contains recipes for training language models for the above datasets.
It supports n-gram LM.
Depending on the ASR token type ("phone", or "char"), an apostrophe (') should
be preceded by a space or not.

Example:
* "C'est" -> modeled as "sɛ" -> mapped to "C'EST"
* "C'est" -> modeled as 2 words "C'" and "EST" -> transcribed to "C' EST"

## Installing Extra Dependencies

If you want to train an n-gram, in this recipe we are using the popular KenLM library. Let's start by installing the Ubuntu library prerequisites. For a complete guide on how to install required dependencies, please refer to [this](https://kheafield.com/code/kenlm/dependencies/) link:
 ```
 sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
 ```

 Next, we need to start downloading and unpacking the KenLM repo.
 ```
 wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
 ```

KenLM is written in C++, so we'll make use of cmake to build the binaries.
 ```
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
 ```

Now, make sure that the executables are added to your .bashrc file. To do it,
- Open the ~/.bashrc file in a text editor.
- Scroll to the end of the file and add the following line:  ```export PATH=$PATH:/your/path/to/kenlm/build/bin ```
- Save it and type:  `source ~/.bashrc `

# How to run:
```shell
python train_ngram.py hparams/train_ngram.yaml  --data_folder=your/data/folder
# your/data/folder should point to the directory containing one or all of the following directories: EPAC  ESTER1  ESTER2  ETAPE  REPERE
## ls your/data/folder -> EPAC  ESTER1  ESTER2  ETAPE  REPERE
## ls your/data/folder/ETAPE -> dev  test  tools  train
```

| Release | hyperparams file | Test PP | Model link | GPUs |
| :---     | :---: | :---: | :---: | :---: |
| 24-03-12 | 3-for-char-gram.arpa - train_ngram.yaml | --.-- | [link](https://www.dropbox.com/TODO) | --.-- |
| 24-03-12 | 4-for-char-gram.arpa - train_ngram.yaml | --.-- | [link](https://www.dropbox.com/TODO) | --.-- |
| 24-03-12 | 3-for-phone-gram.arpa - train_ngram.yaml | --.-- | [link](https://www.dropbox.com/TODO) | --.-- |
| 24-03-12 | 4-for-phone-gram.arpa - train_ngram.yaml | --.-- | [link](https://www.dropbox.com/TODO) | --.-- |


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

