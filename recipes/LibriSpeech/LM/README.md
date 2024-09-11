# Language Model with LibriSpeech
This folder contains recipes for training language models for the LibriSpeech Dataset.
It supports n-gram LM, RNN-based LM, and Transformer-based LM.
The scripts is relying on the HuggingFace dataset for RNN/Transformer based LM, which manages data reading and loading from
large text corpora.

You can download LibriSpeech at http://www.openslr.org/12

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

If you want to train an n-gram, in this recipe we are using  the popular KenLM library. Let's start by installing the Ubuntu library prerequisites. For a complete guide on how to install required dependencies, please refer to [this](https://kheafield.com/code/kenlm/dependencies/) link:
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
python train.py hparams/RNNLM.yaml
python train.py hparams/transformer.yaml
python train_ngram.py hparams/train_ngram.yaml  --data_folder=your/data/folder
```

| Release | hyperparams file | Test PP | Model link | GPUs |
| :---     | :---: | :---: | :---: | :---: |
| 20-05-22 | RNNLM.yaml (1k BPE) | --.-- | [link](https://www.dropbox.com/sh/8xpybezuv70ibcg/AAByv2NuNv_ZFXuDdG89-MVPa?dl=0) | 1xV100 32GB |
| 20-05-22 | RNNLM.yaml (5k BPE) | --.-- | [link](https://www.dropbox.com/sh/8462ef441wvava2/AABNfHr07J_0SsdaM1yO5qkxa?dl=0) | 1xV100 32GB |
| 20-05-22 | transformer.yaml | --.-- | [link](https://www.dropbox.com/sh/6uwqlw2tvv3kiy6/AACgvTR5jihyMrugBrpZPFNha?dl=0) | 1xV100 32GB |
| 22-01-24 | 4-gram - train_ngram.yaml | --.-- | [link](https://www.dropbox.com/scl/fi/kkd5jrwthpahn4t7e7sgk/4gram_lm.arpa?rlkey=mc820i9bugpi3oxtwwd6ulz0b&dl=0) | --.-- |
| 22-01-24 | 3-gram - train_ngram.yaml | --.-- | [link](https://www.dropbox.com/scl/fi/juryiq2e50bsbdy1qx540/3gram_lm.arpa?rlkey=3ntfnkn6zxda9memm5zh1mmt9&dl=0) | --.-- |

# Training time
Training a LM takes a lot of time. In our case, it take 3/4 weeks on 4 TESLA V100. Use the pre-trained model to avoid training it from scratch


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
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
