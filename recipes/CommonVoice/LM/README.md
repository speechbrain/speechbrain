
# Training KenLM
This folder contains recipes for training the kenLM-gram model for the CommonVoice Dataset.
Using Wav2Vec2 in combination with a language model can yield a significant improvement, especially when the model is fine-tuned on small speech datasets. This is a guide to explain how one can create an n-gram language model and combine it with an existing fine-tuned Wav2Vec2.


You can download CommonVoice at https://commonvoice.mozilla.org/en

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

We will use the popular KenLM library to build an n-gram. Let's start by installing the Ubuntu library prerequisites. For a complete guide on how to install required dependencies, please refer to [this](https://kheafield.com/code/kenlm/dependencies/) link:
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

 ```
# How to run:
```shell
python train.py hparams/train_kenlm.yaml  --data_folder=your/data/folder
```

# Results
The script trains a n-gram language model, which is stored in the popular ARPA format.
The output folders with checkpoints and logs can be found [here](https://www.dropbox.com/scl/fo/zw505t10kesqpvkt6m3tu/h?rlkey=6626h1h665tvlo1mtekop9rx5&dl=0).




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
