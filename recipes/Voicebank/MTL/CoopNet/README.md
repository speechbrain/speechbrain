# Training Speech Enhancement + Speech Recognition with CoopNet

Recipe written by Nicolas Duchêne, Sreeramadas Sai Aravind, and Émile Dimas.

This recipe uses a few extra packages, please install them with

    pip install -r extra-dependencies.txt

To use this recipe, one needs to have the VoiceBank dataset available, and replace the placeholder value in the yaml file for where you have the dataset.

Simply use

    python train.py hparams/train_3layer.yaml --data_folder path/to/noisy-vctk-16k

This recipe makes a 3 layer cooperative network.
The fuse mask is an attention mask.

The models and skip connections are kept in the models/ folder, for clarity.

To train this from scratch, there are other yaml files that can be used

    python train.py hparams/pretrain_asr_and_se.yaml --data_folder path/to/noisy-vctk-16k
    python train.py hparams/pretrain_1layer.yaml --data_folder path/to/noisy-vctk-16k --ckpt_path path/to/pretrained
    python train.py hparams/train_3layer.yaml --data_folder path/to/noisy-vctk-16k # change pretrainer paths

Just make sure to change the pretrainer paths so that they point to the right checkpoints.

Currently, the best performance is the following, with intermediate training points reported:

| Input | Mask Loss           | PESQ | eSTOI | dev PER | tst PER  | Epochs |
|-------|---------------------|:----:|:-----:|:-------:|:--------:|:------:|
| Clean | CTC only            |  -   |   -   | 14.29   |    -     | 10     |
| Noisy | MSE on SE only      | 2.68  | 93.4  |    -    |    -     | 50     |
| *Joint Training using pretrained modules*                                |
| Noisy | MSE + CTC 1 layer   | 2.66 | 84.7  | 19.10   | 18.37    | 25     |
| Noisy | MSE + CTC 2 layers  | 2.75 | 85.1  | 16.32   | 16.01    | 50     |
| Noisy | MSE + CTC 3 layers  | 2.86 | 85.4  | 16.26   | 15.77    | 30     |


# **Citing CoopNet**
This recipe is insipred by this work:

@INPROCEEDINGS{CoopNet,
  author={Ravanelli, Mirco and Brakel, Philemon and Omologo, Maurizio and Bengio, Yoshua},
  booktitle={2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={A network of deep neural networks for Distant Speech Recognition}, 
  year={2017},
  pages={4880-4884}}

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

