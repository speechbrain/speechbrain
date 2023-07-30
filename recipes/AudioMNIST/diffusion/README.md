# Denoising Diffusion Probabilistic Model Demo
This folder contrains scripts for running a Denoising Diffusion Probabilistic Model
generative model with the [AudioMNIST](https://huggingface.co/datasets/flexthink/audiomnist) dataset.

https://arxiv.org/pdf/2006.11239.pdf


# Training
For the unconditioned model, run the following:
`python train.py hparams/train.yaml`

For the model conditioned on the speaker, run the following:
`python train.py hparams/train.yaml --speaker_conditioned true`

For the model conditioned on the digit (i.e. a vastly simplified TTS use case), run the following:
`python train.py hparams/train.yaml --digit_conditioned true`

For the latent diffusion model, run the following:
`python train.py hparams/train_latent.yaml`

The scripts will output the results to <output_folder>/samples, for every training epoch

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


