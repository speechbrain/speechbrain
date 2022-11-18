# DiffWave
This folder contrains scripts for running a vocoder using [Diffwave](https://arxiv.org/pdf/2009.09761.pdf)

DiffWave is a versatile diffusion model for audio synthesis, which  produces high-fidelity audio in different waveform generation tasks, including neural vocoding conditioned on mel spectrogram, class-conditional generation, and unconditional generation.

Here it serves as a vocoder that converts a spectrogram into a waveform.

# Training
`python train.py hparams/train.yaml`

The scripts will output the results to <output_folder>/samples, for every training epoch

We suggest using tensorboard_logger by setting `use_tensorboard: True` in the yaml file, thus Tensorboard should be installed.

Training takes about X hours on an nvidia RTX8000.

The training logs are available [here](to add)

You can find the pre-trained model with an easy-inference function on [HuggingFace](to add).

# Fast Sampling
For inference, a fast sampling can be realized by passing user-defined variance schedules. According to the paper, high-quality audios can be generated with only 6 steps.

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