# TIMIT ASR with seq2seq models.
This folder contains the scripts to train a seq2seq RNNN-based system using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1

# Running the Code

To execute the code, use the following command:

```
python train.py hparams/train.yaml --data_folder=your_data_folder/TIMIT --jit
```

**Important Note on Compilation**:
Enabling the just-in-time (JIT) compiler with --jit significantly improves code performance, resulting in a 50-60% speed boost. We highly recommend utilizing the JIT compiler for optimal results.
This speed improvement is observed specifically when using the CRDNN model.

# Results

| Release | hyperparams file | Val. PER | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml |  12.50 | 14.07 | https://www.dropbox.com/sh/cran9y7da18ehb1/AADQ7Nu2eNuNF6V_vyqVAlA_a?dl=0 | 1xV100 16GB |
| 21-04-08 | train_with_wav2vec2.yaml |  7.11 | 8.04 | https://www.dropbox.com/sh/ablljzwv5rl7007/AAAKlTlFw3TZ_lZFZYwNpd8la?dl=0 | 1xV100 32GB |

The output folders with checkpoints and logs for TIMIT recipes can be found [here](https://www.dropbox.com/sh/059jnwdass8v45u/AADTjh5DYdYKuZsgH9HXGx0Sa?dl=0).

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
