# Speech separation with Binaural-WSJ0Mix
This folder contains some recipes for the Binaural-WSJ0Mix task (2/3 sources). Please refer to [Real-time binaural speech separation with preserved spatial cues](https://arxiv.org/abs/2002.06637) [1] for details.


## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r ../extra_requirements.txt

```
To run it:

```
python train.py hparams/convtasnet-parallel.yaml
                --data_folder yourpath/binaural-wsj0mix/2speakers
                --wsj_root yourpath/to/wsj/
```
The training data will be automatically created from the `wsj_root`, which is the root folder that contains
Note that during training we print the negative SNR instead of SI-SNR because the scale-invariance property of SI-SNR makes it insensitive to power rescaling of the estimated signal, which may fail in preserving the ILD between the outputs.

If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:
```
python train.py hparams/convtasnet-parallel.yaml
                --data_folder yourpath/binaural-wsj0mix/2speakers
                --wsj_root yourpath/to/wsj/
                --test_only
```
# Binaural WSJ0-2mix and WSJ0-3mix dataset creation
* The training data generation scripts can be found from [https://github.com/huangzj421/Binaural-WSJ0Mix](https://github.com/huangzj421/Binaural-WSJ0Mix). But the `train.py` also automatically downloads and generates the data. It puts the data under the path specified in `data_folder`.
* The default command to run that automatically generate the data given wsj0 folder:`python train.py hparams/convtasnet-parallel.yaml --data_folder yourpath/binaural-wsj0mix/2speakers --wsj_root yourpath/wsj0-mix/wsj0`


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training. For this you need to have the WSJ0 dataset (available though LDC at `https://catalog.ldc.upenn.edu/LDC93S6A`).


# Results

Here are the SNRi results (in dB) as well as ITD and ILD errors as the metric for the accuracy of preserving interaural cues:

| | SNRi | delta-ITD | delta-ILD |
| --- | --- | --- | --- |
|ConvTasnet-independent| 12.39 | 6.17 | 0.32 |
|ConvTasnet-cross| 11.9 | 5.69 | 0.37 |
|ConvTasnet-parallel| 16.93 | 2.38 | 0.09 |
|ConvTasnet-parallel-noise| 18.25 | 5.56 | 0.23 |
|ConvTasnet-parallel-reverb| 6.95 | 0.0 | 0.40 |

* ConvTasnet-independent.yaml refers to ConvTasnet is applied to each channel independently.
* ConvTasnet-cross.yaml refers to cross-channel features like ILD or IPD are concatenated to the encoder output.
* ConvTasnet-parallel.yaml refers to the proposed multiinput-multi-output (MIMO) TasNet in the original paper [1].
* ConvTasnet-parallel-noise.yaml refers to the above Tasnet applied to 2 speakers with DEMAND noise.
* ConvTasnet-parallel-reverb.yaml refers to the above Tasnet applied to 2 speakers with reverberance(RT60) from the [BRIR Sim Set](http://iosr.uk/software/index.php).

The output folders with the checkpoints, logs, etc are available [here](https://www.dropbox.com/sh/i7fhu7qswjb84gw/AABsX1zP-GOTmyl86PtU8GGua?dl=0)

# Example calls for running the training scripts


* Binaural-WSJ02Mix training without dynamic mixing `python train.py hparams/convtasnet-parallel.yaml --data_folder yourpath/binaural-wsj0mix/2speakers --wsj_root yourpath/to/wsj/`

* Binaural-WSJ02Mix training with dynamic mixing `python train.py hparams/convtasnet-parallel.yaml --data_folder yourpath/binaural-wsj0mix/2speakers --wsj_root yourpath/to/wsj/ --dynamic_mixing True`


# Multi-GPU training

You can run the following command to train the model using Distributed Data Parallel (DDP) with 2 GPUs:

```bash
torchrun --nproc_per_node=2 train.py hparams/convtasnet-parallel.yaml --data_folder /yourdatapath
```
You can add the other runtime options as appropriate. For more complete information on multi-GPU usage, take a look at [our documentation](https://speechbrain.readthedocs.io/en/latest/multigpu.html).




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

# References

> [1] Han, Cong, Yi Luo, and Nima Mesgarani. "Real-time binaural speech separation with preserved spatial cues." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
> [2] V. R. Algazi, R. O. Duda, D. M. Thompson and C. Avendano, “The CIPIC HRTF Database,” Proc. 2001 IEEE Workshop on Applications of Signal Processing to Audio and Electroacoustics, pp. 99-102, Mohonk Mountain House, New Paltz, NY, Oct. 21-24, 2001.
