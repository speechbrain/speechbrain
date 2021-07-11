# VoiceBank Speech Enhancement with SEGAN
This recipe implements a speech enhancement system based on the SEGAN architecture
with the VoiceBank dataset.
(based on the paper: Pascual et al. https://arxiv.org/pdf/1703.09452.pdf)

Performance: PESQ = 2.48, STOI: 0.928


# How to run
python train.py hparams/train.yaml
