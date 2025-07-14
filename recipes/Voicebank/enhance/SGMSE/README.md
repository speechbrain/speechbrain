# VoiceBank Speech Enhancement with SGMSE
This recipe implements a speech enhancement system based on the SGMSE architecture.
with the VoiceBank dataset (based on the paper: https://arxiv.org/abs/2208.05830).

# How to run
cd recipes/Voicebank/enhance/SGMSE
python train.py hparams.yaml

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# ***Citing SGMSE**

```bibtex
@article{richter2023speech,
  title={Speech enhancement and dereverberation with diffusion-based generative models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351--2364},
  year={2023},
  publisher={IEEE}
}
```