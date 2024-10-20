<p align="center">
  <img src="https://raw.githubusercontent.com/speechbrain/speechbrain/develop/docs/images/speechbrain-logo.svg" alt="SpeechBrain Logo"/>
</p>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=40&duration=7000&pause=1000&random=false&width=1200&height=100&lines=Simplify+Conversational+AI+Development)](https://git.io/typing-svg)


| 📘 [Tutorials](https://speechbrain.readthedocs.io) | 🌐 [Website](https://speechbrain.github.io/) | 📚 [Documentation](https://speechbrain.readthedocs.io/en/latest/index.html) | 🤝 [Contributing](https://speechbrain.readthedocs.io/en/latest/contributing.html) | 🤗 [HuggingFace](https://huggingface.co/speechbrain) | ▶️ [YouTube](https://www.youtube.com/@SpeechBrainProject) | 🐦 [X](https://twitter.com/SpeechBrain1) |

![GitHub Repo stars](https://img.shields.io/github/stars/speechbrain/speechbrain?style=social) *Please, help our community project. Star on GitHub!*

**Exciting News (January, 2024):** Discover what is new in SpeechBrain 1.0 [here](https://colab.research.google.com/drive/1IEPfKRuvJRSjoxu22GZhb3czfVHsAy0s?usp=sharing)!
#
# 🗣️💬 What SpeechBrain Offers

- SpeechBrain is an **open-source** [PyTorch](https://pytorch.org/) toolkit that accelerates **Conversational AI** development, i.e., the technology behind *speech assistants*, *chatbots*, and *large language models*.

- It is crafted for fast and easy creation of advanced technologies for **Speech** and **Text** Processing.


## 🌐  Vision
- With the rise of [deep learning](https://www.deeplearningbook.org/), once-distant domains like speech processing and NLP are now very close. A well-designed neural network and large datasets are all you need.

- We think it is now time for a **holistic toolkit** that, mimicking the human brain, jointly supports diverse technologies for complex Conversational AI systems.

- This spans *speech recognition*, *speaker recognition*, *speech enhancement*, *speech separation*, *language modeling*, *dialogue*, and beyond.

- Aligned with our long-term goal of natural human-machine conversation, including for non-verbal individuals, we have recently added support for the [EEG modality](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB).



## 📚 Training Recipes
- We share over 200 competitive training [recipes](recipes) on more than 40 datasets supporting 20 speech and text processing tasks (see below).

- We support both training from scratch and fine-tuning pretrained models such as [Whisper](https://huggingface.co/openai/whisper-large), [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2), [WavLM](https://huggingface.co/docs/transformers/model_doc/wavlm), [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert), [GPT2](https://huggingface.co/gpt2), [Llama2](https://huggingface.co/docs/transformers/model_doc/llama2), and beyond. The models on [HuggingFace](https://huggingface.co/) can be easily plugged in and fine-tuned.

- For any task, you train the model using these commands:
```python
python train.py hparams/train.yaml
```

- The hyperparameters are encapsulated in a YAML file, while the training process is orchestrated through a Python script.

- We maintained a consistent code structure across different tasks.

- For better replicability, training logs and checkpoints are hosted on Dropbox.

## <a href="https://huggingface.co/speechbrain" target="_blank"> <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> </a> Pretrained Models and Inference

- Access over 100 pretrained models hosted on [HuggingFace](https://huggingface.co/speechbrain).
- Each model comes with a user-friendly interface for seamless inference. For example, transcribing speech using a pretrained model requires just three lines of code:

```python
from speechbrain.inference import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
asr_model.transcribe_file("speechbrain/asr-conformer-transformerlm-librispeech/example.wav")
```

##  <a href="https://speechbrain.github.io/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/1200px-Google_Colaboratory_SVG_Logo.svg.png" alt="drawing" width="50"/> </a>  Documentation
- We are deeply dedicated to promoting inclusivity and education.
- We have authored over 30 [tutorials](https://speechbrain.readthedocs.io) that not only describe how SpeechBrain works but also help users familiarize themselves with Conversational AI.
- Every class or function has clear explanations and examples that you can run. Check out the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html) for more details 📚.



## 🎯 Use Cases
- 🚀 **Research Acceleration**: Speeding up academic and industrial research. You can develop and integrate new models effortlessly, comparing their performance against our baselines.

- ⚡️ **Rapid Prototyping**: Ideal for quick prototyping in time-sensitive projects.

- 🎓 **Educational Tool**: SpeechBrain's simplicity makes it a valuable educational resource. It is used by institutions like [Mila](https://mila.quebec/en/), [Concordia University](https://www.concordia.ca/), [Avignon University](https://univ-avignon.fr/en/), and many others for student training.

#
# 🚀 Quick Start

To get started with SpeechBrain, follow these simple steps:

## 🛠️ Installation

### Install via PyPI

1. Install SpeechBrain using PyPI:

    ```bash
    pip install speechbrain
    ```

2. Access SpeechBrain in your Python code:

    ```python
    import speechbrain as sb
    ```

### Install from GitHub
This installation is recommended for users who wish to conduct experiments and customize the toolkit according to their needs.

1. Clone the GitHub repository and install the requirements:

    ```bash
    git clone https://github.com/speechbrain/speechbrain.git
    cd speechbrain
    pip install -r requirements.txt
    pip install --editable .
    ```

2. Access SpeechBrain in your Python code:

    ```python
    import speechbrain as sb
    ```

Any modifications made to the `speechbrain` package will be automatically reflected, thanks to the `--editable` flag.

## ✔️ Test Installation

Ensure your installation is correct by running the following commands:

```bash
pytest tests
pytest --doctest-modules speechbrain
```

## 🏃‍♂️ Running an Experiment

In SpeechBrain, you can train a model for any task using the following steps:

```python
cd recipes/<dataset>/<task>/
python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the YAML file.

## 📘 Learning SpeechBrain

- **Website:** Explore general information on the [official website](https://speechbrain.github.io).

- **Tutorials:** Start with [basic tutorials](https://speechbrain.readthedocs.io/en/latest/tutorials/basics.html) covering fundamental functionalities. Find advanced tutorials and topics in the Tutorial notebooks category in the [SpeechBrain documentation](https://speechbrain.readthedocs.io).

- **Documentation:** Detailed information on the SpeechBrain API, contribution guidelines, and code is available in the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html).

#
# 🔧 Supported Technologies
- SpeechBrain is a versatile framework designed for implementing a wide range of technologies within the field of Conversational AI.
- It excels not only in individual task implementations but also in combining various technologies into complex pipelines.

## 🎙️ Speech/Audio Processing
| Tasks        | Datasets           | Technologies/Models  |
| ------------- |-------------| -----|
| Speech Recognition      | [AISHELL-1](recipes/AISHELL-1), [CommonVoice](recipes/CommonVoice), [DVoice](recipes/DVoice), [KsponSpeech](recipes/KsponSpeech), [LibriSpeech](recipes/LibriSpeech), [MEDIA](recipes/MEDIA), [RescueSpeech](recipes/RescueSpeech), [Switchboard](recipes/Switchboard), [TIMIT](recipes/TIMIT), [Tedlium2](recipes/Tedlium2), [Voicebank](recipes/Voicebank) | [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf), [Transducers](https://arxiv.org/pdf/1211.3711.pdf?origin=publication_detail), [Transformers](https://arxiv.org/abs/1706.03762), [Seq2Seq](http://zhaoshuaijiang.com/file/Hybrid_CTC_Attention_Architecture_for_End-to-End_Speech_Recognition.pdf), [Beamsearch techniques for CTC](https://arxiv.org/pdf/1911.01629.pdf),[seq2seq](https://arxiv.org/abs/1904.02619.pdf),[transducers](https://www.merl.com/publications/docs/TR2017-190.pdf)), [Rescoring](https://arxiv.org/pdf/1612.02695.pdf), [Conformer](https://arxiv.org/abs/2005.08100), [Branchformer](https://arxiv.org/abs/2207.02971), [Hyperconformer](https://arxiv.org/abs/2305.18281), [Kaldi2-FST](https://github.com/k2-fsa/k2) |
| Speaker Recognition      | [VoxCeleb](recipes/VoxCeleb) | [ECAPA-TDNN](https://arxiv.org/abs/2005.07143), [ResNET](https://arxiv.org/pdf/1910.12592.pdf), [Xvectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf), [PLDA](https://ieeexplore.ieee.org/document/6639151), [Score Normalization](https://www.sciencedirect.com/science/article/abs/pii/S1051200499903603) |
| Speech Separation      | [WSJ0Mix](recipes/WSJ0Mix), [LibriMix](recipes/LibriMix), [WHAM!](recipes/WHAMandWHAMR), [WHAMR!](recipes/WHAMandWHAMR), [Aishell1Mix](recipes/Aishell1Mix), [BinauralWSJ0Mix](recipes/BinauralWSJ0Mix) | [SepFormer](https://arxiv.org/abs/2010.13154), [RESepFormer](https://arxiv.org/abs/2206.09507), [SkiM](https://arxiv.org/abs/2201.10800), [DualPath RNN](https://arxiv.org/abs/1910.06379), [ConvTasNET](https://arxiv.org/abs/1809.07454) |
| Speech Enhancement      | [DNS](recipes/DNS), [Voicebank](recipes/Voicebank) | [SepFormer](https://arxiv.org/abs/2010.13154), [MetricGAN](https://arxiv.org/abs/1905.04874), [MetricGAN-U](https://arxiv.org/abs/2110.05866), [SEGAN](https://arxiv.org/abs/1703.09452), [spectral masking](http://staff.ustc.edu.cn/~jundu/Publications/publications/Trans2015_Xu.pdf), [time masking](http://staff.ustc.edu.cn/~jundu/Publications/publications/Trans2015_Xu.pdf) |
| Interpretability | [ESC50](recipes/ESC50) | [Listenable Maps for Audio Classifiers (L-MAC)](https://arxiv.org/abs/2403.13086), [Learning-to-Interpret (L2I)](https://proceedings.neurips.cc/paper_files/paper/2022/file/e53280d73dd5389e820f4a6250365b0e-Paper-Conference.pdf), [Non-Negative Matrix Factorization (NMF)](https://proceedings.neurips.cc/paper_files/paper/2022/file/e53280d73dd5389e820f4a6250365b0e-Paper-Conference.pdf), [PIQ](https://arxiv.org/abs/2303.12659) |
| Speech Generation | [AudioMNIST](recipes/AudioMNIST) | [Diffusion](https://arxiv.org/abs/2006.11239), [Latent Diffusion](https://arxiv.org/abs/2112.10752) |
| Text-to-Speech      | [LJSpeech](recipes/LJSpeech), [LibriTTS](recipes/LibriTTS) | [Tacotron2](https://arxiv.org/abs/1712.05884), [Zero-Shot Multi-Speaker Tacotron2](https://arxiv.org/abs/2112.02418), [FastSpeech2](https://arxiv.org/abs/2006.04558) |
| Vocoding      | [LJSpeech](recipes/LJSpeech), [LibriTTS](recipes/LibriTTS) | [HiFiGAN](https://arxiv.org/abs/2010.05646), [DiffWave](https://arxiv.org/abs/2009.09761)
| Spoken Language Understanding | [MEDIA](recipes/MEDIA), [SLURP](recipes/SLURP), [Fluent Speech Commands](recipes/fluent-speech-commands), [Timers-and-Such](recipes/timers-and-such)  | [Direct SLU](https://arxiv.org/abs/2104.01604), [Decoupled SLU](https://arxiv.org/abs/2104.01604), [Multistage SLU](https://arxiv.org/abs/2104.01604) |
| Speech-to-Speech Translation  | [CVSS](recipes/CVSS) | [Discrete Hubert](https://arxiv.org/pdf/2106.07447.pdf), [HiFiGAN](https://arxiv.org/abs/2010.05646), [wav2vec2](https://arxiv.org/abs/2006.11477) |
| Speech Translation  | [Fisher CallHome (Spanish)](recipes/Fisher-Callhome-Spanish), [IWSLT22(lowresource)](recipes/IWSLT22_lowresource) | [wav2vec2](https://arxiv.org/abs/2006.11477) |
| Emotion Classification      | [IEMOCAP](recipes/IEMOCAP), [ZaionEmotionDataset](recipes/ZaionEmotionDataset) | [ECAPA-TDNN](https://arxiv.org/abs/2005.07143), [wav2vec2](https://arxiv.org/abs/2006.11477), [Emotion Diarization](https://arxiv.org/abs/2306.12991) |
| Language Identification | [VoxLingua107](recipes/VoxLingua107), [CommonLanguage](recipes/CommonLanguage)| [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) |
| Voice Activity Detection  | [LibriParty](recipes/LibriParty) | [CRDNN](https://arxiv.org/abs/2106.04624) |
| Sound Classification  | [ESC50](recipes/ESC50), [UrbanSound](recipes/UrbanSound8k) | [CNN14](https://github.com/ranchlai/sound_classification), [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) |
| Self-Supervised Learning | [CommonVoice](recipes/CommonVoice), [LibriSpeech](recipes/LibriSpeech) | [wav2vec2](https://arxiv.org/abs/2006.11477) |
| Metric Learning | [REAL-M](recipes/REAL-M/sisnr-estimation), [Voicebank](recipes/Voicebank) | [Blind SNR-Estimation](https://arxiv.org/abs/2002.08909), [PESQ Learning](https://arxiv.org/abs/2110.05866) |
| Alignment | [TIMIT](recipes/TIMIT) | [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf), [Viterbi](https://www.cs.cmu.edu/~cga/behavior/rabiner1.pdf), [Forward Forward](https://www.cs.cmu.edu/~cga/behavior/rabiner1.pdf) |
| Diarization | [AMI](recipes/AMI) | [ECAPA-TDNN](https://arxiv.org/abs/2005.07143), [X-vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf), [Spectral Clustering](https://web.archive.org/web/20240305184559/http://www.ifp.illinois.edu/~hning2/papers/Ning_spectral.pdf) |

## 📝 Text Processing
| Tasks        | Datasets           | Technologies/Models  |
| ------------- |-------------| -----|
| Language Modeling | [CommonVoice](recipes/CommonVoice), [LibriSpeech](recipes/LibriSpeech)| [n-grams](https://web.stanford.edu/~jurafsky/slp3/3.pdf), [RNNLM](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf), [TransformerLM](https://arxiv.org/abs/1706.03762) |
| Response Generation | [MultiWOZ](recipes/MultiWOZ/response_generation)| [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Llama2](https://arxiv.org/abs/2307.09288) |
| Grapheme-to-Phoneme | [LibriSpeech](recipes/LibriSpeech) | [RNN](https://arxiv.org/abs/2207.13703), [Transformer](https://arxiv.org/abs/2207.13703), [Curriculum Learning](https://arxiv.org/abs/2207.13703), [Homograph loss](https://arxiv.org/abs/2207.13703) |

## 🧠 EEG Processing
| Tasks        | Datasets           | Technologies/Models  |
| ------------- |-------------| -----|
| Motor Imagery | [BNCI2014001](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/MotorImagery/BNCI2014001), [BNCI2014004](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/MotorImagery/BNCI2014004), [BNCI2015001](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/MotorImagery/BNCI2015001), [Lee2019_MI](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/MotorImagery/Lee2019_MI), [Zhou201](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/MotorImagery/Zhou2016) | [EEGNet](https://github.com/speechbrain/benchmarks/blob/main/benchmarks/MOABB/models/EEGNet.py), [ShallowConvNet](https://github.com/speechbrain/benchmarks/blob/main/benchmarks/MOABB/models/ShallowConvNet.py), [EEGConformer](https://github.com/speechbrain/benchmarks/blob/main/benchmarks/MOABB/models/EEGConformer.py) |
| P300 | [BNCI2014009](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/P300/BNCI2014009), [EPFLP300](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/P300/EPFLP300), [bi2015a](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/P300/bi2015a), | [EEGNet](https://github.com/speechbrain/benchmarks/blob/main/benchmarks/MOABB/models/EEGNet.py) |
| SSVEP | [Lee2019_SSVEP](https://github.com/speechbrain/benchmarks/tree/main/benchmarks/MOABB/hparams/SSVEP/Lee2019_SSVEP) | [EEGNet](https://github.com/speechbrain/benchmarks/blob/main/benchmarks/MOABB/models/EEGNet.py) |




## 🔍 Additional Features

SpeechBrain includes a range of native functionalities that enhance the development of Conversational AI technologies. Here are some examples:

- **Training Orchestration:** The `Brain` class serves as a fully customizable tool for managing training and evaluation loops over data. It simplifies training loops while providing the flexibility to override any part of the process.

- **Hyperparameter Management:** A YAML-based hyperparameter file specifies all hyperparameters, from individual numbers (e.g., learning rate) to complete objects (e.g., custom models). This elegant solution drastically simplifies the training script.

- **Dynamic Dataloader:** Enables flexible and efficient data reading.

- **GPU Training:** Supports single and multi-GPU training, including distributed training.

- **Dynamic Batching:** On-the-fly dynamic batching enhances the efficient processing of variable-length signals.

- **Mixed-Precision Training:** Accelerates training through mixed-precision techniques.

- **Efficient Data Reading:** Reads large datasets efficiently from a shared Network File System (NFS) via [WebDataset](https://github.com/webdataset/webdataset).

- **Hugging Face Integration:** Interfaces seamlessly with [HuggingFace](https://huggingface.co/speechbrain) for popular models such as wav2vec2 and Hubert.

- **Orion Integration:** Interfaces with [Orion](https://github.com/Epistimio/orion) for hyperparameter tuning.

- **Speech Augmentation Techniques:** Includes SpecAugment, Noise, Reverberation, and more.

- **Data Preparation Scripts:** Includes scripts for preparing data for supported datasets.

SpeechBrain is rapidly evolving, with ongoing efforts to support a growing array of technologies in the future.


## 📊 Performance

- SpeechBrain integrates a variety of technologies, including those that achieves competitive or state-of-the-art performance.

- For a comprehensive overview of the achieved performance across different tasks, datasets, and technologies, please visit [here](PERFORMANCE.md).

#
# 📜 License

- SpeechBrain is released under the [Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0), a popular BSD-like license.
- You are free to redistribute SpeechBrain for both free and commercial purposes, with the condition of retaining license headers. Unlike the GPL, the Apache License is not viral, meaning you are not obligated to release modifications to the source code.

#
# 🔮Future Plans

We have ambitious plans for the future, with a focus on the following priorities:

- **Scale Up:** We aim to provide comprehensive recipes and technologies for training massive models on extensive datasets.

- **Scale Down:** While scaling up delivers unprecedented performance, we recognize the challenges of deploying large models in production scenarios. We are focusing on real-time, streamable, and small-footprint Conversational AI.

- **Multimodal Large Language Models**: We envision a future where a single foundation model can handle a wide range of text, speech, and audio tasks. Our core team is focused on enabling the training of advanced multimodal LLMs.

#
# 🤝 Contributing

- SpeechBrain is a community-driven project, led by a core team with the support of numerous international collaborators.
- We welcome contributions and ideas from the community. For more information, check [here](https://speechbrain.github.io/contributing.html).

#
# 🙏 Sponsors

- SpeechBrain is an academically driven project and relies on the passion and enthusiasm of its contributors.
- As we cannot rely on the resources of a large company, we deeply appreciate any form of support, including donations or collaboration with the core team.
- If you're interested in sponsoring SpeechBrain, please reach out to us at speechbrainproject@gmail.com.
- A heartfelt thank you to all our sponsors, including the current ones:



[<img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Image 1" width="250"/>](https://speechbrain.github.io/img/hf.ico) &nbsp; &nbsp;
[<img src="https://speechbrain.github.io/img/sponsors/logo_vd.png" alt="Image 3" width="250"/>](https://viadialog.com/en/) &nbsp; &nbsp;
[<img src="https://speechbrain.github.io/img/sponsors/logo_nle.png" alt="Image 4" width="250"/>](https://europe.naverlabs.com/)

<br><br>

[<img src="https://speechbrain.github.io/img/sponsors/logo_ovh.png" alt="Image 5" width="250"/>](https://www.ovhcloud.com/en-ca/) &nbsp; &nbsp;
[<img src="https://speechbrain.github.io/img/sponsors/logo_badu.png" alt="Image 2" width="250"/>](https://usa.baidu.com/) &nbsp; &nbsp;
[<img src="https://speechbrain.github.io/img/sponsors/samsung_official.png" alt="Image 6" width="250"/>](https://research.samsung.com/aicenter_cambridge)

<br><br>

[<img src="https://speechbrain.github.io/img/sponsors/logo_mila_small.png" alt="Image 7" width="250"/>](https://mila.quebec/en/) &nbsp; &nbsp;
[<img src="https://www.concordia.ca/content/dam/common/logos/Concordia-logo.jpeg" alt="Image 9" width="250"/>](https://www.concordia.ca/) &nbsp; &nbsp;
[<img src="https://speechbrain.github.io/img/partners/logo_lia.png" alt="Image 8" width="250"/>](https://lia.univ-avignon.fr/) &nbsp; &nbsp;
#
# 📖 Citing SpeechBrain

If you use SpeechBrain in your research or business, please cite it using the following BibTeX entry:

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with {SpeechBrain} 1.0},
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

