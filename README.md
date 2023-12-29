<p align="center">
  <img src="https://raw.githubusercontent.com/speechbrain/speechbrain/develop/docs/images/speechbrain-logo.svg" alt="SpeechBrain Logo"/>
</p>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=40&duration=7000&pause=1000&random=false&width=1200&height=100&lines=Simplify+Conversational+AI+Development)](https://git.io/typing-svg)


| 📘 [Tutorials](https://speechbrain.github.io/tutorial_basics.html) | 🌐 [Website](https://speechbrain.github.io/) | 📚 [Documentation](https://speechbrain.readthedocs.io/en/latest/index.html) | 🤝 [Contributing](https://speechbrain.readthedocs.io/en/latest/contributing.html) | 🤗 [HuggingFace](https://huggingface.co/speechbrain) | ▶️ [YouTube](https://www.youtube.com/@SpeechBrainProject) | 🐦 [Twitter](https://twitter.com/SpeechBrain1) |

![GitHub Repo stars](https://img.shields.io/github/stars/speechbrain/speechbrain?style=social) *Please, help our community project. Star on GitHub!*


#
# 🗣️💬 What SpeechBrain Offers

- SpeechBrain is an **open-source** [PyTorch](https://pytorch.org/) toolkit that accelerates **Conversational AI** development, i.e., the technology behind *speech assistants*, *chatbots*, and *large language models*.

- It is crafted for fast and easy creation of advanced technologies for **Speech** and **Text** Processing.


## 🌐  Vision
- With the rise of [deep learning](https://www.deeplearningbook.org/), once-distant domains like speech processing and NLP are now very close. A well-designed neural network and large datasets are all you need.

- We think it is now time for a **holistic toolkit** that, mimicking the human brain, jointly supports diverse technologies for complex Conversational AI systems.

- This spans *speech recognition*, *speaker recognition*, *speech enhancement*, *speech separation*, *language modeling*, *dialogue*, and beyond.



## 📚 Training Recipes
- We share over 200 competitive training [recipes](https://github.com/speechbrain/speechbrain/tree/develop/recipes) on more than 40 datasets supporting 20 speech and text processing tasks (see below).

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
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
asr_model.transcribe_file("speechbrain/asr-conformer-transformerlm-librispeech/example.wav")
```

##  <a href="https://speechbrain.github.io/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/1200px-Google_Colaboratory_SVG_Logo.svg.png" alt="drawing" width="50"/> </a>  Documentation
- We are deeply dedicated to promoting inclusivity and education.
- We have authored over 30 [tutorials](https://speechbrain.github.io/) on Google Colab that not only describe how SpeechBrain works but also help users familiarize themselves with Conversational AI.
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

- **Tutorials:** Start with [basic tutorials](https://speechbrain.github.io/tutorial_basics.html) covering fundamental functionalities. Find advanced tutorials and topics in the Tutorials menu on the [SpeechBrain website](https://speechbrain.github.io).

- **Documentation:** Detailed information on the SpeechBrain API, contribution guidelines, and code is available in the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html).

#
# 🔧 Supported Technologies
- SpeechBrain is a versatile framework designed for implementing a wide range of technologies within the field of Conversational AI.
- It excels not only in individual task implementations but also in combining various technologies into complex pipelines.

## 🎙️ Speech/Audio Processing
| Tasks        | Datasets           | Technologies/Models  |
| ------------- |-------------| -----|
| Speech Recognition      | [AISHELL-1](https://github.com/speechbrain/speechbrain/tree/develop/recipes/AISHELL-1), [CommonVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice), [DVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/DVoice), [KsponSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/KsponSpeech), [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech), [MEDIA](https://github.com/speechbrain/speechbrain/tree/develop/recipes/MEDIA), [RescueSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/RescueSpeech), [Switchboard](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Switchboard), [TIMIT](https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT), [Tedlium2](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Tedlium2), [Voicebank](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank) | [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf), [Tranducers](https://arxiv.org/pdf/1211.3711.pdf?origin=publication_detail), [Tranformers](https://arxiv.org/abs/1706.03762), Seq2Seq, Beamsearch, Rescoring, [Conformer](https://arxiv.org/abs/2005.08100),Streamable Conformer, [Branchformer](https://arxiv.org/abs/2207.02971), [Hyperconformer](https://arxiv.org/abs/2305.18281), [Kaldi2-FST](https://github.com/k2-fsa/k2) |
| Speaker Recognition      | [VoxCeleb](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb) | [ECAPA-TDNN](https://arxiv.org/abs/2005.07143), [ResNET](https://arxiv.org/pdf/1910.12592.pdf), [Xvectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf), [PLDA](https://ieeexplore.ieee.org/document/6639151), [Score Normalization](https://www.sciencedirect.com/science/article/abs/pii/S1051200499903603) |
| Speech Separation      | [WSJ0Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WSJ0Mix), [LibriMix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriMix), [WHAM!](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WHAMandWHAMR), [WHAMR!](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WHAMandWHAMR), [Aishell1Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Aishell1Mix), [BinauralWSJ0Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/BinauralWSJ0Mix) | [SepFormer](https://arxiv.org/abs/2010.13154), [RESepFormer](https://arxiv.org/abs/2206.09507), [SkiM](https://arxiv.org/abs/2201.10800), [DualPath RNN](https://arxiv.org/abs/1910.06379), [ConvTasNET](https://arxiv.org/abs/1809.07454) |
| Speech Enhancement      | [DNS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/DNS), [Voicebank](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank) | SepFormer, MetricGAN, MetricGAN-U, SEGAN, spectral masking, time masking |
| Text-to-Speech      | [LJSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech), [LibriTTS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriTTS) | Tacotron2, Multi-Speaker Tacotron2, FastSpeech2 |
| Vocoding      | [LJSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech), [LibriTTS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriTTS) | HiFiGAN, DiffWave |
| Spoken Language Understanding | [MEDIA](https://github.com/speechbrain/speechbrain/tree/develop/recipes/MEDIA), [SLURP](https://github.com/speechbrain/speechbrain/tree/develop/recipes/SLURP), [Fluent Speech Commands](https://github.com/speechbrain/speechbrain/tree/develop/recipes/fluent-speech-commands), [Timers-and-Such](https://github.com/speechbrain/speechbrain/tree/develop/recipes/timers-and-such)  | Direct SLU, Decoupled SLU, Multistage SLU |
| Speech-to-Speech Translation  | [CVSS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CVSS) | Discrete Hubert, HifiGAN, wav2vec2 |
| Speech Translation  | [Fisher CallHome (Spanish)](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Fisher-Callhome-Spanish), [IWSLT22(lowresource)](https://github.com/speechbrain/speechbrain/tree/develop/recipes/IWSLT22_lowresource) | wav2vec2 |
| Emotion Classification      | [IEMOCAP](https://github.com/speechbrain/speechbrain/tree/develop/recipes/IEMOCAP), [ZaionEmotionDataset](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ZaionEmotionDataset) | ECAPA-TDNN, Wav2vec2, Emotion Diarization |
| Language Identification | [VoxLingua107](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107), [CommonLanguage](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonLanguage)| ECAPA-TDNN |
| Voice Activity Detection  | [LibriParty](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriParty) | CRDNN |
| Sound Classification  | [ESC50](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ESC50), [UrbanSound](https://github.com/speechbrain/speechbrain/tree/develop/recipes/UrbanSound8k) | CNN, CNN14, ECAPA-TDNN |
| Self-Supervised Learning | [CommonVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice), [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech) | Wav2vec2 |
| Interpretabiliy | [ESC50](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ESC50) | Learning-to-Interpret (L2I), Non-Negative Matrix Factorization (NMF), PIQ |
| Speech Generation | [AudioMNIST](https://github.com/speechbrain/speechbrain/tree/develop/recipes/AudioMNIST) | Diffusion, Latent Diffusion |
| Metric Learning | [REAL-M](https://github.com/speechbrain/speechbrain/tree/develop/recipes/REAL-M/sisnr-estimation), [Voicebank](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank) | Blind SNR-Estimation, PESQ Learning |
| Allignment | [TIMIT](https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT) | CTC, Viterbi, Forward Forward |
| Diarization | [AMI](https://github.com/speechbrain/speechbrain/tree/develop/recipes/AMI) | ECAPA-TDNN, X-vectors, Spectral Clustering |

## 📝 Text Processing
| Tasks        | Datasets           | Technologies  |
| ------------- |-------------| -----|
| Language Modeling | [CommonVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice), [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/unstable-v0.6/recipes/LibriSpeech)| n-grams, RNNLM, TransformerLM |
| Response Generation | [MultiWOZ](https://github.com/speechbrain/speechbrain/tree/unstable-v0.6/recipes/MultiWOZ/response_generation)| GPT2, Llama2 |
| Grapheme-to-Phoneme | [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech) | RNN, Transformer, Curriculum Learning, Homograph loss |

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

- For a comprehensive overview of the achieved performance across different tasks, datasets, and technologies, please visit [here](https://github.com/speechbrain/speechbrain/blob/develop/PERFORMANCE.md).

#
# 📜 License

- SpeechBrain is released under the [Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0), a popular BSD-like license.
- You are free to redistribute SpeechBrain for both free and commercial purposes, with the condition of retaining license headers. Unlike the GPL, the Apache License is not viral, meaning you are not obligated to release modifications to the source code.

#
# 🔮Future Plans

We have ambitious plans for the future, with a focus on the following priorities:

- **Scale Up:** Our aim is to provide comprehensive recipes and technologies for training massive models on extensive datasets.

- **Scale Down:** While scaling up delivers unprecedented performance, we recognize the challenges of deploying large models in production scenarios. We are focusing on real-time, streamable, and small-footprint Conversational AI.

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


