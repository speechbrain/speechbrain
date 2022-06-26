# DVoice ASR with CTC based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the DVoice dataset(Darija, Swahili): [Link](https://zenodo.org/record/6342622). The dataset used to train the Wolof, Fongbe and Amharic languages can be founded here: [Link](https://github.com/besacier/ALFFA_PUBLIC).

# Data preparation
[DVoice](https://dvoice.ma) attempts to provide automatic voice processing solutions for African languages and dialects. We use preprocessing techniques including voice augmentation to fill the data gap for each language.

# How to run
- Darija : Just download the DVoice dataset than run `python train_with_wav2vec2.py hparams/train_dar_with_wav2vec.yaml --data_folder=/localscratch/darija/`
- Fongbe : Just download the ALFFA_PUBLIC dataset than run `python train_with_wav2vec2.py hparams/train_fon_with_wav2vec.yaml --data_folder=/localscratch/ALFFA_PUBLIC/ASR/FONGBE/data/`
- Amharic : Just download the ALFFA_PUBLIC dataset than run `python train_with_wav2vec2.py hparams/train_amh_with_wav2vec.yaml --data_folder=/localscratch/ALFFA_PUBLIC/ASR/AMHARIC/data/`
- Wolof : Just download the ALFFA_PUBLIC dataset than run `python train_with_wav2vec2.py hparams/train_wol_with_wav2vec.yaml --data_folder=/localscratch/ALFFA_PUBLIC/ASR/WOLOF/data/`
- Swahili : To train the Swahili recipe you need to download the both datasets (DVoice and ALFFA-PUBLIC) then organizing the folders following the hierarchy below and run  `python train_with_wav2vec2.py hparams/train_sw_with_wav2vec.yaml --data_folder=/localscratch/dvoice_recipe_data/`
- Multilingual : To train the Multilingual recipe you need to download both datasets (DVoice and ALFFA-PUBLIC) then organizing the folders following the hierarchy below and run: `python train_with_wav2vec2.py hparams/train_multi_with_wav2vec.yaml --data_folder=/localscratch/dvoice_recipe_data/`

```
      dvoice_recipe_data
      ├── ...
      ├── DVOICE                    # create a DVOICE folder and put the DVoice dataset inside
      │   ├── darija
      │   └── swahili
      ├── ALFFA_DATASET             # put DVOICE and ALFFA_DATASET at the same level
      |   ├── ASR
      |   |   ├── AMHARIC
      |   |   ├── FONGBE
      |   |   ├── HAUSA
      |   |   ├── SWAHILI
      |   |   ├── WOLOF
      |   ├── ...
      └── ...
```

To get the data for Fongbe, Amharic, Wolof and a part of Swahili, please clone https://github.com/besacier/ALFFA_PUBLIC/.

# Languages
Here is a list of the different African languages and dialects that we tested:
- Darija
- Swahili
- Wolof
- Fongbe
- Amharic

# Results
| Language | DVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:|
| Darija(Moroccan Arabic) | v2.0 | train_dar_with_wav2vec.yaml | No | 5.51 | 18.46 | 5.85 | 18.28 | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-darija)
| Swahili | v2.0 | train_sw_with_wav2vec.yaml | No | 8.83 | 22.78 | 9.46 | 23.16 | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-swahili)
| Wolof | v2.0 | train_wol_with_wav2vec.yaml | No | 4.81 | 16.25 | 4.83 | 16.05 | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-wolof)
| Fongbe | v2.0 | train_fon_with_wav2vec.yaml | No | 4.16 | 9.19 | 3.98 | 9.00 | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-fongbe)
| Amharic | v2.0 | train_amh_with_wav2vec.yaml | No | 6.71 | 25.50 | 6.57 | 24.92 | [Link](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-amharic) |

You can find our training results (models, logs, etc) [here](https://drive.google.com/drive/folders/1vNT7RjRuELs7pumBHmfYsrOp9m46D0ym?usp=sharing).

# Performances of DVoice Multilingual on each language
| Dataset Link | Language | Test WER |
|:---------------------------:| -----:| -----:|
| [DVoice](https://zenodo.org/record/6342622) | Darija | 13.27 |
| [DVoice / VoxLingua107](https://zenodo.org/record/6342622) + [ALFFA](https://github.com/besacier/ALFFA_PUBLIC/tree/master/ASR/SWAHILI) | Swahili | 29.31
| [ALFFA](https://github.com/besacier/ALFFA_PUBLIC/tree/master/ASR/FONGBE) | Fongbe | 10.26
| [ALFFA](https://github.com/besacier/ALFFA_PUBLIC/tree/master/ASR/WOLOF) | Wolof | 21.54
| [ALFFA](https://github.com/besacier/ALFFA_PUBLIC/tree/master/ASR/AMHARIC) | Amharic | 31.15

# How to simply use pretrained models to transcribe my audio file?
SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for DVoice.

# **About DVoice**
DVoice is a community initiative that aims to provide Africa low resources languages with data and models to facilitate their use of voice technologies. The lack of data on these languages makes it necessary to collect data using methods that are specific to each one. The DVoice platform([https://dvoice.ma](https://dvoice.ma) is based on Mozilla Common Voice, for collecting authentic recordings from the community, and transfer learning techniques for automatically labeling recordings that are retrived from social medias. The DVoice platform currently manages 7 languages including Darija(Moroccan Arabic dialect) whose dataset appears on this version, Wolof, Mandingo, Serere, Pular, Diola and Soninke.

For this project, AIOX Labs the SI2M Laboratory are joining forces to build the future of technologies together.

# **About AIOX Labs**
Based in Rabat, London and Paris, AIOX - Labs mobilizes artificial intelligence technologies to meet the business needs and data projects of companies.

- He is at the service of the growth of groups, the optimization of processes or the improvement of the customer experience.
- AIOX - Labs is multi - sector, from fintech to industry, including retail and consumer goods.
- Business ready data products with a solid algorithmic base and adaptability for the specific needs of each client.
- A complementary team made up of doctors in AI and business experts with a solid scientific base and international publications.

Website: [https://www.aiox-labs.com/ ](https://www.aiox-labs.com/)

# **About SI2M Laboratory**
The Information Systems, Intelligent Systems and Mathematical Modeling Research Laboratory(SI2M) is an academic research laboratory of the National Institute of Statistics and Applied Economics(INSEA). The research areas of the laboratories are Information Systems, Intelligent Systems, Artificial Intelligence, Decision Support, Network and System Security, Mathematical Modelling.

Website: [SI2M Laboratory](https://insea.ac.ma/index.php/pole-recherche/equipe-de-recherche/150-laboratoire-de-recherche-en-systemes-d-information-systemes-intelligents-et-modelisation-mathematique)


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex


@misc{speechbrain,
      title = {{SpeechBrain}: A General - Purpose Speech Toolkit},
      author = {Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju - Chieh Chou and Sung - Lin Yeh and Szu - Wei Fu and Chien - Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
      year = {2021},
      eprint = {2106.04624},
      archivePrefix = {arXiv},
      primaryClass = {eess.AS},
      note = {arXiv: 2106.04624}
      }
```
# **Acknowledgements**
This research was supported through computational resources of HPC - MARWAN(www.marwan.ma/hpc) provided by CNRST, Rabat, Morocco. We deeply thank this institution.
