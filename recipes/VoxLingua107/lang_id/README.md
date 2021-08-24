# Language identification experiments using the VoxLingua107 dataset

This folder contains scripts for running language identification experiments with the VoxLingua107 dataset. 
These experiments were highly inspired by the language identification experiments on the CommonLanguage dataset,
as well as speaker recognition experiments with the VoxCeleb dataset. However, due to the large size of
the VoxLingua107 dataset, it has also significant differences, most prominently in using a WebDataset
based data loading pipeline.

The VoxLingua107 dataset contains over 2.5 million utterance-like audio files. Training a model on them
as in the Voxceleb recipe would cause heavy disk loads. Therefore we opted to using WebDataset based training:
before training, audio files are shuffled and distributed into over 500 so-called shards (tar files). During training,
the tar files are opened in random order, the audio files in the shards are shuffled again on-the-fly using a moderately large buffer
and fed to the training process. This reduces the disk load during training by large margin. This is all 
handled by the WebDataset library.

## Downloading the data

You have three options how to download and prepare the VoxLingua107 dataset for training the model:

  - Download the VoxLingua107 language-specific zips from http://bark.phon.ioc.ee/voxlingua107/ and convert them
    to WebDataset format. This is the most flexible option, as it allows selecting a subset of VoxLingua107 languages,
    or adding new languages. It will require around 2.2 TB disk space.
        
  - Download the pre-compiled WebDataset shards from http://bark.phon.ioc.ee/voxlingua107/. It will require around 1.4T of disk space.
  
  - Train directly on the WebDataset shards hosted on the web (i.e., the shards will be downloaded automatically during training). In this
    case it is recomended to cache the shards (configurable in the hyperparameter configuration file).
    
### 1st option: download the VoxLingua107 zips and create the Webdataset shards

Download the zips:
  
```
# Select a place with around 1 TB of free space
cd /data/
mkdir voxlingua107
cd voxlingua107
wget http://bark.phon.ioc.ee/voxlingua107/zip_urls.txt
cat zip_urls.txt | xargs  wget --continue
wget bark.phon.ioc.ee/voxlingua107/dev.zip

```

Create WebDataset shards:

```
python create_wds_shards.py /data/voxlingua107/train/ /data/voxlingua107_shards/train
python create_wds_shards.py /data/voxlingua107/dev/ /data/voxlingua107_shards/dev
```

### 2nd option: download the pre-compiled WebDataset shards

Download the shards:

```
# Select a place with around 1 TB of free space
cd /data/
mkdir voxlingua107_shards
cd voxlingua107_shards
wget  -r -nH --cut-dirs=4 --no-parent --reject="index.html*" http://bark.phon.ioc.ee/lw/korpused/voxlingua107/shards/
```

## 3rd option:

Just set the `shard_cache_dir` property in `hparams/train_epaca_tdnn_wds.yaml` to something, e.g. `/data/voxlingua107_shards`.


## Training

```
python train.py hparams/train_epaca_tdnn_wds.yaml
```

Training is run for 30 epochs. One epoch takes one hour and 40 minutes on a NVidia A100 GPU.


# Performance
| Release | hyperparams file | Dev error rate | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| :-----------:|
| 21-08-24 | train.yaml | 7.3 |https://drive.google.com/drive/folders/1NWIOXpHvC7qUZ16TmNC8oFjfEcjXnKop?usp=sharing | 2xA100 40GB |



# Inference
The pre-trained model + easy inference is available on HuggingFace:
- https://huggingface.co/TalTechNLP/voxlingua107-epaca-tdnn

You can run inference with only few lines of code:

```python
import torchaudio
from speechbrain.pretrained import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")
# Download Thai language sample from Omniglot
signal = language_id.load_audio("https://omniglot.com/soundfiles/udhr/udhr_th.mp3")
prediction =  language_id.classify_batch(signal)
print(prediction)
  (tensor([[0.3210, 0.3751, 0.3680, 0.3939, 0.4026, 0.3644, 0.3689, 0.3597, 0.3508,
           0.3666, 0.3895, 0.3978, 0.3848, 0.3957, 0.3949, 0.3586, 0.4360, 0.3997,
           0.4106, 0.3886, 0.4177, 0.3870, 0.3764, 0.3763, 0.3672, 0.4000, 0.4256,
           0.4091, 0.3563, 0.3695, 0.3320, 0.3838, 0.3850, 0.3867, 0.3878, 0.3944,
           0.3924, 0.4063, 0.3803, 0.3830, 0.2996, 0.4187, 0.3976, 0.3651, 0.3950,
           0.3744, 0.4295, 0.3807, 0.3613, 0.4710, 0.3530, 0.4156, 0.3651, 0.3777,
           0.3813, 0.6063, 0.3708, 0.3886, 0.3766, 0.4023, 0.3785, 0.3612, 0.4193,
           0.3720, 0.4406, 0.3243, 0.3866, 0.3866, 0.4104, 0.4294, 0.4175, 0.3364,
           0.3595, 0.3443, 0.3565, 0.3776, 0.3985, 0.3778, 0.2382, 0.4115, 0.4017,
           0.4070, 0.3266, 0.3648, 0.3888, 0.3907, 0.3755, 0.3631, 0.4460, 0.3464,
           0.3898, 0.3661, 0.3883, 0.3772, 0.9289, 0.3687, 0.4298, 0.4211, 0.3838,
           0.3521, 0.3515, 0.3465, 0.4772, 0.4043, 0.3844, 0.3973, 0.4343]]), tensor([0.9289]), tensor([94]), ['th'])
# The scores in the prediction[0] tensor can be interpreted as cosine scores between
# the languages and the given utterance (i.e., the larger the better)
# The identified language ISO code is given in prediction[3]
print(prediction[3])
  ['th']

# Alternatively, use the utterance embedding extractor:
emb =  language_id.encode_batch(signal)
print(emb.shape)
  torch.Size([1, 1, 256])
```


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

