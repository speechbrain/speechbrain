# Speaker recognition experiments with VoxCeleb.
This folder contains scripts for running speaker identification and verification experiments with the VoxCeleb dataset(http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

# Training Xvectors
Run the following command to train xvectors:

`python train_speaker_embeddings.py hyperparams/train_x_vectors.yaml`

You can use the same script for voxceleb1, voxceleb2, and voxceleb1+2. Just change the datafolder and the corresponding number or speakers (1211 vox1, 5994 vox2, 7205 vox1+2).
For voxceleb1 + voxceleb2, see preparation instructions below).

It is possible to train embeddings with more augmentation with the following command:

`python train_speaker_embeddings.py hyperparams/train_ecapa_tdnn_big.yaml`

It this case, we concatenate waveform dropout, speed change, reverberarion, noise, and noise+rev. The batch is 6 times larger that the original one. This normally leads to
a performance improvement at the cost of longer training time.

The system trains a TDNN for speaker embeddings coupled with a speaker-id classifier. The speaker-id accuracy should be around 97-98% for both voxceleb1 and voceleb2.

# Speaker verification with PLDA
After training the speaker embeddings, it is possible to perform speaker verification using PLDA.  You can run it with the following command:

`python speaker_verification_plda.py hyperparams/verification_plda_xvector.yaml`

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER = 5.8 % on voxceleb1, EER = 4.7 % on voxceleb2, and EER = 4.3 % on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)

# Speaker verification with contrastive learning
SpeechBrain supports speaker verification using contrastive learning.
We employ a pre-trained encoder followed by a binary discriminator. The discriminator is fed with either positive or negative embeddings that are properly sampled from the dataset.  To run this experiment, type the following command:

`python speaker_verification_discriminator.py hyperparams/verfication_discriminator_xvector.yaml`

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER = 3.8 % on voxceleb1, EER = 3.1 % on voxceleb2, and EER = 2.9 % on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)

# Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py hyperparams/train_ecapa_tdnn.yaml`


The speaker-id accuracy should be around 98-99% for both voxceleb1 and voceleb2.

After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity.  You can run it with the following command:

`python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml`

This system achieves an EER = 3.4 % on voxceleb1, EER = 1.6 % on voxceleb2, and EER = 1.4 % on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)

# VoxCeleb2 preparation
Voxceleb2 audio files are released in ma4 format. All the files must be converted in wav files before
feeding them is SpeechBrain. Please, follow these steps to prepare the dataset correctly:

1. Download both Voxceleb1 and Voxceleb2.
You can find download instructions here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
Note that for the speaker verification experiments with Voxceleb2 the official split of voxceleb1 is used to compute EER.

2. Convert .ma4 to wav
Voxceleb2 stores files with the ma4 audio format. To use them within SpeechBrain you have to convert all the ma4 files into wav files.
You can do the conversion using ffmpeg(see for instance conversion scripts in https://gitmemory.com/issue/pytorch/audio/104/493137979 or https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830). This operation might take several hours and should be only once.

2. Put all the wav files in a folder called wav. You should have something like `voxceleb2/wav/id*/*.wav` (e.g, `voxceleb2/wav/id00012/21Uxsk56VDQ/00001.wav`)

3. copy the `voxceleb1/vox1_test_wav.zip` file into the voxceleb2 folder.

4. Unpack voxceleb1 test files(verification split).

Go to the voxceleb2 folder and run `unzip vox1_test_wav.zip`.

5. Copy the verification split(`voxceleb1/ meta/veri_test.txt`) into voxceleb2(`voxceleb2/meta/ veri_test.txt`)

6. Now everything is ready and you can run voxceleb2 experiments:
- training embeddings:

`python train_speaker_embeddings.py hyperparams/train_xvector_voxceleb2.yaml`

Note: To prepare the voxceleb1 + voxceleb2 dataset you have to copy and unpack vox1_dev_wav.zip for the voxceleb1 dataset.

# Performance summary

[Speaker Verification Results with Voxceleb 1]
| System          | Dataset    | EER  |
|-----------------|------------|------|
| Xvector + PLDA  | VoxCeleb 1 | 5.8% |
| Xvector + CL    | Voxceleb 1 | 3.8% |
| ECAPA-TDNN      | Voxceleb 1 | 3.4% |

[Speaker Verification Results with Voxceleb 2]
| System          | Dataset    | EER  |
|-----------------|------------|------|
| Xvector + PLDA  | VoxCeleb 2 | 4.7% |
| Xvector + CL    | Voxceleb 2 | 3.1% |
| ECAPA-TDNN      | Voxceleb 2 | 1.6% |


[Speaker Verification Results with Voxceleb 1 + Voxceleb2]
| System          | Dataset    | EER  |
|-----------------|------------|------|
| Xvector + PLDA  | VoxCeleb 2 | 4.3% |
| Xvector + CL    | Voxceleb 2 | 2.9% |
| ECAPA-TDNN      | Voxceleb 2 | 1.3% |
| ECAPA-TDNN big  | Voxceleb 2 | 1.2% |


# Resources

## Voxceleb 1
- xvector(full exp folder): https://www.dropbox.com/sh/e00wna66l1xhefc/AAAaVP9l5tzTs9YM_wapvW5Na?dl=1

- xvector(model only): https://www.dropbox.com/s/skfz2sme5nw7jji/xvector_model.ckpt?dl=1

- xvector + CL (full exp folder): https://www.dropbox.com/sh/n7j75yurvfolq5l/AACpHxqZqs3HPow5byxGq-dRa?dl=1

- xvector + CL (model only): https://www.dropbox.com/s/1ep1ccgvswa2bl8/embedding_model.ckpt?dl=1

- ecapa-tdnn(model only): https://www.dropbox.com/s/lbv09i1nb8f9z7t/embedding_model.ckpt?dl=1


## Voxceleb 2
- xvector(full exp folder): https://www.dropbox.com/sh/3ui8ju1kjqnvh70/AAB5ALciI7ObSy8_HsmJrnOOa?dl=1

- xvector(model only): https://www.dropbox.com/s/exzbyt4qoabo7v4/embedding_model.ckpt?dl=1

- xvector + CL (full exp folder): https://www.dropbox.com/sh/egbd9jsywbsjm45/AABC1hh3AngRZZ_yQBCMkKVQa?dl=1

- xvector + CL (model only): https://www.dropbox.com/s/exzbyt4qoabo7v4/embedding_model.ckpt?dl=1

- ecapa-tdnn (full exp folder): https://www.dropbox.com/sh/9tpae97au7yopwj/AABF6weFsd4Gb7EkU7vZco6ha?dl=1

- ecapa-tdnn(model only): https://www.dropbox.com/s/n4kkhss16fbku5a/embedding_model.ckpt?dl=1


## Voxceleb 1 + Voxceleb2

- xvector(full exp folder): https://www.dropbox.com/sh/fu9mwk42qa5ufqw/AAAoFqD_sZF8FwLpx6Yi4nrta?dl=1

- xvector(model only): https://www.dropbox.com/s/uq6vxk3e9zosvwd/xvector_model.ckpt?dl=1

- xvector + CL (full exp folder): https://www.dropbox.com/sh/m4y3y1s7974j51j/AAB9V8xP4T1jQCxxvFUu3Fsca?dl=1

- xvector + CL (model only): https://www.dropbox.com/s/blsr7iybtcjrusy/embedding_model.ckpt?dl=1

- ecapa-tdnn(full exp folder): https://www.dropbox.com/sh/9tpae97au7yopwj/AABF6weFsd4Gb7EkU7vZco6ha?dl=1

- ecapa-tdnn(model only): https://www.dropbox.com/s/ovrzhwnik651rzj/embedding_model.ckpt?dl=1

- ecapa-tdnn(full exp folder): https://www.dropbox.com/sh/3it4isnwul20lov/AAAlVdQtcWfk3Bld7gmc5Ljea?dl=1

- ecapa-tdnn(model only): https://www.dropbox.com/s/2mdnl784ram5w8o/embedding_model.ckpt?dl=1

