# Speaker recognition experiments with VoxCeleb.
This folder contains scripts for running speaker identification and verification experiments with the VoxCeleb dataset(http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

# Training Speaker Embeddings
Run the following command to train speaker embeddings using xvectors:

`python train_speaker_embeddings.py hyperparams/train_xvector_voxceleb1.yaml` (for voxceleb1)
`python train_speaker_embeddings.py hyperparams/train_xvector_voxceleb2.yaml` (for voxceleb2, see preparation instructions below).

The system trains a TDNN for speaker embeddings coupled with a speaker-id classifier. The speaker-id accuracy should be around 97-98% for both voxceleb1 and voceleb2.

# Speaker verification with PLDA
After training the speaker embeddings, it is possible to perform speaker verification using PLDA.  You can run it with the following command:

`python speaker_verification_plda.py hyperparams/verification_plda_xvector_voxceleb1.yaml` (for voxceleb1)
`python speaker_verification_plda.py hyperparams/verification_plda_xvector_voxceleb2.yaml` (for voxceleb2)

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER = 6.9 % on voxceleb1 and EER = 5.8 % on voxceleb2.

# Speaker verification with a binary discriminator
It is possible to perform speaker verification using contrastive learning.
In particular, we employ a pre-trained encoder followed by a binary discriminator. The discriminator is fed with either positive or negative embeddings that are properly sampled from the dataset.  To run this experiment, type the following command:

`python speaker_verification_discriminator.py hyperparams/verfication_discriminator_xvector_voxceleb1.yaml`(for voxceleb1)
`python speaker_verification_discriminator.py hyperparams/verfication_discriminator_xvector_voxceleb2.yaml`(for voxceleb2)


If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER = 3.8 % on voxceleb1 and EER = 3.1 % on voxceleb2.

# Model/Experment folder donwload:
Voxceleb1:

xvector(full exp folder): https://www.dropbox.com/sh/e00wna66l1xhefc/AAAaVP9l5tzTs9YM_wapvW5Na?dl=1
xvector(model only): https://www.dropbox.com/s/skfz2sme5nw7jji/xvector_model.ckpt?dl=1

Voxceleb2:

xvector(full exp folder): https://www.dropbox.com/sh/3ui8ju1kjqnvh70/AAB5ALciI7ObSy8_HsmJrnOOa?dl=1
xvector(model only): https://www.dropbox.com/s/exzbyt4qoabo7v4/embedding_model.ckpt?dl=1

# VoxCeleb2 Preparation
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
