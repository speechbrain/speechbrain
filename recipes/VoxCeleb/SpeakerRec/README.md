## Speaker recognition experiments with VoxCeleb.
This folder contains scripts for running speaker identification and verification experiments with the VoxCeleb dataset (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

### Training Speaker Embeddings
Run the following command to train speaker embeddings using xvectors:

`python train_speaker_embeddings.py hyperparams/train_xvector_voxceleb1.yaml`

The system trains a TDNN for speaker embeddings coupled with a speaker-id classifier. The speaker-id accuracy should be around 97-98%.

### Speaker verification with PLDA
After training the speaker embeddings, it is possible to perform speaker verification using PLDA.  You can run it with the following command:

`python speaker_verification_plda.py hyperparams/verification_plda_xvector_voxceleb1.yaml`

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER=6.9%

### Speaker verification with a binary discriminator
It is possible to perform speaker verification using contrastive learning.
In particular, we employ a pre-trained encoder followed by a binary discriminator. The discriminator is fed with either positive or negative embeddings that are properly sampled from the dataset.  To run this experiment, type the following command:

`python speaker_verification_discriminator.py hyperparams/verfication_discriminator_xvector_voxceleb1.yaml`

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER=4.3%


