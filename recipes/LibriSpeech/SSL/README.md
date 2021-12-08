# wav2vec 2.0 pretraining with SpeechBrain

To run call `python train.py hparams/train_wav2vec.yaml --find_unused_parameters --max_grad_norm 0.0`

See config file for model definition.

The output of the forward method of the wav2vec module (Wav2Vec2) is a dictionary with the encoded embeddings in "embeddings".
