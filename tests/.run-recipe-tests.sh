#!/bin/bash

# TIMIT
python recipes/TIMIT/ASR/CTC/train/train.py recipes/TIMIT/ASR/CTC/train/hparams/train.yaml --output_folder=test_results/TIMIT_CTC --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug 


python recipes/TIMIT/ASR/seq2seq/train/train.py recipes/TIMIT/ASR/seq2seq/train/hparams/train.yaml --output_folder=test_results/TIMIT_seq2seq --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

python recipes/TIMIT/ASR/transducer/train/train.py recipes/TIMIT/ASR/transducer/train/hparams/train.yaml --output_folder=test_results/TIMIT_RNNT --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# LibriSpeech
python recipes/LibriSpeech/ASR/seq2seq/train/train.py recipes/LibriSpeech/ASR/seq2seq/train/hparams/train_BPE_1000.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Libri1000BPE --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=[samples/audio_samples/nn_training_samples/debug.csv] --lm_hparam_file=recipes/LibriSpeech/LM/pretrained/hparams/pretrained_RNNLM_BPE1000.yaml --skip_prep=True --debug

python recipes/LibriSpeech/G2P/train/train.py recipes/LibriSpeech/G2P/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/G2P --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# CommonVoice
python recipes/CommonVoice/ASR/seq2seq/train/train.py recipes/CommonVoice/ASR/seq2seq/train/hparams/train.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/CommonVoice --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=30 --skip_prep=True --debug


# VoiceBank
python recipes/Voicebank/ASR/CTC/train/train.py recipes/Voicebank/ASR/CTC/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_ASR --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

# VoxCeleb
python recipes/VoxCeleb/SpeakerRec/train/train_speaker_embeddings.py recipes/VoxCeleb/SpeakerRec/train/hparams/train_ecapa_tdnn.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Voxceleb_ecapa --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --sentence_len=1.0 --skip_prep=True --debug

python recipes/VoxCeleb/SpeakerRec/train/train_speaker_embeddings.py recipes/VoxCeleb/SpeakerRec/train/hparams/train_x_vectors.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Voxceleb_xvect --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --sentence_len=1.0 --skip_prep=True --debug

