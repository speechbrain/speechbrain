#!/bin/bash
# This script runs several recipes in debug modality with the small dataset saved in samples. 
# It must be called from the main SpeechBrain folder:
# tests/.run-recipe-tests.sh 
#
# Author: Mirco Ravanelli 2021


# TIMIT
python recipes/TIMIT/ASR/CTC/train/train.py recipes/TIMIT/ASR/CTC/train/hparams/train.yaml --output_folder=test_results/TIMIT_CTC --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug 


python recipes/TIMIT/ASR/seq2seq/train/train.py recipes/TIMIT/ASR/seq2seq/train/hparams/train.yaml --output_folder=test_results/TIMIT_seq2seq --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=46 --skip_prep=True --debug

python recipes/TIMIT/ASR/transducer/train/train.py recipes/TIMIT/ASR/transducer/train/hparams/train.yaml --output_folder=test_results/TIMIT_RNNT --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# LibriSpeech
python recipes/LibriSpeech/ASR/seq2seq/train/train.py recipes/LibriSpeech/ASR/seq2seq/train/hparams/train_BPE_1000.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Libri1000BPE --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=[samples/audio_samples/nn_training_samples/debug.csv] --lm_hparam_file=recipes/LibriSpeech/LM/pretrained/hparams/pretrained_RNNLM_BPE1000.yaml --skip_prep=True --debug

python recipes/LibriSpeech/G2P/train/train.py recipes/LibriSpeech/G2P/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/G2P --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# CommonVoice
python recipes/CommonVoice/ASR/seq2seq/train/train.py recipes/CommonVoice/ASR/seq2seq/train/hparams/train.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/CommonVoice --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=30 --skip_prep=True --debug


# VoiceBank
python recipes/Voicebank/ASR/CTC/train/train.py recipes/Voicebank/ASR/CTC/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_ASR --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --batch_size=1 --output_neurons=17 --debug

python recipes/Voicebank/ASR/seq2seq/train/train.py recipes/Voicebank/ASR/seq2seq/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_ASR --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --batch_size=1 --debu

python recipes/Voicebank/enhance/waveform_map/train/train.py recipes/Voicebank/enhance/waveform_map/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --debug

python recipes/Voicebank/enhance/spectral_mask/train/train.py recipes/Voicebank/enhance/spectral_mask/train/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --debug

python recipes/Voicebank/MTL/ASR_enhance/train/train.py recipes/Voicebank/MTL/ASR_enhance/train/hparams/pretrain_perceptual.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --output_neurons=17 --debug

python recipes/Voicebank/MTL/ASR_enhance/train/train.py recipes/Voicebank/MTL/ASR_enhance/train/hparams/enhance_mimic.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --output_neurons=17 --debug

python recipes/Voicebank/MTL/ASR_enhance/train/train.py recipes/Voicebank/MTL/ASR_enhance/train/hparams/robust_asr.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --debug

# VoxCeleb
python recipes/VoxCeleb/SpeakerRec/train/train_speaker_embeddings.py recipes/VoxCeleb/SpeakerRec/train/hparams/train_ecapa_tdnn.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Voxceleb_ecapa --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --sentence_len=1.0 --skip_prep=True --debug

python recipes/VoxCeleb/SpeakerRec/train/train_speaker_embeddings.py recipes/VoxCeleb/SpeakerRec/train/hparams/train_x_vectors.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Voxceleb_xvect --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --sentence_len=1.0 --skip_prep=True --debug

# timers and Such
python recipes/timers-and-such/direct/train/train.py recipes/timers-and-such/direct/train/hparams/train.yaml --output_folder=test_results/timers_and_such_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

python recipes/timers-and-such/multistage/train/train.py recipes/timers-and-such/multistage/train/hparams/train_TAS_LM.yaml --output_folder=test_results/timers_and_such_multistage --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

python recipes/timers-and-such/multistage/train/train.py recipes/timers-and-such/multistage/train/hparams/train_LS_LM.yaml --output_folder=test_results/timers_and_such_multistage --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

python recipes/timers-and-such/decoupled/train/train.py recipes/timers-and-such/decoupled/train/hparams/train_LS_LM.yaml --output_folder=test_results/timers_and_such_decoupled --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

python recipes/timers-and-such/decoupled/train/train.py recipes/timers-and-such/decoupled/train/hparams/train_TAS_LM.yaml --output_folder=test_results/timers_and_such_decoupled --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

python recipes/timers-and-such/LM/train/train.py recipes/timers-and-such/LM/train/hparams/train.yaml --output_folder=test_results/timers_and_such_LM --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

# SLURP
python recipes/SLURP/direct/train/train.py recipes/SLURP/direct/train/hparams/train.yaml --output_folder=test_results/SLURP_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --debug

python recipes/SLURP/NLU/train/train.py recipes/SLURP/NLU/train/hparams/train.yaml --output_folder=test_results/NLU_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --asr_tokenizer_file=recipes/timers-and-such/Tokenizer/pretrained/pretrained_tok/51_unigram.model --slu_tokenizer_file=recipes/SLURP/Tokenizer/pretrained/pretrained_tok/51_unigram.model  --debug

# WSJ
python recipes/WSJ2Mix/separation/train/train.py recipes/WSJ2Mix/separation/train/hparams/sepformer.yaml --output_folder=test_results/WSJ_sepformer --data_folder=samples/audio_samples/nn_training_samples  --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/WSJ2Mix/separation/train/train.py recipes/WSJ2Mix/separation/train/hparams/convtasnet.yaml --output_folder=test_results/WSJ_convtasnet --data_folder=samples/audio_samples/nn_training_samples  --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/WSJ2Mix/separation/train/train.py recipes/WSJ2Mix/separation/train/hparams/dprnn.yaml --output_folder=test_results/WSJ_dprnn --data_folder=samples/audio_samples/nn_training_samples  --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

# Fluent Speech Command
python recipes/fluent-speech-commands/direct/train.py recipes/fluent-speech-commands/direct/hparams/train.yaml --output_folder=test_results/fluent_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug






