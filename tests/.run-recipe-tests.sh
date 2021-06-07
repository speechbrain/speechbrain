#!/bin/bash
# This script runs several recipes in debug modality with the small dataset saved in samples. 
# It must be called from the main SpeechBrain folder:
# tests/.run-recipe-tests.sh 
#
# Author: Mirco Ravanelli 2021

# TEMPLATES
python templates/enhancement/train.py templates/enhancement/train.yaml --output_folder=test_results/template_enhancement --data_folder='data_enh' --train_annotation='data_enh/train.json' --valid_annotation='data_enh/valid.json' --test_annotation='data_enh/test.json' --debug

python templates/speaker_id/train.py templates/speaker_id/train.yaml --output_folder=test_results/template_speaker_id --data_folder='data_spk_id' --train_annotation='data_spk_id/train.json' --valid_annotation='data_spk_id/valid.json' --test_annotation='data_spk_id/test.json' --debug

python templates/speech_recognition/Tokenizer/train.py templates/speech_recognition/Tokenizer/tokenizer.yaml --output_folder=test_results/template_tokenizer/ --data_folder='data_tok' --train_annotation='data_tok/train.json' --valid_annotation='data_tok/valid.json' --test_annotation='data_tok/test.json' --debug

python templates/speech_recognition/LM/train.py templates/speech_recognition/LM/RNNLM.yaml --output_folder=test_results/template_lm/ --lm_train_data=templates/speech_recognition/LM/data/train.txt --lm_valid_data=templates/speech_recognition/LM/data/valid.txt --lm_test_data=templates/speech_recognition/LM/data/test.txt --tokenizer_file=templates/speech_recognition/Tokenizer/save/1000_unigram.model --debug

python templates/speech_recognition/ASR/train.py templates/speech_recognition/ASR/train.yaml --output_folder=test_results/template_asr/ --data_folder='data_asr' --train_annotation='data_asr/train.json' --valid_annotation='data_asr/valid.json' --test_annotation='data_asr/test.json' --debug

# TIMIT
python recipes/TIMIT/ASR/CTC/train.py recipes/TIMIT/ASR/CTC/hparams/train.yaml --output_folder=test_results/TIMIT_CTC --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug 


python recipes/TIMIT/ASR/seq2seq/train.py recipes/TIMIT/ASR/seq2seq/hparams/train.yaml --output_folder=test_results/TIMIT_seq2seq --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=46 --skip_prep=True --debug

python recipes/TIMIT/ASR/transducer/train.py recipes/TIMIT/ASR/transducer/hparams/train.yaml --output_folder=test_results/TIMIT_RNNT --data_folder=samples/audio_samples/nn_training_samples  --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --test_annotation=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# LibriSpeech
python recipes/LibriSpeech/ASR/seq2seq/train.py recipes/LibriSpeech/ASR/seq2seq/hparams/train_BPE_1000.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Libri1000BPE --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=[samples/audio_samples/nn_training_samples/debug.csv] --skip_prep=True --debug

python recipes/LibriSpeech/ASR/seq2seq/train.py recipes/LibriSpeech/ASR/seq2seq/hparams/train_BPE_5000.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Libri5000BPE --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=[samples/audio_samples/nn_training_samples/debug.csv] --skip_prep=True --debug

python recipes/LibriSpeech/ASR/transformer/train.py recipes/LibriSpeech/ASR/transformer/hparams/transformer.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Libritransformer --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=[samples/audio_samples/nn_training_samples/debug.csv] --skip_prep=True --debug


python recipes/LibriSpeech/G2P/train.py recipes/LibriSpeech/G2P/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/G2P --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# CommonVoice
python recipes/CommonVoice/ASR/seq2seq/train.py recipes/CommonVoice/ASR/seq2seq/hparams/train_it.yaml  --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/CommonVoice --train_csv=samples/audio_samples/nn_training_samples/debug.csv --valid_csv=samples/audio_samples/nn_training_samples/debug.csv --test_csv=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=30 --skip_prep=True --debug

#AISHELL
python recipes/AISHELL-1/ASR/train.py recipes/AISHELL-1/ASR/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/AISHELL --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --output_neurons=44 --skip_prep=True --debug

# VoiceBank
python recipes/Voicebank/ASR/CTC/train.py recipes/Voicebank/ASR/CTC/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_ASR --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --batch_size=1 --output_neurons=17 --debug

python recipes/Voicebank/ASR/seq2seq/train.py recipes/Voicebank/ASR/seq2seq/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_ASR --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --batch_size=1 --debu

python recipes/Voicebank/enhance/waveform_map/train.py recipes/Voicebank/enhance/waveform_map/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --debug

python recipes/Voicebank/enhance/spectral_mask/train.py recipes/Voicebank/enhance/spectral_mask/hparams/train.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --debug

python recipes/Voicebank/MTL/ASR_enhance/train.py recipes/Voicebank/MTL/ASR_enhance/hparams/pretrain_perceptual.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --output_neurons=17 --debug

python recipes/Voicebank/MTL/ASR_enhance/train.py recipes/Voicebank/MTL/ASR_enhance/hparams/enhance_mimic.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --output_neurons=17 --debug

python recipes/Voicebank/MTL/ASR_enhance/train.py recipes/Voicebank/MTL/ASR_enhance/hparams/robust_asr.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/VoiceBank_enh --train_annotation=samples/audio_samples/nn_training_samples/debug.json --valid_annotation=samples/audio_samples/nn_training_samples/debug.json --test_annotation=samples/audio_samples/nn_training_samples/debug.json --skip_prep=True --debug

# VoxCeleb
python recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Voxceleb_ecapa --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --sentence_len=1.0 --skip_prep=True --debug

python recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py recipes/VoxCeleb/SpeakerRec/hparams/train_x_vectors.yaml --data_folder=samples/audio_samples/nn_training_samples --output_folder=test_results/Voxceleb_xvect --train_annotation=samples/audio_samples/nn_training_samples/debug.csv --valid_annotation=samples/audio_samples/nn_training_samples/debug.csv --sentence_len=1.0 --skip_prep=True --debug

# timers and Such
python recipes/timers-and-such/direct/train.py recipes/timers-and-such/direct/hparams/train.yaml --output_folder=test_results/timers_and_such_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/timers-and-such/multistage/train.py recipes/timers-and-such/multistage/hparams/train_TAS_LM.yaml --output_folder=test_results/timers_and_such_multistage --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/timers-and-such/multistage/train.py recipes/timers-and-such/multistage/hparams/train_LS_LM.yaml --output_folder=test_results/timers_and_such_multistage --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/timers-and-such/decoupled/train.py recipes/timers-and-such/decoupled/hparams/train_LS_LM.yaml --output_folder=test_results/timers_and_such_decoupled --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/timers-and-such/decoupled/train.py recipes/timers-and-such/decoupled/hparams/train_TAS_LM.yaml --output_folder=test_results/timers_and_such_decoupled --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/timers-and-such/LM/train.py recipes/timers-and-such/LM/hparams/train.yaml --output_folder=test_results/timers_and_such_LM --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test_real=samples/audio_samples/nn_training_samples/debug.csv --csv_test_synth=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

# SLURP
python recipes/SLURP/direct/train.py recipes/SLURP/direct/hparams/train.yaml --output_folder=test_results/SLURP_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/SLURP/NLU/train.py recipes/SLURP/NLU/hparams/train.yaml --output_folder=test_results/NLU_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

# WSJ
python recipes/WSJ0Mix/separation/train.py recipes/WSJ0Mix/separation/hparams/sepformer.yaml --output_folder=test_results/WSJ_sepformer --data_folder=samples/audio_samples/nn_training_samples  --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/WSJ0Mix/separation/train.py recipes/WSJ0Mix/separation/hparams/convtasnet.yaml --output_folder=test_results/WSJ_convtasnet --data_folder=samples/audio_samples/nn_training_samples  --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

python recipes/WSJ0Mix/separation/train.py recipes/WSJ0Mix/separation/hparams/dprnn.yaml --output_folder=test_results/WSJ_dprnn --data_folder=samples/audio_samples/nn_training_samples  --train_data=samples/audio_samples/nn_training_samples/debug.csv --valid_data=samples/audio_samples/nn_training_samples/debug.csv --test_data=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug

# Fluent Speech Command
python recipes/fluent-speech-commands/direct/train.py recipes/fluent-speech-commands/direct/hparams/train.yaml --output_folder=test_results/fluent_direct --data_folder=samples/audio_samples/nn_training_samples  --csv_train=samples/audio_samples/nn_training_samples/debug.csv --csv_valid=samples/audio_samples/nn_training_samples/debug.csv --csv_test=samples/audio_samples/nn_training_samples/debug.csv --skip_prep=True --debug






