#!/bin/bash
for rth in 0.2 0.4 0.6 0.8
do
	EXPERIMENT_NAME=cnn14_stft_roarth_$rth
	TS_SOCKET=/tmp/gpu1 CUDA_VISIBLE_DEVICES=1 ts python -Wignore train_classifier.py hparams/roar_cnn14.yaml --experiment_name $EXPERIMENT_NAME \
    	--add_wham_noise True --seed 2 \
    	--n_mels 80 --roar_th $rth
done

