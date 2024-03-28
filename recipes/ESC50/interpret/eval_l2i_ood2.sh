python eval.py hparams/l2i_cnn14.yaml --experiment_name wn_ao_cnn14_l10 \
  --pretrained_PIQ $1 --exp_method l2i --add_wham_noise False --mrt False --wham_folder /work/wham_noise --data_folder /work/ESC50 \
  --nmf_decoder_path $2 --overlap_type white_noise --use_stft2mel True --use_melspectra False
