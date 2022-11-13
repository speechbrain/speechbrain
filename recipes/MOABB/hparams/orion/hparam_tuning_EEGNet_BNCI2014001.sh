#!/bin/sh

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_BNCI2014001 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/EEGNet_BNCI2014001_seed_variability_moabb
HPOPT_CONFIG_FILE=hparams/orion/hparams_tpe.yaml   #hparam file for orion

export _MNE_FAKE_HOME_DIR='/network/scratch/r/ravanelm/' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/network/scratch/r/ravanelm/tpe_EEGNet_BNCI2014001.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..
# dborra paths
# - export ORION_DB_ADDRESS=/mnt/Dilbert/dborra/tpe_EEGNet_BNCI2014001.pkl
# - /mnt/Dilbert/dborra/mne_data/

orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=250  \
        ./run_experiments_seed_variability.sh hparams/EEGNet_BNCI2014001_seed_variability.yaml \
        /localscratch/eeg_data $OUTPUT_FOLDER 1 1 'random_seed' 1 acc valid_metrics.pkl false true \
        --number_of_epochs~"uniform(250, 1000, discrete=True)" \
        --avg_models~"uniform(1, 15,discrete=True)" \
        --batch_size_exponent~"uniform(4, 6,discrete=True)" \
        --lr~"choices([0.01, 0.005, 0.001, 0.0005, 0.0001])" \
        --tmax~"uniform(1.0, 4.0, precision=2)" \
        --fmin~"uniform(0.1, 5, precision=2)" \
        --fmax~"uniform(30.0, 50.0, precision=3)" \
        --n_steps_channel_selection~"uniform(1, 3,discrete=True)" \
        --cnn_temporal_kernels~"uniform(4, 64,discrete=True)" \
        --cnn_temporal_kernelsize~"uniform(24, 62,discrete=True)" \
        --cnn_spatial_depth_multiplier~"uniform(1, 4,discrete=True)" \
        --cnn_septemporal_point_kernels_ratio_~"uniform(0, 8, discrete=True)" \
        --cnn_septemporal_kernelsize_~"uniform(3, 24,discrete=True)" \
        --cnn_septemporal_pool~"uniform(1, 8,discrete=True)" \
        --dropout~"uniform(0.0, 0.5)" \
        --repeat_augment~"uniform(0, 2,discrete=True)" \
        --idx_combination_augmentations~"uniform(1, 15,discrete=True)"
# compact hparam space (17 hparams) removing direct hparam dependancies (e.g., batch size and step size, spatial_pool and cnn_septemporal_kernelsize_

# 5 augmenters: 31 total combinations  + no augm: 32 (idx=0 no augm, idx=31 all augmenters)
# 4 augmenters: 15 total combinations + no augm: 16 (idx=0 no augm, idx=15 all augmenters)
# todo: 1) test n_initial_points: 50 in hparams_tpe; 2) lr in --lr~"loguniform(1e-4, 1.0)"; 3) restore drop_chunk and up to 31 augmenters