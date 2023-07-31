#!/bin/sh
module load python/3.10.2
source $HOME/venv0/bin/activate

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_BNCI2014001 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/EEGNet_BNCI2014001_seed_variability_moabb
HPOPT_CONFIG_FILE=hparams/orion/hparams_tpe.yaml   #hparam file for orion
#/home/dborra/projects/def-ravanelm/dborra/
export _MNE_FAKE_HOME_DIR='/home/dborra/projects/def-ravanelm/dborra' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/home/dborra/projects/def-ravanelm/dborra/tpe_csbj_EEGNet_BNCI2014001_stage2_v3.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..
# dborra paths
# - export ORION_DB_ADDRESS=/mnt/Dilbert/dborra/tpe_EEGNet_BNCI2014001.pkl
# - /mnt/Dilbert/dborra/mne_data/

orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=50  \
        ./run_experiments.sh hparams/EEGNet_BNCI2014001.yaml \
        /home/dborra/projects/def-ravanelm/dborra /home/dborra/scratch $OUTPUT_FOLDER 9 2 'random_seed' 1 acc valid_metrics.pkl false true \
        --number_of_epochs 940 \
        --avg_models 15 \
        --batch_size_exponent 4 \
        --lr 0.0001\
        --tmax 3.9 \
        --fmin 0.14 \
        --fmax 45.7 \
        --n_steps_channel_selection 2 \
        --cnn_temporal_kernels 61 \
        --cnn_temporal_kernelsize 29 \
        --cnn_spatial_depth_multiplier 3 \
        --cnn_septemporal_point_kernels_ratio_ 7 \
        --cnn_septemporal_kernelsize_ 15 \
        --cnn_septemporal_pool 3 \
        --dropout 0.1748 \
        --repeat_augment 1 \
        --max_num_segments~"uniform(2, 6, discrete=True)" \
        --amp_delta~"uniform(0.0, 0.5)" \
        --shift_delta_~"uniform(0, 25, discrete=True)" \
        --snr_white_low~"uniform(0.0, 15, precision=2)" \
        --snr_white_delta~"uniform(5.0, 20.0, precision=3)"