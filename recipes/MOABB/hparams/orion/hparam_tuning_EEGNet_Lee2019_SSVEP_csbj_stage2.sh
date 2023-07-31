#!/bin/sh
module load python/3.10.2
source $HOME/venv0/bin/activate

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_Lee2019_SSVEP # name of the orion experiment
OUTPUT_FOLDER=$2 # results/EEGNet_Lee2019_SSVEP
HPOPT_CONFIG_FILE=hparams/orion/hparams_tpe.yaml   #hparam file for orion

export _MNE_FAKE_HOME_DIR='/home/dborra/projects/def-ravanelm/dborra' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/home/dborra/projects/def-ravanelm/dborra/tpe_csbj_EEGNet_Lee2019_SSVEP_stage2.pkl # This is the database where orion will save the results

# Running orion
cd ../..

orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=50  \
        ./run_experiments.sh hparams/EEGNet_Lee2019_SSVEP.yaml \
        /home/dborra/projects/def-ravanelm/dborra /home/dborra/scratch $OUTPUT_FOLDER 54 2 'random_seed' 1 acc valid_metrics.pkl false true \
        --number_of_epochs 559 \
        --avg_models 7 \
        --batch_size_exponent 5 \
        --lr 0.0005 \
        --tmax 2.3 \
        --fmin 4.8 \
        --fmax 49.9 \
        --n_steps_channel_selection 4 \
        --cnn_temporal_kernels 30 \
        --cnn_temporal_kernelsize 42 \
        --cnn_spatial_depth_multiplier 1 \
        --cnn_septemporal_point_kernels_ratio_ 6 \
        --cnn_septemporal_kernelsize_ 8 \
        --cnn_septemporal_pool 1 \
        --dropout 0.2662 \
        --repeat_augment 1 \
        --max_num_segments~"uniform(2, 6, discrete=True)" \
        --amp_delta~"uniform(0.0, 0.5)" \
        --shift_delta_~"uniform(0, 25, discrete=True)" \
        --snr_white_low~"uniform(0.0, 10, precision=2)" \
        --snr_white_delta~"uniform(10.0, 20.0, precision=3)"
