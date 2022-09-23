#!/bin/sh

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_EPFLP300 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/MOABB/EEGNet_EPFLP300_seed_variability
HPOPT_CONFIG_FILE=hparams/orion/hparams_random_search.yaml  #hparam file for orion

#export _MNE_FAKE_HOME_DIR='' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/home/dborra/Documents/codes/speechbrain-2/recipes/MOABB/results/random_search_EEGNet_EPFLP300.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..
#'random_seed'
orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=250  \
	./run_experiments_seed_variability.sh hparams/EEGNet_EPFLP300_hparam_search.yaml \
	/home/dborra/Documents/data $OUTPUT_FOLDER 1 1 'random_seed' 1 f1 valid_metrics.pkl false true \
  --number_of_epochs~"uniform(100, 1000,discrete=True)" \
	--avg_models~"uniform(1, 20,discrete=True)" \
	--step_size~"uniform(50,100,discrete=True)" \
	--lr~"loguniform(1e-5, 1e-1)" \
	--n_steps_channel_selection~"uniform(1, 3,discrete=True)" \
	--fmax~"uniform(10, 50,discrete=True)" \
	--cnn_temporal_kernels~"uniform(4, 64,discrete=True)" \
	--cnn_temporal_kernelsize~"uniform(8, 64,discrete=True)" \
	--cnn_spatial_depth_multiplier~"uniform(1, 4,discrete=True)" \
	--cnn_spatial_pool~"uniform(1, 4,discrete=True)" \
	--cnn_septemporal_depth_multiplier~"uniform(1, 4,discrete=True)" \
	--cnn_septemporal_kernelsize~"uniform(4, 32,discrete=True)" \
	--cnn_septemporal_pool~"uniform(1, 8,discrete=True)" \
	--dropout~"uniform(0.0, 0.50)" \
  --dims_to_normalize~"uniform(1, 2,discrete=True)"
