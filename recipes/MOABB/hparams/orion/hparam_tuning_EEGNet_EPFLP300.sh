#!/bin/sh

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_EPFLP300 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/MOABB/EEGNet_EPFLP300_seed_variability
HPOPT_CONFIG_FILE=hparams/orion/hparams_tpe.yaml  #hparam file for orion

export _MNE_FAKE_HOME_DIR='/network/scratch/r/ravanelm/' # change with your own folder (needed for Mirco)
export ORION_DB_ADDRESS=/network/scratch/r/ravanelm/tpe_EEGNet_EPFLP300.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..
#'random_seed'
orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=250  \
	./run_experiments_seed_variability.sh hparams/EEGNet_EPFLP300_hparam_search.yaml \
	/localscratch/eeg_data $OUTPUT_FOLDER 1 1 'random_seed' 1 f1 valid_metrics.pkl false true \
  --number_of_epochs~"uniform(100, 1000,discrete=True)" \
	--avg_models~"uniform(1, 20,discrete=True)" \
	--step_size~"uniform(50,150,discrete=True)" \
	--batch_size~"uniform(32,128,discrete=True)" \
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
  --dims_to_normalize~"uniform(1, 2,discrete=True)" \
	--snr_pink_low~"uniform(0.0, 15.0)" \
	--snr_pink_high~"uniform(15.0, 30.0)" \
  --snr_white_low~"uniform(0.0, 15.0)" \
  --snr_white_high~"uniform(15.0, 30.0)" \
  --snr_muscular_low~"uniform(0.0, 15.0)" \
  --snr_muscular_high~"uniform(15.0, 30.0)" \
	--repeat_augment~"uniform(1, 3,discrete=True)" \
  --n_augmentations~"uniform(1, 10,discrete=True)"
