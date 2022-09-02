#!/bin/sh

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_EPFLP300 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/MOABB/EEGNet_EPFLP300_seed_variability
HPOPT_CONFIG_FILE=hparams/orion/hparams_random_search.yaml  #hparam file for orion

#export _MNE_FAKE_HOME_DIR='' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/home/dborra/Documents/codes/speechbrain-2/recipes/MOABB/hparams/orion/results/MOABB/random_seach_EEGNet_EPFLP300.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..

orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=250  \
	./run_experiments_seed_variability.sh hparams/EEGNet_EPFLP300_hparam_search.yaml \
	/home/dborra/Documents/data $OUTPUT_FOLDER 1 1 'random_seed' 1 f1 valid_metrics.pkl false true \
  --number_of_epochs~"uniform(100, 1000,discrete=True)" \
	--avg_models~"uniform(1, 20,discrete=True)" \
	--fmax~"choices([20,30,40,50])" \
	--lr~"choices([0.01,0.005, 0.001,0.0005, 0.0001])" \
	--test_with~"choices(['last','best'])" \
	--batch_size~"choices([32,64,128])" \
	--step_size~"choices([50,100,150])" \
	--label_smoothing~"uniform(0.00, 0.075)" \
	--cnn_temporal_kernels~"uniform(4, 64,discrete=True)" \
	--cnn_temporal_kernelsize~"uniform(8, 64,discrete=True)" \
	--cnn_spatial_depth_multiplier~"choices([1,2,3,4])" \
	--cnn_spatial_pool~"uniform(2, 8,discrete=True)" \
	--cnn_spatial_max_norm~"uniform(0.1, 2)" \
	--cnn_septemporal_depth_multiplier~"choices([1,2,3,4])" \
	--cnn_septemporal_kernelsize~"uniform(4, 32,discrete=True)" \
	--cnn_septemporal_pool~"uniform(2, 8,discrete=True)" \
	--cnn_pool_type~"choices(['avg','max'])" \
	--dense_max_norm~"uniform(0.1, 2)" \
	--dropout~"uniform(0.0, 0.50)" \
	--activation_type~"choices(['elu', 'relu', 'leaky_relu'])" \
	--snr_pink_low~"uniform(-5.0, 5.0)" \
	--snr_pink_high~"uniform(15.0, 30.0)" \
  --snr_white_low~"uniform(5.0, 15.0)" \
  --snr_white_high~"uniform(15.0, 30.0)" \
  --snr_muscular_low~"uniform(0.0, 15.0)" \
  --snr_muscular_high~"uniform(15.0, 30.0)" \
	--repeat_augment~"choices([1,2,3])" \
  --n_augmentations~"uniform(1, 10,discrete=True)" \
  --dims_to_normalize~"choices([1, 2])"