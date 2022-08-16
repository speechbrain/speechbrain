#!/bin/sh

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_BNCI2014001 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/EEGNet_BNCI2014001_seed_variability_moabb
HPOPT_CONFIG_FILE=hparams/orion/hparams_random_search.yaml  #hparam file for orion

export _MNE_FAKE_HOME_DIR='/network/scratch/r/ravanelm/' # change with your own folder (needed for mne)
export ORION_DB_ADDRESS=/network/scratch/r/ravanelm/random_seach_EEGNet_BNCI2014001.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..

orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=250  \
	./run_experiments_seed_variability.sh hparams/EEGNet_BNCI2014001_seed_variability.yaml \
	/localscratch/eeg_data $OUTPUT_FOLDER 1 1 'random_seed' 1 acc valid_metrics.pkl false true \
       	--number_of_epochs~"uniform(100, 1000,discrete=True)" \
	--avg_models~"uniform(1, 20,discrete=True)" \
	--tmin~"uniform(0.0, 1.0)" \
	--tmax~"uniform(2.0, 4.0)" \
	--fmin~"uniform(0, 10)" \
	--fmax~"uniform(30, 60)" \
	--lr~"choices([0.01,0.005, 0.001,0.0005, 0.0001])" \
	--test_with~"choices(['last','best'])" \
	--batch_size~"choices([32,64,128])" \
	--step_size~"choices([50,100,150])" \
	--label_smoothing~"uniform(0.00, 0.075)" \
	--cnn_temporal_kernels~"uniform(4, 64,discrete=True)" \
	--cnn_temporal_kernelsize~"uniform(8, 64,discrete=True)" \
	--cnn_spatial_depth_multiplier~"choices([1,2,3,4])" \
	--cnn_septemporal_depth_multiplier~"choices([1,2,3,4])" \
	--cnn_pool_type~"choices(['avg','max'])" \
	--cnn_septemporal_pool~"uniform(2, 16,discrete=True)" \
	--dropout~"uniform(0.0, 0.50)" \
	--snr_pink_low~"uniform(-5.0, 5.0)" \
	--snr_pink_high~"uniform(15.0, 30.0)" \
        --snr_white_low~"uniform(5.0, 15.0)" \
        --snr_white_high~"uniform(15.0, 30.0)" \
        --snr_muscular_low~"uniform(0.0, 15.0)" \
        --snr_muscular_high~"uniform(15.0, 30.0)" \
	--repeat_augment~"choices([1,2,3])" \
        --n_augmentations~"uniform(1, 10,discrete=True)"




