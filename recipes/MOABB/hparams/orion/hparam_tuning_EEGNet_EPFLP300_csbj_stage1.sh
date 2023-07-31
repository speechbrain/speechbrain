#!/bin/sh
module load python/3.10.2
source $HOME/venv0/bin/activate

HPOPT_EXPERIMENT_NAME=$1 #random_seach_EEGNet_EPFLP300 # name of the orion experiment
OUTPUT_FOLDER=$2 # results/MOABB/EEGNet_EPFLP300_seed_variability
HPOPT_CONFIG_FILE=hparams/orion/hparams_tpe.yaml  #hparam file for orion

export _MNE_FAKE_HOME_DIR='/home/dborra/projects/def-ravanelm/dborra/' # change with your own folder (needed for Mirco)
export ORION_DB_ADDRESS=/home/dborra/projects/def-ravanelm/dborra/tpe_csbj_EEGNet_EPFLP300_stage1.pkl # This is the database where orion will save the results
export ORION_DB_TYPE=PickledDB

# Running orion
cd ../..
#'random_seed'
orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE --exp-max-trials=50  \
	./run_experiments.sh hparams/EEGNet_EPFLP300.yaml \
	/home/dborra/projects/def-ravanelm/dborra/ /home/dborra/scratch $OUTPUT_FOLDER 8 4 'random_seed' 1 f1 valid_metrics.pkl false true \
  --number_of_epochs~"uniform(250, 1000, discrete=True)" \
  --avg_models~"uniform(1, 15,discrete=True)" \
  --batch_size_exponent~"uniform(4, 6,discrete=True)" \
  --lr~"choices([0.01, 0.005, 0.001, 0.0005, 0.0001])" \
  --fmin~"uniform(0.1, 5, precision=2)" \
  --fmax~"uniform(20.0, 50.0, precision=3)" \
  --n_steps_channel_selection~"uniform(1, 3,discrete=True)" \
  --cnn_temporal_kernels~"uniform(4, 64,discrete=True)" \
  --cnn_temporal_kernelsize~"uniform(24, 62,discrete=True)" \
  --cnn_spatial_depth_multiplier~"uniform(1, 4,discrete=True)" \
  --cnn_septemporal_point_kernels_ratio_~"uniform(0, 8, discrete=True)" \
  --cnn_septemporal_kernelsize_~"uniform(3, 24,discrete=True)" \
  --cnn_septemporal_pool~"uniform(1, 8,discrete=True)" \
  --dropout~"uniform(0.0, 0.5)"
