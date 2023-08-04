#!/bin/bash

###########################################################
# Script to run leave-one-subject-out and/or leave-one-session-out training, optionally with multiple seeds.
# This script loops over the different subjects and sessions and trains different models. 
# At the end, the final performance is computed with the aggregate_results.py script that provides the average performance.
#
# Example:
# ./run_experiments.sh --hparams=hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data \
# --output_folder=results/MotorImagery/BNCI2014001/EEGNet --nsbj=9 --nsess=2 --seed=1986 --nruns=2 --number_of_epochs=10
#
# Authors:
# - Mirco Ravanelli (2023)
# - Davide Borra (2023)
###########################################################

# Initialize variables
hparams=""
data_folder=""
cached_data_folder=""
output_folder=""
nsbj=""
nsess=""
seed=""
nruns=""
eval_metric="acc"
eval_set="test"
train_mode="leave-one-session-out"
additional_flags=""

# Function to print argument descriptions and exit
print_argument_descriptions() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --hparams=hparams_path            Hparam YAML file"
    echo "  --data_folder=data_folder_path    Data folder path"
    echo "  --cached_data_folder=cache_path   Cached data folder path"
    echo "  --output_folder=output_path       Output folder path"
    echo "  --nsbj=num_subjects               Number of subjects"
    echo "  --nsess=num_sessions              Number of sessions"
    echo "  --seed=random_seed                Seed (random if not specified)"
    echo "  --nruns=num_runs                  Number of runs"
    echo "  --eval_metric=metric              Evaluation metric (e.g., acc or f1)"
    echo "  --eval_set=dev or test            Evaluation set. Default: test"
    echo "  --train_mode=mode   	      The training mode can be leave-one-subject-out or leave-one-session-out. Default: leave-one-session-out"
    exit 1
}

# Parse command-line arguments
for arg in "$@"; do
    case "$arg" in
        --hparams=*) hparams="${arg#*=}" ;;
        --data_folder=*) data_folder="${arg#*=}" ;;
        --cached_data_folder=*) cached_data_folder="${arg#*=}" ;;
        --output_folder=*) output_folder="${arg#*=}" ;;
        --nsbj=*) nsbj="${arg#*=}" ;;
        --nsess=*) nsess="${arg#*=}" ;;
        --seed=*) seed="${arg#*=}" ;;
        --nruns=*) nruns="${arg#*=}" ;;
        --eval_metric=*) eval_metric="${arg#*=}" ;;
        --eval_set=*) eval_set="${arg#*=}" ;;
        --train_mode=*) train_mode="${arg#*=}" ;;
        *) additional_flags+="$arg " ;;
    esac
done

# Check for required arguments
if [ -z "$hparams" ] || [ -z "$data_folder" ] || [ -z "$output_folder" ] || [ -z "$nsbj" ] || [ -z "$nsess" ] || [ -z "$nruns" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
fi

# Progress eval_set argument
if [ "$eval_set" = "dev" ]; then
  metric_file=valid_metrics.pkl
elif [ "$eval_set" = "test" ]; then
  metric_file=test_metrics.pkl
else
  echo "Invalid eval_set value: $eval_set. It can be test or dev only."
  exit 1
fi


# Manage Seed (optional argument)
seed="${seed:-$RANDOM}"

# Assign default value to cached_data_folder
if [ -z "$cached_data_folder" ]; then
    cached_data_folder="$data_folder/pkl"
fi

echo "hparams file : $hparams"
echo "Data folder: $data_folder"
echo "Cached data folder: $cached_data_folder"
echo "Output folder: $output_folder"
echo "No. of subjects: $nsbj"
echo "No. of sessions: $nsess"
echo "No. of runs: $nruns"
echo "Evaluation set: $eval_set"
echo "Training modality: $train_mode"
# Creating output folder
mkdir -p $output_folder\_seed\_$seed_init
mkdir -p $data_folder
mkdir -p $cached_data_folder

# Function to run the training experiment
run_experiment() {
  local target_session_idx="$1"

  for target_subject_idx in $(seq 0 1 $(( nsbj - 1 ))); do
    echo "Subject $target_subject_idx"
    python train.py $hparams --seed=$seed --data_folder=$data_folder --cached_data_folder=$cached_data_folder --output_folder=$output_folder/$seed \
      --target_subject_idx=$target_subject_idx --target_session_idx=$target_session_idx \
      --data_iterator_name="$train_mode" $additional_flags
  done
}

# Run multiple training experiments (with different seeds)
for i in $(seq 0 1 $(( nruns - 1 ))); do
  echo $seed

  # LEAVE-ONE-SUBJECT-OUT
  if [ "$train_mode" = "leave-one-subject-out" ]; then
    run_experiment 0
  # LEAVE-ONE-SESSION-OUT
  elif [ "$train_mode" = "leave-one-session-out" ]; then
    # Loop over sessions
    for j in $(seq 0 1 $(( nsess - 1 ))); do
      run_experiment $j
    done
  else
      echo "Invalid train_model value: $train_mode. It can be leave-one-subject-out or leave-one-session-out  only."
  exit 1
  fi


  # Store the results
  python utils/parse_results.py $output_folder/$seed $metric_file $eval_metric | tee -a  $output_folder/$seed\_results.txt

  # Changing Random seed
  seed=$((seed+1))
done


echo 'Final Results (Performance Aggregation)'
python utils/aggregate_results.py $output_folder $eval_metric | tee -a  $output_folder/aggregated_performance.txt

