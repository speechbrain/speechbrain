#!/bin/bash

###########################################################
# Script to run leave-one-subject-out and/or leave-one-session-out training, optionally with multiple seeds.
# This script loops over the different subjects and sessions and trains different models.
# At the end, the final performance is computed with the aggregate_results.py script that provides the average performance.
#
# Usage:
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
rnd_dir=False
additional_flags=""

# Function to print argument descriptions and exit
print_argument_descriptions() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --hparams hparams_path            Hparam YAML file"
    echo "  --data_folder data_folder_path    Data folder path"
    echo "  --cached_data_folder cache_path   Cached data folder path"
    echo "  --output_folder output_path       Output folder path"
    echo "  --nsbj num_subjects               Number of subjects"
    echo "  --nsess num_sessions              Number of sessions"
    echo "  --seed random_seed                Seed (random if not specified)"
    echo "  --nruns num_runs                  Number of runs"
    echo "  --eval_metric metric              Evaluation metric (e.g., acc or f1)"
    echo "  --eval_set dev or test            Evaluation set. Default: test"
    echo "  --train_mode mode                 The training mode can be leave-one-subject-out or leave-one-session-out. Default: leave-one-session-out"
    echo "  --rnd_dir                         If True the results are stored in a subdir of the output folder with a random name (useful to store all the results of an hparam tuning).  Default: False"
    exit 1
}


# Parse command line
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --hparams)
      hparams="$2"
      shift
      shift
      ;;

    --data_folder)
      data_folder="$2"
      shift
      shift
      ;;

    --cached_data_folder)
      cached_data_folder="$2"
      shift
      shift
      ;;

    --output_folder)
      output_folder="$2"
      shift
      shift
      ;;

    --nsbj)
      nsbj="$2"
      shift
      shift
      ;;

    --nsess)
      nsess="$2"
      shift
      shift
      ;;

    --seed)
      seed="$2"
      shift
      shift
      ;;

    --nruns)
      nruns="$2"
      shift
      shift
      ;;

    --eval_metric)
      eval_metric="$2"
      shift
      shift
      ;;

    --eval_set)
      eval_set="$2"
      shift
      shift
      ;;

    --train_mode)
      train_mode="$2"
      shift
      shift
      ;;

    --rnd_dir)
      rnd_dir="$2"
      shift
      shift
      ;;


    --help)
      print_argument_descriptions
      ;;

    -*|--*)
      additional_flags+="$1 $2 " # store additional flags
      shift # past argument
      ;;


    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done


# Check for required arguments
if [ -z "$hparams" ] || [ -z "$data_folder" ] || [ -z "$output_folder" ] || [ -z "$nsbj" ] || [ -z "$nsess" ] || [ -z "$nruns" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
fi

# Process eval_set argument
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


if [ "$rnd_dir" = True ]; then
    rnd_dirname=$(tr -dc 'a-zA-Z' < /dev/urandom | head -c 6)
    output_folder="$output_folder/$rnd_dirname"
fi

# Make sure  the output_folder is created
mkdir -p $output_folder

# Print command line arguments and save to file
{
    echo "hparams: $hparams"
    echo "data_folder: $data_folder"
    echo "cached_data_folder: $cached_data_folder"
    echo "output_folder: $output_folder"
    echo "nsbj: $nsbj"
    echo "nsess: $nsess"
    echo "seed: $seed"
    echo "nruns: $nruns"
    echo "eval_metric: $eval_metric"
    echo "eval_set: $eval_set"
    echo "train_mode: $train_mode"
    echo "rnd_dir: $rnd_dir"
    echo "additional flags: $additional_flags"
} | tee "$output_folder/flags.txt"


# Creating output folder
mkdir -p $output_folder
mkdir -p $data_folder
mkdir -p $cached_data_folder

# Function to run the training experiment
run_experiment() {
  local target_session_idx="$1"
  local output_folder_exp="$2"

  for target_subject_idx in $(seq 0 1 $(( nsbj - 1 ))); do
    echo "Subject $target_subject_idx"
    python train.py $hparams --seed=$seed --data_folder=$data_folder --cached_data_folder=$cached_data_folder --output_folder=$output_folder_exp\
      --target_subject_idx=$target_subject_idx --target_session_idx=$target_session_idx \
      --data_iterator_name="$train_mode" $additional_flags
  done
}

# Run multiple training experiments (with different seeds)
for i in $(seq 0 1 $(( nruns - 1 ))); do
  ((run_idx = i + 1))
  run_name=run"$run_idx"
  output_folder_exp="$output_folder"/"$run_name"/$seed

  if [ "$train_mode" = "leave-one-subject-out" ]; then
    run_experiment 0 $output_folder_exp

  elif [ "$train_mode" = "leave-one-session-out" ]; then
    # Loop over sessions
    for j in $(seq 0 1 $(( nsess - 1 ))); do
      run_experiment $j $output_folder_exp
    done
  else
      echo "Invalid train_model value: $train_mode. It can be leave-one-subject-out or leave-one-session-out  only."
  exit 1
  fi

  # Store the results
  python utils/parse_results.py $output_folder_exp $metric_file $eval_metric | tee -a  $output_folder/$run_name\_results.txt

  # Changing Random seed
  seed=$((seed+1))
done


echo 'Final Results (Performance Aggregation)'
python utils/aggregate_results.py $output_folder $eval_metric | tee -a  $output_folder/aggregated_performance.txt
