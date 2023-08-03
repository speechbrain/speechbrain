#!/bin/bash

###########################################################
# Script to run leave-one-subject-out and/or leave-one-session-out training, optionally with multiple seeds.
# This script loops over the different subjects and sessions and trains different models. 
# At the end, the final performance is computed with the aggregate_results.py script that provides the average performance.
#
# Example:
# ./run_experiments.sh --hparams=hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data \
#   --output_folder=results/MotorImagery/BNCI2014001/EEGNet --nsbj=9 --nsess=2 --seed=1986 --nruns=2 --eval_metric=acc --metric_file=valid_metrics.pkl \
#   --do_leave_one_subject_out=false --do_leave_one_session_out=true --number_of_epochs=2 --device='cpu'
#
# Please, see the README.md file for more info
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
eval_metric=""
metric_file=""
do_leave_one_subject_out=""
do_leave_one_session_out=""
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
    echo "  --metric_file=metric_file_path    File where to store the eval metrics"
    echo "  --do_leave_one_subject_out=flag   Runs leave one subject out training "
    echo "  --do_leave_one_session_out=flag   Runs one session out training"
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
        --metric_file=*) metric_file="${arg#*=}" ;;
        --do_leave_one_subject_out=*) do_leave_one_subject_out="${arg#*=}" ;;
        --do_leave_one_session_out=*) do_leave_one_session_out="${arg#*=}" ;;
        *) additional_flags+="$arg " ;;
    esac
done

# Check for required arguments
if [ -z "$hparams" ] || [ -z "$data_folder" ] || [ -z "$output_folder" ] || [ -z "$nsbj" ] || [ -z "$nsess" ] || [ -z "$nruns" ] || [ -z "$eval_metric" ] || [ -z "$metric_file" ] || [ -z "$do_leave_one_subject_out" ] || [ -z "$do_leave_one_session_out" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
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

# Creating output folder
mkdir -p $output_folder\_seed\_$seed_init
mkdir -p $data_folder
mkdir -p $cached_data_folder

# Function to run the training experiment
run_experiment() {
  local data_iterator_name="$1"
  local target_session_idx="$2"

  for target_subject_idx in $(seq 0 1 $(( nsbj - 1 ))); do
    echo "Subject $target_subject_idx"
    python train.py $hparams --seed=$seed --data_folder=$data_folder --cached_data_folder=$cached_data_folder --output_folder=$output_folder/$seed \
      --target_subject_idx=$target_subject_idx --target_session_idx=$target_session_idx \
      --data_iterator_name="$data_iterator_name" $additional_flags
  done
}

# Run multiple training experiments (with different seeds)
for i in $(seq 0 1 $(( nruns - 1 ))); do
  echo $seed

  # LEAVE-ONE-SUBJECT-OUT
  if [ "$do_leave_one_subject_out" = true ]; then
    run_experiment "leave-one-subject-out" 0
  fi

  # LEAVE-ONE-SESSION-OUT
  if [ "$do_leave_one_session_out" = true ]; then
    # Loop over sessions
    for j in $(seq 0 1 $(( nsess - 1 ))); do
      run_experiment "leave-one-session-out" $j
    done
  fi


  # Store the results
  python utils/parse_results.py $output_folder/$seed $metric_file $eval_metric | tee -a  $output_folder/$seed\_results.txt

  # Changing Random seed
  seed=$((seed+1))
done


echo 'Final Results (Performance Aggregation)'
python utils/aggregate_results.py $output_folder $eval_metric | tee -a  $output_folder/aggregated_performance.txt

