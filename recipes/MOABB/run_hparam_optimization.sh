#!/bin/bash

###########################################################
# Hyperparameter Tuning Script for EEG Model with Orion
###########################################################

# Description:
# This script facilitates hyperparameter tuning for a given EEG model and dataset using Orion.
# It supports leave-one-subject-out and/or leave-one-session-out training strategies.

# Usage:
# ./run_hparam_optimization.sh --exp_name 'EEGNet_BNCI2014001_hopt' \
#                             --output_folder results/MotorImagery/BNCI2014001/EEGNet/hopt \
#                             --data_folder eeg_data/ \
#                             --hparams hparams/MotorImagery/BNCI2014001/EEGNet.yaml \
#                             --nsbj 9 --nsess 2 \
#                             --nruns 1 --nruns_eval 10 \
#                             --eval_metric acc \
#                             --train_mode leave-one-session-out \
#                             --exp_max_trials 50 \
#                             --store_all True \
#                             --device 'cpu'
#
# Optimization Steps:
# The script supports multiple hyperparameter optimization steps.
# We found it convenient to first optimize training and model hyperparameters,
# and then optimize data augmentation hyperparameters in a separate step.

# Script Workflow:
# 1. Search for the orion flags in the specified hparam file.
# 2. Run the orion-hunt command for hyperparameter tuning.
#    By default, TPE (Tree-structured Parzen Estimator) hyperparameter tuning is
#    performed, as specified in the default orion config file at hparams/orion/hparams_tpe.yaml.
# 3. Save the best hyperparameters, which can be viewed using torch-info.
# 4. Loop until flags like @orion_step<stepid> are found in the YAML file.
#
# Final Performance Evaluation:
# At the end of the optimization process, the script computes the final performance
# using the best hyperparameters on the test set.
# This is done by averaging over nruns_eval different seeds.
#
# Note: More detailed information can be found in the README.md file.

# Authors:
# - Mirco Ravanelli (2023)
# - Davide Borra (2023)
###########################################################


# Initialize variables
exp_name=""
output_folder=""
data_folder=""
cached_data_folder=""
hparams=""
nsbj=""
nsess=""
nruns=""
nruns_eval=10
eval_metric="acc"
train_mode="leave-one-session-out"
seed=1986
config_file="hparams/orion/hparams_tpe.yaml"
mne_dir=""
orion_db_address=""
orion_db_type="PickledDB"
exp_max_trials=50
store_all=False
compress_exp=False

# Function to print argument descriptions and exit
print_argument_descriptions() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --exp_name Name                       Name that Orion gives to the experiment"
    echo "  --output_folder output_path           Output folder were the results will be stored"
    echo "  --data_folder data_path               Folder were the data are stored. If not available, they will be downloaded there."
    echo "  --cached_data_folder path [Optional]  Folder were the data in pkl format will be cached."
    echo "  --hparms hparam_file                  YAML file containing the hparam to optimize. The hyperparameters decorated with @orion_step1 or @orion_step1 in the YAML file will be used"
    echo "  --nsbj num_subjects                   Number of subjects"
    echo "  --nsess num_sessions                  Number of sessions"
    echo "  --nruns num_runs                      Number of runs for each hparam selection."
    echo "  --nruns_eval num_runs                 Number of runs for the final evaluation  (with best hparams) on the test set"
    echo "  --eval_metric metric [Optional]       Evaluation metric description. Default:acc"
    echo "  --seed random_seed [Optional]         Seed (random if not specified)"
    echo "  --train_mode mode [Optional]          The training mode can be leave-one-subject-out or leave-one-session-out. Default: leave-one-session-out"
    echo "  --config_file config_file [Optional]  Orion config file. Default: hparams/orion/hparams_tpe.yaml"
    echo "  --mne_dir mne_dir [Optional]          MNE directory. Need it different from your home (see notes on MNE in README.md)"
    echo "  --orion_db_address [Optional]         Path of the database where orion will store hparams and performance"
    echo "  --orion_db_type db_type [Optional]    Type of the dataset that orion will use. Default: PickledDB"
    echo "  --exp_max_trials int [Optional]       Maximum number of hparam trials for each oprimization step. Default:50"
    echo "  --store_all Bool [Optional]           When set to True, the output folders of all hparam trials will be stored in randomly named folders. Default: False"
    echo "  --compress_exp Bool [Optional]        When set to True, this option compresses the output folders of all hyperparameter trials into a single tar.gz file. This is particularly useful when store_all is set to True, as it helps prevent the accumulation of a large number of files. Default: False"
    exit 1
}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in

    --exp_name)
      exp_name="$2"
      shift
      shift
      ;;

    --output_folder)
      output_folder="$2"
      shift
      shift
      ;;

    --data_folder)
      data_folder="$2"
      shift
      shift
      ;;

    --hparams)
      hparams="$2"
      shift
      shift
      ;;

    --cached_data_folder)
      cached_data_folder="$2"
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

    --nruns_eval)
      nruns_eval="$2"
      shift
      shift
      ;;


    --eval_metric)
      eval_metric="$2"
      shift
      shift
      ;;

    --train_mode)
      train_mode="$2"
      shift
      shift
      ;;

    --config_file)
      config_file="$2"
      shift
      shift
      ;;

    --mne_dir)
      mne_dir="$2"
      shift
      shift
      ;;

    --orion_db_address)
      orion_db_address="$2"
      shift
      shift
      ;;

    --orion_db_type)
      orion_db_type="$2"
      shift
      shift
      ;;

    --exp_max_trials)
      exp_max_trials="$2"
      shift
      shift
      ;;

    --store_all)
      store_all="$2"
      shift
      shift
      ;;

    --compress_exp)
      compress_exp="$2"
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
if [ -z "$exp_name" ] || [ -z "$output_folder" ] || [ -z "$data_folder" ]  || [ -z "$hparams" ] || [ -z "$nsbj" ] || [ -z "$nsess" ] || [ -z "$nruns" ]; then
    echo "ERROR: Missing required arguments! Please provide all required options."
    print_argument_descriptions
fi

# Set mne_dir if specified
if [ "$mne_dir" ]; then
   export _MNE_FAKE_HOME_DIR=$mne_dir
fi

# Assign default value to cached_data_folder
if [ -z "$cached_data_folder" ]; then
    cached_data_folder="$data_folder/pkl"
fi


# Set orion db address if specified
if [ -z "$orion_db_address" ]; then
    orion_db_address=$output_folder'/'$exp_name'.pkl'
fi
export ORION_DB_ADDRESS=$orion_db_address
export ORION_DB_TYPE=$orion_db_type

echo "-------------------------------------"
echo "Experiment Name: $exp_name"
echo "Output Folder: $output_folder"
echo "Data Folder: $data_folder"
echo "Cached Data Folder: $cached_data_folder"
echo "Hparam File: $hparams"
echo "Number of Subjects: $nsbj"
echo "Number of Sessions: $nsess"
echo "Number of Runs: $nruns"
echo "Number of Eval Runs: $nruns_eval"
echo "Eval Metric: $eval_metric"
echo "Training modality: $train_mode"
echo "Seed: $seed"
echo "Additional Flags: $additional_flags"
echo "Orion Config File: $config_file"
echo "Orion Database type: $orion_db_type"
echo "Orion Database file: $orion_db_address"
echo "Experiment Max Trials: $exp_max_trials"
echo "-------------------------------------"


# This function will extract all the optimization flags added in the yaml file
# The input is a text file (e.g, a yaml file) and a pattern (e.g, "@orion_step1:")
# The ouput are the detected flags (e.g.,  --dropout~"uniform(0.0, 0.5)").
get_flag() {
    local file_path="$1"
    local pattern="$2"

    # Check if the file exists
    if [ ! -f "$file_path" ]; then
        echo "Error: File '$file_path' not found."
        return 1
    fi

    # Use grep to find all lines containing the pattern and then extract the flags using sed
    grep -o "$pattern.*" "$file_path" | sed "s/$pattern//" | tr -d '\n'
}


# Function for updatading the hparam yaml file with the best hparams found at step 1
update_hparams() {
    local best_hparams_file="$1"
    local hparams_yaml_file="$2"
    local output_yaml_file="$3"

    # Read the values from best_hparams.txt into an associative array
    declare -A best_hparams
    while IFS=": " read -r key value; do
        best_hparams["$key"]=$value
    done < "$best_hparams_file"


    # Read the hparams.yaml file into a variable
    local hparams_content=$(cat "$hparams_yaml_file")

    # Update values in hparams_content using values from best_hparams
    for key in "${!best_hparams[@]}"; do
        local pattern="^$key: .*"
        local replacement="$key: ${best_hparams[$key]}"
        hparams_content=$(sed "s/$pattern/$replacement/g" <<< "$hparams_content")
    done

    # Write the updated content to a new YAML file
    echo "$hparams_content" > "$output_yaml_file"
}

# Function for extracting the best hparams from orion-info
function extract_best_params() {
    local input_file="$1"
    local best_trial_line=$(grep -n "best trial:" "$input_file" | cut -d ":" -f 1)
    local params_lines=$(tail -n +$best_trial_line "$input_file" | awk '/params:/{flag=1;next}/start time:/{flag=0}flag')
    local formatted_params=$(echo "$params_lines" | sed -e 's/^[[:space:]]*//' -e 's/: /: /' -e '/^$/d' -e 's#^/##')
    echo "$formatted_params"
}

# Running hparam tuning (loop over multiple steps)
step_id=1
hparams_step=$hparams
pattern="@orion_step1:"
opt_flags=$(get_flag "$hparams_step" "$pattern")

# Check if the string is empty and exit with an error if it is
if [ -z "$opt_flags" ]; then
    echo "Error: Optimization flags not found in '$hparams'"
    echo "Please ensure that the Orion optimization flags are set in the hparam file using in-line comments like:"
    echo "# @orion_step1: --dropout~\"uniform(0.0, 0.5)\""
    exit 1  # Exit with a non-zero error code
fi


while [ -n "$opt_flags" ]; do
    # Do something
    output_folder_step="$output_folder"/step"$step_id"
    mkdir -p $output_folder_step
    exp_name_step="$exp_name"_step"$step_id"

    echo
    echo "**********************************************************************************************"
    echo "Running hparam tuning (step $step_id)..."
    echo "- This might take several hours!"
    echo "- The best set of hparams will be save in $output_folder_step"
    echo "- You can monitor the evolution of the hparam optimization with: orion status -n $exp_name"
    echo "......"
    echo "**********************************************************************************************"
    echo
    # Setting up orion command
    orion_hunt_command="orion hunt -n $exp_name_step -c $config_file --exp-max-trials $exp_max_trials \
    	./run_experiments.sh --hparams $hparams_step --data_folder $data_folder --seed $seed \
    	--output_folder $output_folder_step/exp  --nsbj $nsbj --nsess $nsess --nruns $nruns \
    	--eval_metric $eval_metric --eval_set dev --train_mode $train_mode --rnd_dir $store_all $additional_flags"


    # Appending the optimization flags
    orion_hunt_command="$orion_hunt_command $opt_flags"

    echo $orion_hunt_command &> "$output_folder_step/orion_hunt_command.txt"

    # Execute the command for hparm tuning
    eval $orion_hunt_command

    # Compress the exp folder (if required)
    if [ "$compress_exp" = True ]; then
        tar -czf "$output_folder_step/exp.tar.gz" "$output_folder_step/exp"
        if [ -d "$output_folder_step/exp" ]; then
            rm -rf "$output_folder_step/exp"
        fi

    fi

    # Storing best haprams
    orion info --name $exp_name_step &> $output_folder_step/orion-info.txt

    # Extract list of the best hparams from orion-info
    # Find the line number where "best trial:" appears
    best_trial_line=$(grep -n "best trial:" $output_folder_step/orion-info.txt | cut -d ":" -f 1)

    # Extract and store the best set of hparams
    best_params_output=$(extract_best_params "$output_folder_step/orion-info.txt")
    best_hparams_file="$output_folder_step/best_hparams.txt"
    echo "$best_params_output" > $best_hparams_file

    # Store the current best yaml file
    best_yaml_file="$output_folder_step/best_hparams.yaml"
    update_hparams "$best_hparams_file" "$hparams_step" "$best_yaml_file"

    # Update best hparam step
    hparams_step=$best_yaml_file

    # Update step variable
    ((step_id++))

    # Update search pattern
    pattern="@orion_step$step_id:"

    # update optimization flags pattern
    opt_flags=$(get_flag "$hparams_step" "$pattern")
done

echo
echo "**********************************************************************************************"
echo "Running Final Evaluation on the best hparams (test-set)..."
echo "**********************************************************************************************"
echo

final_yaml_file="$output_folder/best_hparams.yaml"
scp $best_yaml_file $final_yaml_file

# Running evaluation on the test set for the best models
 ./run_experiments.sh --hparams $final_yaml_file --data_folder $data_folder \
  --seed $seed --output_folder $output_folder/best --nsbj $nsbj --nsess $nsess \
  --nruns $nruns_eval --eval_metric $eval_metric --eval_set test \
  --train_mode $train_mode --rnd_dir $store_all $additional_flags


echo "The test performance with best hparams is available at  $output_folder/best"

