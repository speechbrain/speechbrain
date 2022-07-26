# Script to test seed variability
# Example:
# nohup ./run_experiments_seed_variability.sh 'hparams/EEGNet_EPFLP300_seed_variability.yaml' '/path/to/MOABB/data' \
# 'results/MOABB/EEGNet_EPFLP300_seed_variability' 8 8 4 1431 0
hparams=$1
data_folder=$2 #MOABB data folder
output_folder=$3 #results folder
nparallel=$4 #number of parallel processes
nsbj=$5
nsess=$6
random_seed=$7
cuda_worker=$8

echo "hparams file : $hparams"
echo "Data folder: $data_folder"
echo "Output folder: $output_folder"
echo "No. of parallel processes : $nparallel"
echo "No. of subjects: $nsbj"
echo "No. of sessions: $nsess"
nseeds=50 #number of seeds
RANDOM=$random_seed #1341

# SEED LOOP
for i in $(seq 0 1 $(( nseeds - 1 ))); do
  seed=$RANDOM
  echo $seed

  # LEAVE-ONE-SUBJECT-OUT
  ./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder 0 'leave-one-subject-out' $seed $cuda_worker
  wait

  # LEAVE-ONE-SESSION-OUT
  for j in $(seq 0 1 $(( nsess - 1 ))); do
    ./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder $j 'leave-one-session-out' $seed $cuda_worker
    wait
  done
  wait

  # PARSE RESULTS
  python parse_results.py "$output_folder/$seed" acc loss f1
done