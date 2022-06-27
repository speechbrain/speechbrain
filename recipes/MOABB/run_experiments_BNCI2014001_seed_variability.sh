# Script to test seed variability in BNCI2014001
nsbj=9 #number of subjects
nsess=2 #number of sessions
nparallel=3 #number of parallel processes
hparams='hparams/EEGNet_BNCI2014001_seed_variability.yaml'
data_folder='/path/to/MOABB_datasets' #MOABB data folder
nseeds=50 #number of seeds
RANDOM=1341

# SEED LOOP
for i in $(seq 0 1 $(( nseeds - 1 ))); do
  seed=$RANDOM
  echo $seed

  # LEAVE-ONE-SUBJECT-OUT
  ./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder 0 'leave-one-subject-out' $seed
  wait

  # LEAVE-ONE-SESSION-OUT
  for j in $(seq 0 1 $(( nsess - 1 ))); do
    ./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder $j 'leave-one-session-out' $seed
    wait
  done
  wait

  # PARSE RESULTS
  python parse_results.py "results/MOABB/EEGNet_BNCI2014001_seed_variability/$seed" acc loss f1
done