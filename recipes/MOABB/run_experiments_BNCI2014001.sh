#!/usr/bin/env bash
# Script to run experiments on BNCI2014001 dataset with parallel processes
nsbj=9 #total number of subjects
nsess=2 #total number of sessions
nparallel=3 #3 processes in parallel
hparams='hparams/EEGNet_BNCI2014001.yaml' #hparams file
data_folder='/path/to/MOABB_datasets' #data folder
seed=1234 #override seed

# LEAVE-ONE-SUBJECT-OUT
./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder 0 'leave-one-subject-out' $seed
wait
# LEAVE-ONE-SESSION-OUT
for j in $(seq 0 1 $(( nsess - 1 ))); do
  ./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder $j 'leave-one-session-out' $seed
  wait
done