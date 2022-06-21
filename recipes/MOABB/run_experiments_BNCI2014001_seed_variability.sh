# Script to test seed variability in BNCI2014001
nsbj=9
nsess=2
nparallel=3
hparams='hparams/EEGNet_BNCI2014001_seed_variability.yaml'
data_folder='/path/to/MOABB_datasets'
# SEED VARIABILITY VARIABLES
nseeds=2
RANDOM=1341

# SEED VARIABILITY LOOP
for i in $(seq 0 1 $(( nseeds - 1 ))); do
  seed=$RANDOM
  echo $seed
  # LEAVE-ONE-SUBJECT-OUT
  ./train_on_subjects.sh $nsbj $nparallel $hparams $data_folder 0 'leave-one-subject-out' $seed &

done




