# This script loop over subjects to train in parallel deep neural networks
nsbj=$1
nparallel=$2
hparams=$3
data_folder=$4
target_session_idx=$5
data_iterator_name=$6
seed=$7

(
for target_subject_idx in $(seq 0 1 $(( nsbj - 1 ))); do
   ((i=i%nparallel)); ((i++==0)) && wait
   echo $target_subject_idx
   python train.py $hparams --seed=$seed --data_folder $data_folder \
        --target_subject_idx $target_subject_idx --target_session_idx $target_session_idx \
        --data_iterator_name $data_iterator_name &
   #pids[${target_subject_idx}]=${!}
done
)
#for pid in ${pids[*]}; do
#    wait $pid
#done
