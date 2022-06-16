# Sample script to run experiments on BNCI2014001 dataset with parallel processes
nparallel=6
data_folder='/path/to/MOABB_datasets'
# LEAVE-ONE-SUBJECT-OUT
(
for target_subject_idx in {0..8}; do
   ((i=i%nparallel)); ((i++==0)) && wait
        python3 train.py 'EEGNet_BNCI2014001.yaml' --data_folder $data_folder \
        --target_subject_idx $target_subject_idx --data_iterator_name 'leave-one-subject-out'
done
)
# LEAVE-ONE-SESSION-OUT
for target_session_idx in {0..1}; do

(
for target_subject_idx in {0..8}; do
   ((i=i%nparallel)); ((i++==0)) && wait
        python3 train.py 'EEGNet_BNCI2014001.yaml' --data_folder $data_folder \
        --target_subject_idx $target_subject_idx --target_session_idx $target_session_idx \
        --data_iterator_name 'leave-one-session-out'
done
)
wait

done

