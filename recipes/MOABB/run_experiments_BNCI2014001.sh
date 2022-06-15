nparallel=6
data_folder='/path/to/MOABB_datasets'
data_iterator_name='leave-one-subject-out'
(
for target_subject_idx in {0..8}; do
   ((i=i%nparallel)); ((i++==0)) && wait
        python3 train.py 'EEGNet_BNCI2014001.yaml' --data_folder $data_folder \
        --target_subject_idx $target_subject_idx --data_iterator_name $data_iterator_name
done
)