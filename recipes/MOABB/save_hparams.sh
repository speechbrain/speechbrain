#!/bin/bash
###########################################################
# This script saves the hparam files based on the combination of different datasets and architecture to test.
# Three partial yaml files will be combinated depending: a first yaml file that define the dataset
# (contained in hparams/datasets/*.yaml), a second yaml file that contains common hyper-parameters (shared across
# architectures and datasets, contained in hparams/common.yaml), and a third yaml file that contains hyper-parameters
# of the model (contained in hparams/models/*yaml).
# The three partial yaml files are contained in the 'sources' directory within hparams dir (hparams_dir).
# The generated yaml files will be saved in hparams_dir.
# When the flag save_by_paradigm is True, the script will try to separate yaml files by paradigm in different
# directories (MotorImagery, P300, SSVEP), based on known dataset names found in hparams/datasets.
# In case no known datasets are in hparams/datasets, then hparam file will be saved directly in hparams_dir.
#
# Authors:
# - Davide Borra (2023)
# - Mirco Ravanelli (2023)
###########################################################

hparams_dir=hparams
save_by_paradigm=True

sources_type='' # tag for sources (default to '')
endfname=$sources_type
sources_dir=$hparams_dir'/sources'
if [ "$sources_type" != '' ]; then
  endfname='_'$sources_type
  sources_dir=$sources_dir'_'$sources_type
fi

for i in $sources_dir'/datasets/'*.yaml; # datasets loop
do
  fname_dataset="$(basename -- $i)" # getting filename
  fname_dataset="${fname_dataset%.*}" # stripping out extension from filename
  echo $fname_dataset

  save_dir=$hparams_dir'/'$fname_dataset # initialize default save_dir value
  # Try to save by paradigm.
  if [ "$save_by_paradigm" = True ]; then
    if [ "$fname_dataset" = BNCI2014001 ] || [ "$fname_dataset" = BNCI2014004 ] || \
         [ "$fname_dataset" = BNCI2015001 ] ||[ "$fname_dataset" = BNCI2015004 ] || \
         [ "$fname_dataset" = Lee2019_MI ] ||[ "$fname_dataset" = Zhou2016 ] ; then
        save_dir=$hparams_dir'/MotorImagery/'$fname_dataset
    elif [ "$fname_dataset" = bi2015a ] || [ "$fname_dataset" = BNCI2014009 ] || \
         [ "$fname_dataset" = EPFLP300 ] || [ "$fname_dataset" = Lee2019_ERP ] ; then
      save_dir=$hparams_dir'/P300/'$fname_dataset
    elif [ "$fname_dataset" = Lee2019_SSVEP ] ; then
      save_dir=$hparams_dir'/SSVEP/'$fname_dataset
    fi
  fi

  mkdir -p $save_dir

  for j in $sources_dir'/models/'*.yaml; # models loop
  do
    fname_model="$(basename -- $j)" # getting filename
    fname_model="${fname_model%.*}" # stripping out extension from filename
    echo $fname_model
    cat $i $sources_dir'/common.yaml' $j > $save_dir'/'$fname_model$endfname'.yaml'
  done
done
