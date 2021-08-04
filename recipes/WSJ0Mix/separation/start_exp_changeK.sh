#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=1-23:59
#SBATCH --output=logs/out_%x-%j.log
#SBATCH --error=logs/err_%x-%j.log
echo 'copying the data...'
cp /home/csubakan/scratch/wsj0-2mix-8k-min.tar.gz $SLURM_TMPDIR
echo 'extracting the data...'
tar -xzf $SLURM_TMPDIR/wsj0-2mix-8k-min.tar.gz -C $SLURM_TMPDIR
echo 'starting the experiment...'
source /home/csubakan/venv_sb_fork/bin/activate
python train.py $1 --chunk_size $2 --data_folder $SLURM_TMPDIR/wsj0-mix/2speakers/ --limit_training_signal_len True --training_signal_len 32000
