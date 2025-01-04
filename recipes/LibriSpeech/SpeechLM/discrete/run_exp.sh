#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-9
#SBATCH --time=10:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=dac_16khz_ls
#SBATCH --output=logs/dac_16khz_ls_%A_%a.out # Customize your output filename
#SBATCH --error=logs/dac_16khz_ls_%A_%a.err   # Customize your error filename

module load gcc arrow/17.0.0
source $HOME/gslms/bin/activate

# Copy dataset to SLURM temporary directory
echo "Copying dataset to SLURM_TMPDIR..."
scp -r $HOME/projects/def-ravanelm/datasets/librispeech $SLURM_TMPDIR/
echo "Dataset copied successfully."

cd $SLURM_TMPDIR/

# Extract dataset
echo "Extracting dataset..."
for file in librispeech/*.tar.gz; do
    tar -zxf $file
done
echo "Dataset extracted."


cd /home/adelmou/proj/speechbrain/gslms/speechbrain/recipes/LibriSpeech/SpeechLM/discrete

python extract.py hparams/dac.yaml \
    --data_folder $SLURM_TMPDIR/LibriSpeech/ --model_type 16khz --num_codebooks 12 --sample_rate 16000 \
    --precision bf16 --batch_size 1 --num_shards 10 --rank $SLURM_ARRAY_TASK_ID \
    --save_folder $SCRATCH/results/dac