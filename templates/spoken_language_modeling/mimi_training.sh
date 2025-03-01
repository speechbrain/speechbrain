#!/bin/bash
#SBATCH --job-name=mimi
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G  
#SBATCH --ntasks-per-node=2
#SBATCH --time=12:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --output=logs/mimi_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am3303@cam.ac.uk
#SBATCH --array=0-5%1
module load python/3.12.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# flash attn is not compatible with v100 :/
# pip install flash-attn
# pip install -U torch torchaudio
pip install kaldiio
cd $HOME/proj/speechbrain/speechbrain
pip install -r requirements.txt
pip install -e . 
pip install triton
pip install wandb-core
pip install wandb==0.18.0

mkdir -p $SLURM_TMPDIR/mimi_libriheavy/save/
scp -r /scratch/adelmou/tokens_extracted/mimi/libriheavy $SLURM_TMPDIR/mimi_libriheavy/save/

ls -l $SLURM_TMPDIR/mimi/save/libriheavy
scp  $SCRATCH/train.csv $SLURM_TMPDIR/
scp  $SCRATCH/dev-dev.csv $SLURM_TMPDIR/
scp  $SCRATCH/test-clean.csv $SLURM_TMPDIR/
scp  $SCRATCH/test-other.csv $SLURM_TMPDIR/

cd /home/adelmou/proj/speechbrain/speechbrain/templates/spoken_language_modeling

export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname)  # Use the current node's hostname as MASTER_ADDR
export MASTER_PORT=3456

echo "Node ID: $SLURM_NODEID, Master: $MASTER_ADDR"
echo "Launching Python script..."

# Compute the total world size for distributed training
export WORLD_SIZE=$(( SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES ))

srun python train.py hparams/mimi.yaml --data_path $SLURM_TMPDIR --tokens_folder $SLURM_TMPDIR/mimi_libriheavy/save/libriheavy --experiment_name mimi_libriheavy
