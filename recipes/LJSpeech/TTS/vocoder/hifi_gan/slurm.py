import random
import argparse
from simple_slurm import Slurm


EMOJI = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐸', '🐵', '🐔', '🐧', '🐦', '🐤', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🐛', '🦋', '🐌', '🐞', '🐜', '🦟', '🦗', '🕷', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀', '🐠']

parser = argparse.ArgumentParser(description='Process cfg file')
parser.add_argument("-c", "--cfg", type=str, required=True, help="Put this argument to run sbatch")
args = parser.parse_args()

slurm = Slurm()
slurm.add_arguments(ntasks='1')
slurm.add_arguments(cpus_per_task='16')
slurm.add_arguments(partition='gpu')
slurm.add_arguments(gpus_per_node='rtx_3090:1')
#slurm.add_arguments(gpus_per_node='1')
slurm.add_arguments(job_name=random.choice(EMOJI))
slurm.add_arguments(output=r'results/slurm/%j.out')
slurm.add_arguments(mem='32G')
slurm.add_arguments(time='300:00:00')

slurm.sbatch(f'python train.py --device=cuda:0 --max_grad_norm=1.0 {args.cfg}')