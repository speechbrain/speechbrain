# Basics of multi-GPU

SpeechBrain provides two different ways of using multiple gpus while training or inferring. For further information, please see our multi-gpu tutorial: amazing multi-gpu tutorial

## Multi-GPU training using Data Parallel
The common pattern for using multi-GPU training over a single machine with Data Parallel is:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml --data_parallel_backend
```
If you want to use a specific set of GPU devices, condiser using `CUDA_VISIBLE_DEVICES` as follow:
```
> cd recipes/<dataset>/<task>/
> CUDA_VISIBLE_DEVICES=1,5 python experiment.py params.yaml --data_parallel_backend
```

Important: the batch size for each GPU process will be: `batch_size / Number of GPUs`. So you should consider changing the batch_size value according to you need.

## Multi-GPU training using Distributed Data Parallel (DDP)

DDP implements data parallelism on different processes. This way, the GPUs do not necessarily have to be in the same server. This solution is much more flexible. However, the training routines must be written considering multi-threading.

With SpeechBrain, we put several efforts to make sure the code is compliant with DDP. For instance, to avoid conflicts across processes we develop the `run_on_main` function. It is called when critical operations such as writing a file on disk are performed. It ensures that these operations are run in a single process only. The other processes are waiting until this operation is completed.

Using DDP in speechbrain with a single server (node) is quite easy:

```
cd recipes/<dataset>/<task>/
python -m torch.distributed.launch --nproc_per_node=4 experiment.py hyperparams.yaml --distributed_launch --distributed_backend='nccl'
```

Where:
- nproc_per_node must be equal to the number of GPUs.
- distributed_backend is the type of backend managing multiple processes synchronizations (e.g, 'nccl', 'gloo'). Try to switch the DDP backend if you have issues with nccl.

Running DDP over multiple servers (nodes) is quite system dependent. Let's start with a simple example where a user is able to connect to each node directly. If we want to run 2 GPUs on 2 different nodes (i.e total of 4 GPUs), we must do:

```shell
# Machine 1
cd recipes/<dataset>/<task>/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr machine_1_adress --master_port 5555 experiment.py hyperparams.yaml --distributed_launch --distributed_backend='nccl'

# Machine 2
cd recipes/<dataset>/<task>/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr machine_1_adress --master_port 5555 experiment.py hyperparams.yaml --distributed_launch --distributed_backend='nccl'
```

In this case, Machine 1 will have 2 subprocesses (subprocess1: with local_rank=0, rank=0, and subprocess2: with local_rank=1, rank=1). Machine 2 will have 2 subprocess (subprocess1: with local_rank=0, rank=2, and subprocess2: with local_rank=1, rank=3).

In practice, using `torch.distributed.launch` ensures that the right environment variables are set (`local_rank` and `rank`), so you don't have to bother about it.

Now, let's try to scale this up a bit with a resource manager like SLURM. Here, we will create two scripts:
- a SBATCH script that will request the node configuration and call the second script.
- a SRUN script that will call the training on each node.

```shell
## sbatch.sh

#SBATCH --nodes=2 # We want two nodes (servers)
#SBATCH --ntasks-per-node=1 # we will run once the next srun per node
#SBATCH --gres=gpu:4 # we want 4 GPUs per node
#SBATCH --job-name=SBisSOcool
#SBATCH --cpus-per-task=10 # the only task will request 10 cores
#SBATCH --time=20:00:00 # Everything will run for 20H.

# We jump into the submission dir
cd ${SLURM_SUBMIT_DIR}

# And we call the srun that will run --ntasks-per-node times (once here) per node
srun srun_script.sh
```

```shell
## srun_script.sh

#!/bin/bash

# We jump into the submission dir
cd ${SLURM_SUBMIT_DIR}

# We activate our env
conda activate super_cool_sb_env

# We extract the master node address (the one that every node must connects to)
LISTNODES=`scontrol show hostname $SLURM_JOB_NODELIST`
MASTER=`echo $LISTNODES | cut -d" " -f1`

# here --nproc_per_node=4 because we want torch.distributed to spawn 4 processes (4 GPUs). Then we give the total amount of nodes requested (--nnodes) and then --node_rank that is necessary to dissociate the node that we are calling this from.
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} --master_addr=${MASTER} --master_port=5555 train.py hparams/myrecipe.yaml
```

Note that using DDP on different machines introduces a **communication overhead** that might slow down training (depending on how fast is the connection across the different machines).

We would like to advise our users that despite being more efficient, DDP is also more prone to exhibit unexpected bugs. Indeed, DDP is quite server-dependent and some setups might generate errors with the PyTorch implementation of DDP.  The future version of pytorch will improve the stability of DDP.
