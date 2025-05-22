# Basics of multi-GPU

Training speed can greatly benefit from being distributed across multiple GPUs. However, even on a single machine, this is **NOT** the default. To enable multi-GPU training, we strongly recommend you use **Distributed Data Parallel** (DDP).

## Multi-GPU training using Distributed Data Parallel (DDP)

DDP implements data parallelism by spawning **one process per GPU**. DDP allows you to distribute work across GPUs **on the same machine _or_ across several machines on a network** if wanted.

When using CUDA (which we will assume in this document), PyTorch uses [NCCL](https://developer.nvidia.com/nccl) behind the scenes to synchronize everything. PyTorch documentation [further details](https://pytorch.org/docs/stable/distributed.html) distributed backends.

### Writing DDP-safe code in SpeechBrain

DDP requires your training routines to be written to be DDP-safe, because your script will be run several times concurrently (potentially across multiple machines). Stock SpeechBrain recipes will work with DDP. We also provide functionality to assist with writing DDP-safe scripts.

`run_on_main` ensures that a specific function is executed only once, in only one process, forcing other processes to wait. It is frequently used to run a dataset preparation step in recipes.

Many functions like `Brain.fit` are written to be DDP-aware. In practice, there is not a lot you need to do to make your code DDP-safe, but it is something you should keep in mind.

> **NOTE:**
> With DDP, batch size is defined for a single process/GPU. This is different from Data Parallel (DP), where batches are split according to the number of GPUs. For example, with DDP, if you specify a batch size of 16, each GPU/process will use batches of 16 regardless of how many GPUs you have.

### Single-node setup

_This covers the case where you want to split training across **multiple GPUs** on **a single machine** (node)._

Using SpeechBrain, this would look like:

```bash
cd recipes/<dataset>/<task>/
torchrun --standalone --nproc_per_node=4 experiment.py hyperparams.yaml
```

... where `nproc_per_node` is the the number of processes to spawn/GPUs to use.

### Multi-node setup

_This covers the case where you want to split training across **multiple machines** on a network, with any amount of GPUs per machine._

Note that using DDP across multiple machines introduces a **communication overhead** that might slow down training significantly, sometimes more than if you were to train on a single node! This largely depends on the network speed between the nodes.
Make sure you are actually observing any benefits from distributing the work across machines.

While DDP is more efficient than `DataParallel`, it is somewhat prone to exhibit unexpected bugs. DDP is quite server-dependent, so some setups may face issues. If you are encountering problems, make sure PyTorch is well up to date.

#### Basics & manual multi-node setup

Let's start with a simple example where a user is able to connect to each node directly. Consider that we have 2 nodes with 2 GPUs each (for a total of 4 GPUs).

We use `torchrun` once on each machine, with the following parameters:

- `--nproc_per_node=2` means we will spawn 2 processes per node, which equates to 2 GPUs per nodes.
- `--nnodes=2` means we will be using two nodes in total.
- `--node_rank=0` and `--node_rank=1` refer to the rank/"index" we are attributing to the node/machine.
- `--master_addr`/`--master_port` define the IP address and the port of the "master" machine. In this case, we're arbitrarily choosing the first machine to be the "master" of everyone else (the 2nd machine in our case). Note that `5555` might be taken by a different process if you are unlucky or if you would run multiple different training scripts on that node, so you may need to choose a different free port.

Hence, we get:

```bash
# Machine 1
cd recipes/<dataset>/<task>/
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr machine_1_address --master_port 5555 experiment.py hyperparams.yaml
```

```bash
# Machine 2
cd recipes/<dataset>/<task>/
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr machine_1_address --master_port 5555 experiment.py hyperparams.yaml
```

In this setup:

- Machine 1 will have 2 subprocesses:
    - Subprocess #1: `local_rank`=0, `rank`=0
    - Subprocess #2: `local_rank`=1, `rank`=1
- Machine 2 will have 2 subprocess:
    - Subprocess #1: `local_rank`=0, `rank`=2
    - Subprocess #2: `local_rank`=1, `rank`=3

In practice, using `torchrun` ensures that the right environment variables are set (`LOCAL_RANK` and `RANK`), so you don't have to bother with it.

#### Multi-node setup with Slurm

If you have access to a compute cluster using Slurm, you can automate this process. We will create two scripts:

- a SBATCH script that will request the node configuration and call the second script.
- a SRUN script that will call the training on each node.

`sbatch.sh`:

```bash
#SBATCH --nodes=2 # We want two nodes (servers)
#SBATCH --ntasks-per-node=1 # we will run once the next srun per node
#SBATCH --gres=gpu:4 # we want 4 GPUs per node #cspell:ignore gres
#SBATCH --job-name=SBisSOcool
#SBATCH --cpus-per-task=10 # the only task will request 10 cores
#SBATCH --time=20:00:00 # Everything will run for 20H.

# We jump into the submission dir
cd ${SLURM_SUBMIT_DIR}

# And we call the srun that will run --ntasks-per-node times (once here) per node
srun srun_script.sh
```

`srun_script.sh`:

```bash
#!/bin/bash

# We jump into the submission dir
cd ${SLURM_SUBMIT_DIR}

# We activate our env
conda activate super_cool_sb_env

# We extract the master node address (the one that every node must connects to)
LISTNODES=`scontrol show hostname $SLURM_JOB_NODELIST`
MASTER=`echo $LISTNODES | cut -d" " -f1`

# here --nproc_per_node=4 because we want torchrun to spawn 4 processes (4 GPUs). Then we give the total amount of nodes requested (--nnodes) and then --node_rank that is necessary to dissociate the node that we are calling this from.
torchrun --nproc_per_node=4 --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${SLURM_NODEID} --master_addr=${MASTER} --master_port=5555 train.py hparams/myrecipe.yaml
```

## (DEPRECATED) Single-node multi-GPU training using Data Parallel

[**We strongly recommend AGAINST using `DataParallel`, even for single-node setups**](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)! Use `DistributedDataParallel` instead. We no longer provide support for `DataParallel`. Future PyTorch versions may even remove `DataParallel` altogether.

The common pattern for using multi-GPU training over a single machine with Data Parallel is:

```bash
cd recipes/<dataset>/<task>/
python experiment.py params.yaml --data_parallel_backend
```

If you want to use a specific set of GPU devices, consider using `CUDA_VISIBLE_DEVICES` as follow:

```bash
cd recipes/<dataset>/<task>/
CUDA_VISIBLE_DEVICES=1,5 python experiment.py params.yaml --data_parallel_backend
```

Important: the batch size for each GPU process will be: `batch_size / Number of GPUs`. So you should consider changing the batch_size value according to you need.
