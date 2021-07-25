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

*We would like to advise our users that despite being more efficient, DDP is also
more prone to exhibit unexpected bugs. Indeed, DDP is quite server dependent and some setups might generate errors with the PyTorch implementation of DDP. If you encounter an issue, please report it on our github with as much information as possible. Indeed, DDP bugs are very challenging to replicate ...*

The common pattern for using multi-GPU training with DDP (on a single machine with 4 GPUs):
```
cd recipes/<dataset>/<task>/
python -m torch.distributed.launch --nproc_per_node=4 experiment.py hyperparams.yaml --distributed_launch --distributed_backend='nccl'
```
Try to switch the DDP backend if you have issues with `nccl`.

To using DDP, you should consider using `torch.distributed.launch` for setting the subprocess with the right Unix variables `local_rank` and `rank`. The `local_rank` variable allows setting the right `device` argument for each DDP subprocess, while the `rank` variable (which is unique for each subprocess) will be used for registering the subprocess rank to the DDP group. In that way, **we can manage multi-GPU training over multiple machines**.

### With multiple machines (suppose you have 2 servers with 2 GPUs):
```
# Machine 1
cd recipes/<dataset>/<task>/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node=0 --master_addr machine_1_adress --master_port 5555 experiment.py hyperparams.yaml --distributed_launch --distributed_backend='nccl'

# Machine 2
cd recipes/<dataset>/<task>/
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node=1 --master_addr machine_1_adress --master_port 5555 experiment.py hyperparams.yaml --distributed_launch --distributed_backend='nccl'
```
Machine 1 will have 2 subprocess (subprocess1: with `local_rank=0`, `rank=0`, and subprocess2: with `local_rank=1`, `rank=1`).
Machine 2 will have 2 subprocess (subprocess1: with `local_rank=0`, `rank=2`, and subprocess2: with `local_rank=1`, `rank=3`).

In this way, the current DDP group will contain 4 GPUs.
