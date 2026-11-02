#!/usr/bin/env python3
"""最小 NCCL all-reduce 探针：torchrun --standalone --nproc_per_node=2。"""
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
x = torch.ones(4, device=f"cuda:{local_rank}") * (rank + 1)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"NCCL_OK world={world} all_reduce_sum={x.tolist()}", flush=True)
dist.barrier()
dist.destroy_process_group()
