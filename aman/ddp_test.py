import torch.distributed as dist
import torch
import os

# torchrun sets these; init_process_group reads them from the environment automatically
dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

# Set the correct GPU for this process
torch.cuda.set_device(local_rank)

print(f"Hello from rank {rank} (local rank {local_rank}) out of {world_size} processes")