import torch.distributed as dist
import os
import torch

import torch
import torch.distributed as dist
import itertools

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def model(n):
    return [n + 0.1, n + 0.2, n + 0.3]

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    print(f"Rank {dist.get_rank()} is using GPU {torch.cuda.current_device()}")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    dataset = list(range(5))
    sampler = InferenceSampler(len(dataset))
    local_data = [dataset[i] for i in sampler]
    outputs = [model(data) for data in local_data]

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]
    torch.distributed.barrier()
    # 在进程0中打印结果
    if local_rank == 0:
        for output in merged_outputs:
            print(output)



if __name__ == "__main__":
    main()
    pass
