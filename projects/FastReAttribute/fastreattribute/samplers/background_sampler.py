import copy
import itertools
from collections import defaultdict
from typing import Optional
from torch.utils.data.sampler import Sampler
import numpy as np

from fastreid.utils import comm
from fastreid.data.samplers.triplet_sampler import reorder_index



class BackgroundSampler(Sampler):
    """
    """

    def __init__(self, data_source: str, mini_batch_size: int, num_instances: int, num_instances_in_background_group:int, seed: Optional[int] = None):
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_instances_in_background_group = num_instances_in_background_group
        print(f"num_instances_in_background_group={num_instances_in_background_group}")
        self.num_pids_per_batch = mini_batch_size // self.num_instances

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.batch_size = mini_batch_size * self._world_size

        self.pid_index = defaultdict(list)


        for index, info in enumerate(data_source):
            pid = info[1]
            self.pid_index[pid].append(index)

        # record background
        self.background_dict = defaultdict(list)
        for index, info in enumerate(data_source):
            pid = info[1]
            img_path = info[0]
            option_labels = info[4]
            background = option_labels['background']
            self.background_dict[background].append((pid, index, option_labels, img_path))

        self.pids = sorted(list(self.pid_index.keys()))
        self.background_list = sorted(list(self.background_dict.keys()))
        self.num_identities = len(self.pids)
        self.num_background = len(self.background_list)

        print(f"num_background: {self.num_background}")
        print(f"num_pids_per_batch: {self.num_pids_per_batch}")
        # print(f"background_dict sample: {self.background_dict[self.background_list[0]]}")
        background_num = self.batch_size // self.num_instances_in_background_group // self.num_instances
        print(f"background_num={background_num}")
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        print(f"self.batch_size={self.batch_size}")
        print(f"self.num_instances_in_background_group={self.num_instances_in_background_group}")
        print(f"self.num_instances={self.num_instances}")
        background_num = self.batch_size // self.num_instances_in_background_group // self.num_instances
        batch_indices = []
        print(f"background_num={background_num}")

        while True:
            cur_background_list = np.random.choice(self.background_list, background_num, replace=False).tolist()
            for background in cur_background_list:
                background_sample_list = self.background_dict[background]
                # print(f"background: {background}")
                # print(f"length of background_sample_list: {len(background_sample_list)}")
                sampled_idx = np.random.choice(range(0, len(background_sample_list)), self.num_instances_in_background_group, replace=True).tolist()
                sample_list = []
                for idx in sampled_idx:
                    sample_list.append(background_sample_list[idx])
                for sample in sample_list:
                    pid, index, option_labels, img_path = sample
                    # print(f"w={option_labels['img_width']},h={option_labels['img_height']}, path={img_path}")
                    batch_indices.append(index)
                    idxs = self.pid_index[pid]
                    idxs = np.random.choice(idxs, size=self.num_instances - 1, replace=True).tolist()
                    batch_indices.extend(idxs)

            if len(batch_indices) == self.batch_size:
                yield from reorder_index(batch_indices, self._world_size)
                batch_indices = []
