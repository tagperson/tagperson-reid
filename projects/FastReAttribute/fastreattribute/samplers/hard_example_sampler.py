import copy
import itertools
from collections import defaultdict
from typing import Optional
from torch.utils.data.sampler import Sampler
import numpy as np

from fastreid.utils import comm
from fastreid.data.samplers.triplet_sampler import reorder_index



class HardExampleSampler(Sampler):
    """
    """

    def __init__(self, data_source: str, mini_batch_size: int, num_instances: int, num_instances_in_mesh_group:int, seed: Optional[int] = None):
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_instances_in_mesh_group = num_instances_in_mesh_group
        self.num_pids_per_batch = mini_batch_size // self.num_instances

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.batch_size = mini_batch_size * self._world_size

        self.pid_index = defaultdict(list)


        for index, info in enumerate(data_source):
            pid = info[1]
            self.pid_index[pid].append(index)

        # record mesh
        self.pid_to_mesh_group_id_map = {}
        for index, info in enumerate(data_source):
            pid = info[1]
            option_labels = info[4]
            mesh_id = option_labels['mesh_id']
            mesh_group_id = self._calculate_mesh_group_id(mesh_id)
            self.pid_to_mesh_group_id_map[pid] = mesh_group_id
        
        self.mesh_group_dict = defaultdict(list)
        for pid, mesh_group_id in self.pid_to_mesh_group_id_map.items():
            self.mesh_group_dict[mesh_group_id].append(pid)


        self.pids = sorted(list(self.pid_index.keys()))
        self.mesh_group_ids = sorted(list(self.mesh_group_dict.keys()))
        self.num_identities = len(self.pids)
        self.num_mesh_group = len(self.mesh_group_ids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            avl_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}
            
            # use mesh_group_id
            avl_mesh_group_ids = copy.deepcopy(self.mesh_group_ids)
            batch_pids_dict = {}
 
            batch_indices = []
            # while len(avl_pids) >= self.num_pids_per_batch:
            while len(avl_pids) >= self.num_pids_per_batch and len(avl_mesh_group_ids) >= self.num_pids_per_batch // self.num_instances_in_mesh_group:
                # selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
                selected_mesh_group_ids = np.random.choice(avl_mesh_group_ids, self.num_pids_per_batch // self.num_instances_in_mesh_group, replace=False).tolist()
                
                # print(f"avl_mesh_group_ids={avl_mesh_group_ids}")
                # print(f"selected_mesh_group_ids={selected_mesh_group_ids}")
                selected_pids = []
                for mesh_group_id in selected_mesh_group_ids:
                    # print(f"mesh_group_id={mesh_group_id}, self.mesh_group_dict[mesh_group_id]={self.mesh_group_dict[mesh_group_id]}")
                    if mesh_group_id not in batch_pids_dict:
                        m_pids = copy.deepcopy(self.mesh_group_dict[mesh_group_id])
                        if len(m_pids) < self.num_instances_in_mesh_group:
                            m_pids = np.random.choice(m_pids, size=self.num_instances_in_mesh_group, replace=True).tolist()
                        np.random.shuffle(m_pids)
                        batch_pids_dict[mesh_group_id] = m_pids
                    
                    avl_pids = batch_pids_dict[mesh_group_id]
                    # selected_pids.extend(self.mesh_group_dict[mesh_group_id])
                    for _ in range(self.num_instances_in_mesh_group):
                        selected_pids.append(avl_pids.pop(0))
                
                # print(f"selected_pids={selected_pids}")

                for pid in selected_pids:
                    # Register pid in batch_idxs_dict if not
                    if pid not in batch_idxs_dict:
                        idxs = copy.deepcopy(self.pid_index[pid])
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                        np.random.shuffle(idxs)
                        batch_idxs_dict[pid] = idxs
                        # print(f"pid={pid}, batch_idxs_dict[pid]={batch_idxs_dict[pid]}")

                    avl_idxs = batch_idxs_dict[pid]
                    # print(f"pid={pid}, avl_idx={avl_idxs}")
                    for _ in range(self.num_instances):
                        batch_indices.append(avl_idxs.pop(0))

                    # mesh relative
                    mesh_group_id = self.pid_to_mesh_group_id_map[pid]
                    if len(avl_idxs) < self.num_instances: 
                        avl_pids.remove(pid)
                        batch_pids_dict[mesh_group_id].remove(pid)
                        if len(batch_pids_dict[mesh_group_id]) < self.num_instances_in_mesh_group:
                            if mesh_group_id in avl_mesh_group_ids:
                                avl_mesh_group_ids.remove(mesh_group_id)

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self._world_size)
                    batch_indices = []

    def _calculate_mesh_group_id(self, mesh_id):
        # TODO, modify 656 this magic number
        if mesh_id <= 656:
            mesh_group_id = mesh_id
        else:
            mesh_group_id = (mesh_id - 656 - 1) // 3 + 1
        return mesh_group_id