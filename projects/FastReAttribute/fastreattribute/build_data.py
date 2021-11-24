# encoding: utf-8
"""
"""

import os

import torch
from torch._six import container_abcs, string_classes, int_classes

from fastreid.data import samplers
from fastreid.data.common import CommDataset
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.utils import comm
from .reattribute_dataset import ReAttributeDataset
from .samplers.hard_example_sampler import HardExampleSampler
from .samplers.background_sampler import BackgroundSampler
from .samplers.expand_triplet_sampler import ExpandNaiveIdentitySampler
from .samplers.resolution_balance_sampler import ResolutionBalanceSampler
from fastreid.data.data_utils import DataLoaderX

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reattribute_train_loader(cfg, **kwargs):
    cfg = cfg.clone()

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL, personx3d_subpath=cfg.DATASETS.PERSONX3D.SUBPATH, makehuman_subpath=cfg.DATASETS.MAKEHUMAN.SUBPATH, market_camera_subpath=cfg.DATASETS.MARKET_CAMERA.SUBPATH, cfg=cfg, **kwargs)
        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    transforms = build_transforms(cfg, is_train=True)

    train_set = ReAttributeDataset(train_items, transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()
    num_instance_in_mesh_group = cfg.DATALOADER.NUM_INSTANCE_IN_MESH_GROUP
    num_instance_in_background_group = cfg.DATALOADER.NUM_INSTANCE_IN_BACKGROUND_GROUP

    
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    
    if cfg.DATALOADER.CAMERA_BALANCE_SAMPLER:
        sampler = samplers.CameraBalanceSampler(train_set.img_items, mini_batch_size, num_instance, num_cams_per_batch)
    elif cfg.DATALOADER.BACKGROUND_SAMPLER:
        sampler = BackgroundSampler(train_set.img_items, mini_batch_size, num_instance, num_instance_in_background_group)
    elif cfg.DATALOADER.HARD_EXAMPLE_SAMPLER:
        sampler = HardExampleSampler(train_set.img_items, mini_batch_size, num_instance, num_instance_in_mesh_group)
    elif sampler_name == 'ExpandNaiveIdentitySampler':
        expand_camera_enabled = cfg.DATALOADER.EXPAND_NAIVE_IDENTITY_SAMPLER.CAMERA.ENABLED
        sampler = ExpandNaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance, expand_camera_enabled=expand_camera_enabled)
    elif sampler_name == 'ResolutionBalanceSampler':
        num_resolution_per_batch = cfg.DATALOADER.RESOLUTION_BALANCE_SAMPLER.NUM_RESOLUTION_PER_BATCH
        sampler = ResolutionBalanceSampler(train_set.img_items, mini_batch_size, num_instance, num_resolution_per_batch=num_resolution_per_batch)
    elif sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(train_set))
    elif sampler_name == "NaiveIdentitySampler":
        sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    elif sampler_name == "BalancedIdentitySampler":
        sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    elif sampler_name == "SetReWeightSampler":
        set_weight = cfg.DATALOADER.SET_WEIGHT
        sampler = samplers.SetReWeightSampler(train_set.img_items, mini_batch_size, num_instance, set_weight)
    elif sampler_name == "ImbalancedDatasetSampler":
        sampler = samplers.ImbalancedDatasetSampler(train_set.img_items)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    expand_mini_batch_size = mini_batch_size
    if cfg.DATALOADER.EXPAND_NAIVE_IDENTITY_SAMPLER.CAMERA.ENABLED:
        expand_mini_batch_size += mini_batch_size

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, expand_mini_batch_size, True)

    train_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=reattribute_batch_collator,
        pin_memory=True,
    )
    return train_loader


def build_reattribute_test_loader(cfg, dataset_name, **kwargs):
    cfg = cfg.clone()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root, personx3d_subpath=cfg.DATASETS.PERSONX3D.SUBPATH, makehuman_subpath=cfg.DATASETS.MAKEHUMAN.SUBPATH, market_camera_subpath=cfg.DATASETS.MARKET_CAMERA.SUBPATH, cfg=cfg, **kwargs)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query + dataset.gallery

    transforms = build_transforms(cfg, is_train=False)

    if cfg.TEST.ATTRIBUTE.ENABLED:
        test_set = ReAttributeDataset(test_items, transforms, relabel=True)
    else:
        test_set = CommDataset(test_items, transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=test_set,
        batch_sampler=batch_sampler,
        num_workers=4,  # save some memory
        collate_fn=reattribute_batch_collator,
        pin_memory=True,
    )
    return test_loader, len(dataset.query)
    # return test_loader



def reattribute_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: reattribute_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
    elif isinstance(elem, list):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, dict):
        return torch.tensor(batched_inputs)

