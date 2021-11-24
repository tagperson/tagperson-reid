# encoding: utf-8
"""
anonymous
anonymous
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class UnrealPerson(ImageDataset):
    """UnrealPerson.

    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "unreal_person"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, 'UnrealPerson')

        self.train_dir = osp.join(self.data_dir, 'unrealperson')
        self.train_file = osp.join(self.train_dir, 'list_unreal_train.txt')
        self.query_dir = osp.join(self.data_dir, 'unrealperson')
        self.gallery_dir = osp.join(self.data_dir, 'unrealperson')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.train_file)
        query = self.process_dir(self.query_dir, self.train_file, is_train=False)
        gallery = self.process_dir(self.gallery_dir, self.train_file, is_train=False)
        super(UnrealPerson, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, train_file, is_train=True):
        with open(train_file) as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            (img_path, pid, camid) = line.replace("\r", "").replace("\n", "").strip().split(" ")
            img_path = osp.join(self.train_dir, img_path)

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((img_path, pid, camid))

        return data
