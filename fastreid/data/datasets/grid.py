# encoding: utf-8
"""
anonymous
anonymous
"""

import os
from glob import glob
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['GRID', ]


@DATASET_REGISTRY.register()
class GRID(ImageDataset):
    """GRID
    """
    dataset_dir = "underground_reid"
    dataset_name = 'grid'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir, 'images')
        self.query_dir = os.path.join(self.root, self.dataset_dir, 'probe')
        self.gallery_dir = os.path.join(self.root, self.dataset_dir, 'gallery')

        required_files = [
            self.train_path,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = lambda: self.process_train(self.train_path)
        query = lambda: self.process_train(self.query_dir, is_train=False)
        gallery = lambda: self.process_train(self.gallery_dir, is_train=False)
        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, train_path, is_train=True):
        data = []
        img_paths = glob(os.path.join(train_path, "*.jpeg"))

        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            img_info = img_name.split('_')
            pid = int(img_info[0])
            camid = int(img_info[1])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))
        return data
