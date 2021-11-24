# encoding: utf-8
"""
anonymous
anonymous
"""

import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PRID', ]


@DATASET_REGISTRY.register()
class PRID(ImageDataset):
    """PRID
    """
    dataset_dir = "prid_2011"
    dataset_name = 'prid'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        # self.train_path = os.path.join(self.root, self.dataset_dir, 'slim_train')
        self.train_path = os.path.join(self.root, self.dataset_dir, 'single_shot')

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        query = self.process_query(self.train_path, 'cam_a')
        gallery = self.process_gallery(self.train_path, 'cam_b')

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, train_path):
        data = []
        for root, dirs, files in os.walk(train_path):
            for img_name in filter(lambda x: x.endswith('.png'), files):
                img_path = os.path.join(root, img_name)
                pid = self.dataset_name + '_' + root.split('/')[-1].split('_')[1]
                camid = self.dataset_name + '_' + img_name.split('_')[0]
                data.append([img_path, pid, camid])
        return data

    def process_query(self, train_path, sub_dir):
        data = []
        query_dir = os.path.join(train_path, sub_dir)
        query_names = os.listdir(query_dir)
        query_names = filter(lambda x: x.endswith('.png'), query_names)

        for file_name in query_names:
            img_path = os.path.join(query_dir, file_name)
            pid = int(file_name.split('_')[1].split('.')[0])
            if pid > 200:
                continue
            camid = 0 if sub_dir == 'cam_a' else 1
            data.append([img_path, pid, camid])
        return data

    def process_gallery(self, train_path, sub_dir):
        data = []
        gallery_dir = os.path.join(train_path, sub_dir)
        gallery_names = os.listdir(gallery_dir)
        gallery_names = filter(lambda x: x.endswith('.png'), gallery_names)

        for file_name in gallery_names:
            img_path = os.path.join(gallery_dir, file_name)
            pid = int(file_name.split('_')[1].split('.')[0])
            camid = 0 if sub_dir == 'cam_a' else 1
            data.append([img_path, pid, camid])
        return data

        