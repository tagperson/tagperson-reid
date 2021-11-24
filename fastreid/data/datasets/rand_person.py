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
class RandPerson(ImageDataset):
    """RandPerson.

    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = ''
    dataset_name = "rand_person"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'RandPerson')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'randperson_subset')
        self.query_dir = osp.join(self.data_dir, 'randperson_subset')
        self.gallery_dir = osp.join(self.data_dir, 'randperson_subset')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        super(RandPerson, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_s([\d]+)_c([\d]+)')
        data = []
        for img_path in img_paths:
            pid, _, camid = map(int, pattern.search(img_path).groups())
            if pid > 342:
                continue
            if pid == -1:
                continue  # junk images are just ignored
            # camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
