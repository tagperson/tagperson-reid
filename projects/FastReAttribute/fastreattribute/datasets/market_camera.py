# encoding: utf-8

import glob
import os.path as osp
import re
import warnings
import os

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

import json


@DATASET_REGISTRY.register()
class MarketCamera(ImageDataset):

    _junk_pids = [0, -1]
    dataset_dir = 'MarketCamera'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market_camera"

    def __init__(self, root='datasets', market_camera_subpath=[], **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, '')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, market_camera_subpath[0])
        self.query_dir = osp.join(self.data_dir, market_camera_subpath[1])
        self.gallery_dir = osp.join(self.data_dir, '')

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

        super(MarketCamera, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # if is_train == True and pid > 199:
            #     continue
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            # data.append((img_path, pid, camid))
            attribute_labels = [camid]
            option_labels = []
            data.append((img_path, pid, camid, attribute_labels, option_labels))
        return data
