# encoding: utf-8

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PersonX3DGenerated(ImageDataset):
    """PersonX3DGenerated.
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = None
    dataset_name = "PersonX"

    def __init__(self, root='datasets', personx3d_subpath='', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'PersonX3D/')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "1/bounding_box_train" under '
                          '"PersonX_v1".')

        self.train_dir = osp.join(self.data_dir, personx3d_subpath)
        self.query_dir = osp.join(self.data_dir, personx3d_subpath)
        self.gallery_dir = osp.join(self.data_dir, personx3d_subpath)

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

        super(PersonX3DGenerated, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # TODO: remove this
            # if pid > 801:
                # continue
            assert 0 <= pid <= 1266  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
