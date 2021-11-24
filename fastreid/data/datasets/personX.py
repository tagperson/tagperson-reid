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
class PersonX(ImageDataset):
    """PersonX.

    Reference:
        Dissecting person re-identification from the viewpoint of viewpoint. CVPR 2019.

    URL: `<https://github.com/sxzrt/Instructions-of-the-PersonX-dataset>`_

    Dataset statistics:
        - identities: 1266 (+1 for background).
        - images: x (train) + x (query) + x (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = None
    dataset_name = "PersonX"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.subsets = ['5', '6']

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'PersonX_v1')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "1/bounding_box_train" under '
                          '"PersonX_v1".')

        self.train_dir_list = [osp.join(self.data_dir, subset, 'bounding_box_train') for subset in self.subsets]
        self.query_dir_list = [osp.join(self.data_dir, subset, 'query') for subset in self.subsets]
        self.gallery_dir_list = [osp.join(self.data_dir, subset, 'bounding_box_test') for subset in self.subsets]
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir,
        ]
        required_files.extend(self.train_dir_list)
        required_files.extend(self.query_dir_list)
        required_files.extend(self.gallery_dir_list)
        
        self.check_before_run(required_files)

        train = []
        for train_dir in self.train_dir_list:
            train += self.process_dir(train_dir)
        query = []
        for query_dir in self.query_dir_list:
            query += self.process_dir(query_dir, is_train=False)
        gallery = []
        for gallery_dir in self.gallery_dir_list:
            gallery += self.process_dir(gallery_dir, is_train=False)

        super(PersonX, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1266  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
