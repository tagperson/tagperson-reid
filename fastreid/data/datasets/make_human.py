# encoding: utf-8

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import os


@DATASET_REGISTRY.register()
class MakeHuman(ImageDataset):
    """MakeHuman.
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = None
    dataset_name = "MakeHuman"

    def __init__(self, root='datasets', makehuman_subpath=[''], cfg=None, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.cfg = cfg

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MakeHuman/')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "1/bounding_box_train" under '
                          '"MakeHuman".')

        self.train_dir = [osp.join(self.data_dir, sub) for sub in makehuman_subpath]
        self.query_dir = [osp.join(self.data_dir, sub) for sub in makehuman_subpath]
        self.gallery_dir = [osp.join(self.data_dir, sub) for sub in makehuman_subpath]

        required_files = [
            self.data_dir,
        ]
        required_files.extend(self.train_dir)
        required_files.extend(self.query_dir)
        required_files.extend(self.gallery_dir)

        self.check_before_run(required_files)

        train = self.process_dir_list(self.train_dir)
        query = self.process_dir_list(self.query_dir, is_train=False, is_query=True)
        gallery = self.process_dir_list(self.gallery_dir, is_train=False)

        super(MakeHuman, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, pid_offset=0, filter_path=None, is_query=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([\d]+)s1_([-\d]+)')

        # 20210705 prepare filter_path
        filter_name_dict = None
        if filter_path is not None and filter_path != '':
            if os.path.exists(filter_path):
                with open(filter_path) as f:
                    filter_name_list = f.readlines()
                    filter_name_list = [filter_name.rstrip() for filter_name in filter_name_list]
                    filter_name_dict = set(filter_name_list)

        # 20210715 used for split train val
        camera_flag_for_query = {}

        data = []
        for img_path in img_paths:
            # 20210705 filter logic
            if filter_name_dict is not None:
                if os.path.basename(img_path) not in filter_name_dict:
                    continue

            pid, camid, seq_id = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1266  # pid == 0 means background
            # if pid > 656:
            #     continue
            # if (pid % 1148) % 168 >= 2:
            #     continue
            if seq_id > self.cfg.DATASETS.MAKEHUMAN.MAX_SEQ_ID:
                continue

            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            pid += pid_offset

            if self.cfg.DATASETS.MAKEHUMAN.SPLIT_TRAIN_VAL:
                if is_train:
                    # pid even number split to query/gallery
                    if pid % 2 == 0:
                        continue
                if not is_train:
                    # pid odd number split to train
                    if pid % 2 == 1:
                        continue
                    else:
                        # select first one in camera in each qid into query
                        if pid not in camera_flag_for_query:
                            camera_flag_for_query[pid] = {}
                        if is_query:
                            if camid not in camera_flag_for_query[pid]:
                                camera_flag_for_query[pid][camid] = True
                                # add to query
                            else:
                                # camid has been added, do not add to query
                                continue
                        else:
                            if camid in camera_flag_for_query[pid]:
                                camera_flag_for_query[pid][camid] = True
                                continue
                            else:
                                pass
            
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

    def process_dir_list(self, dir_path_list, is_train=True, is_query=False):
        if not isinstance(dir_path_list, list):
            return self.process_dir(dir_path_list, is_train, is_query=is_query)
        else:
            data = []
            pid_offset = 0
            for idx, dir_path in enumerate(dir_path_list):
                if idx < len(self.cfg.DATASETS.MAKEHUMAN.FILTER_FILE_PATH):
                    filter_path = self.cfg.DATASETS.MAKEHUMAN.FILTER_FILE_PATH[idx]
                cur_data = self.process_dir(dir_path, is_train, pid_offset, filter_path, is_query=is_query)
                data.extend(cur_data)
                pid_offset = len(data)
            return data