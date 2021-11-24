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
class MakeHumanAttr(ImageDataset):
    """MakeHumanAttr.
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = None
    dataset_name = "MakeHumanAttrs"

    def __init__(self, root='datasets', makehuman_subpath=[''], cfg=None, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.cfg = cfg

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MakeHumanAttr/')
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
        query = self.process_dir_list(self.query_dir, is_train=False)
        gallery = self.process_dir_list(self.gallery_dir, is_train=False)

        super(MakeHumanAttr, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, pid_offset=0):

        img_attr_labels_list_dict, img_options_labels_list_dict, img_options_values_list_dict = self._load_attr_annotation(dir_path)

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            camid -= 1  # index starts from 0

            pid += pid_offset
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            img_name = os.path.basename(img_path)
            attribute_labels = img_attr_labels_list_dict[img_name]
            option_labels = img_options_labels_list_dict[img_name]
            option_values = img_options_values_list_dict[img_name]
            data.append((img_path, pid, camid, attribute_labels, option_labels, option_values))

            # # TODO: remove this debug
            if not is_train and len(data) >= 8192:
                break               

        return data

    def process_dir_list(self, dir_path_list, is_train=True):
        if not isinstance(dir_path_list, list):
            return self.process_dir(dir_path_list, is_train)
        else:
            data = []
            pid_offset = 0
            for dir_path in dir_path_list:
                cur_data = self.process_dir(dir_path, is_train, pid_offset)
                data.extend(cur_data)
                pid_offset = len(data)
            return data

    def _load_attr_annotation(self, dir_path):
        """
        load attribute annotation from 
        the annotation should contains a dict like:
        {
            
            'file_path': {
                'file_name': '',
                'attrs': [
                    {
                        'name': 'shoes',
                        'value': 'name of shoes'
                    }
                ]
            },
        }
        """

        attr_annotation_file_name = os.path.join(dir_path, 'attr.json')
        with open(attr_annotation_file_name) as f:
            attr_annotation_dict = json.load(f)

        # prepare attribute label_map

        img_attr_labels_list_dict = {}
        assert 'img_dict' in attr_annotation_dict
        for img_name, node in attr_annotation_dict['img_dict'].items():
            img_attr_labels_list_dict[img_name] = [attr['value'] for attr in node['attrs']]

        img_options_labels_list_dict = {}
        img_options_values_list_dict = {}
        for img_name, node in attr_annotation_dict['img_dict'].items():
            img_options_labels_list_dict[img_name] = node['option_dict']
            
            # for option 20210825
            # img_options_values_list_dict[img_name] = [
            #     node['option_dict']['camera_azim'] / 360,
            #     node['option_dict']['camera_elev'] / 90,
            #     node['option_dict']['camera_distance'] / 40,
            #     node['option_dict']['img_width'] / 200,
            #     node['option_dict']['img_height'] / 600,
            #     node['option_dict']['cre_x_bias'],
            #     node['option_dict']['cre_y_bias'] / 0.2,
            #     node['option_dict']['cre_z_bias'],
            # ]

            # for option light_elev 20210916
            # img_options_values_list_dict[img_name] = [
            #     node['option_dict']['light_elev'] / 90,
            # ]
            
            # for option light_azim 20210923
            # azim_rela = abs((node['option_dict']['light_azim'] + 360) % 360 - ((node['option_dict']['camera_azim'] + 360) % 360))
            # img_options_values_list_dict[img_name] = [
            #     azim_rela / 360,
            # ]

            # for option gamma 20211013
            # gamma_value = (node['option_dict']['gamma_value']) 
            # img_options_values_list_dict[img_name] = [
            #     gamma_value / 3.0,
            # ]

            # for option camera_elev 20211028
            # camera_elev = (node['option_dict']['camera_elev']) 
            # img_options_values_list_dict[img_name] = [
            #     camera_elev / 90,
            # ]

            
            img_options_values_list_dict[img_name] = []
            for option_item in self.cfg.DATASETS.MAKEHUMAN_ATTR.OPTIONS:
                # for option camera_elev 20211028
                option_name = option_item['OPTION_NAME']
                option_value = node['option_dict'][option_name]
                normalized_value = (option_value - option_item['OPTION_VALUE_MIN']) / (option_item['OPTION_VALUE_MAX'] - option_item['OPTION_VALUE_MIN'])
                img_options_values_list_dict[img_name].append(normalized_value)



        return img_attr_labels_list_dict, img_options_labels_list_dict, img_options_values_list_dict
