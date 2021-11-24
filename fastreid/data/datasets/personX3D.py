# encoding: utf-8
"""
anonymous
anonymous
"""

import glob
import os.path as osp
import os
import re
import warnings
import cv2
from tqdm import tqdm
import time
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from fastreid.utils import renderer, renderer_v2, renderer_util
import torch
import random

MARKET_PATH = 'datasets/Market-1501-v15.09.15/bounding_box_train'


@DATASET_REGISTRY.register()
class PersonX3D(ImageDataset):
    """PersonX3D.

    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_3d_dir = 'PersonX3D/3d'
    dataset_generated_dir = 'PersonX3D/generated'
    dataset_url = None
    dataset_name = "PersonX3D"


    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # generated 3d images
        self.use_cached_id_dir = True
        self.generate_images(use_cached_id_dir=self.use_cached_id_dir)

        # generated end
        self.subsets = ['1', '2', '3', '4']
        # allow alternative directory structure
        data_dir = self.dataset_dir + self.dataset_generated_dir
        # data_dir = osp.join(self.data_dir, 'PersonX_3D')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "1/bounding_box_train" under '
                          '"PersonX_3D".')

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

        super(PersonX3D, self).__init__(train, query, gallery, **kwargs)

    def generate_images(self, use_cached_id_dir=True):
        dataset_3d_dir_path = os.path.join(self.dataset_dir, self.dataset_3d_dir)
        dataset_generated_dir_path = os.path.join(self.dataset_dir, self.dataset_generated_dir)
        id_3d_list = ["A" + str(i) for i in range(1, 361)]
        id_3d_list_train = id_3d_list[:100]
        id_3d_list_test = id_3d_list[100:200]

        camera_ids = [1, 2, 3, 4]
        self.__create_camera_dir(camera_ids, dataset_generated_dir_path)
        # for train entity, render and save to train folder
        print(f"start to generate image train")
        batch_size = 16
        t0 = time.time()

        light_locations = [
            [0.0, 0.0, 5.0],
            # [0.0, 0.0, -5.0],
            [5.0, 0.0, 0.0],
            # [-5.0, 0.0, 0.0],
        ]
        specular_colors = [
            # [1.0, 0.0, 0.0],
            # [0.0, 1.0, 0.0],
            # [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ]

        light_location = light_locations[0]
        specular_color = specular_colors[0]


        target_domain_image_paths = self.get_target_domain_image_path_list()

        for id_index, id_3d in tqdm(enumerate(id_3d_list_train)):
            probe_save_name = f"{id_index}_c1s1_0.jpg"
            probe_save_path = os.path.join(dataset_generated_dir_path, str(1), "bounding_box_train", probe_save_name)
            if use_cached_id_dir and os.path.exists(probe_save_path):
                # print(f"{id_3d} for camera#{camera_id} has exist, skip...")
                continue

            obj_file_path = os.path.join(dataset_3d_dir_path, f"SampleScene_{id_3d}.obj")
            meshes = renderer_v2.load_meshes_from_obj_file_paths([obj_file_path])
            
            for camera_id in camera_ids:
                # temp process to avoid redundant render process

                image_batch = renderer_v2.batch_render_images(meshes, camera_id=camera_id, light_location=light_location, specular_color=specular_color)
                # save to path
                background_count = 20
                for i, img in enumerate(image_batch):
                    for j in range(0, background_count):
                        img_c = self.__add_background(img, target_domain_image_paths)
                        save_name = f"{id_index}_c{camera_id}s1_{i*background_count+j}.jpg"
                        save_path = os.path.join(dataset_generated_dir_path, str(camera_id), "bounding_box_train", save_name)
                        cv2.imwrite(save_path, img_c)
        
        print(f"finish generate image train, cost {time.time() - t0}s")

        # for test entity, render and save to test/query folder
        id_with_ambiguity = ['A147', 'A182','A183','A184','A187','A190','A195']
        print(f"start to generate image test")
        t0 = time.time()
        for id_index, id_3d in tqdm(enumerate(id_3d_list_test)):
            if id_3d in id_with_ambiguity:
                continue
            # temp process to avoid redundant render process
            probe_save_name = f"{id_index}_c1s1_0.jpg"
            probe_save_path = os.path.join(dataset_generated_dir_path, str(1), "query", probe_save_name)
            probe_save_name_ga = f"{id_index}_c1s1_1.jpg"
            probe_save_path_ga = os.path.join(dataset_generated_dir_path, str(1), "bounding_box_test", probe_save_name_ga)
            if use_cached_id_dir and os.path.exists(probe_save_path) and os.path.exists(probe_save_path_ga):
                # print(f"{id_3d} for camera#{camera_id} has exist, skip...")
                continue

            obj_file_path = os.path.join(dataset_3d_dir_path, f"SampleScene_{id_3d}.obj")
            meshes = renderer_v2.load_meshes_from_obj_file_paths([obj_file_path])
            for camera_id in camera_ids:
                image_batch = renderer_v2.batch_render_images(meshes, camera_id=camera_id, light_location=light_location, specular_color=specular_color)
                image_batch_2 = renderer_v2.batch_render_images(meshes, camera_id=camera_id, light_location=light_locations[1], specular_color=specular_color)
                image_batch.extend(image_batch_2)
                # save to path
                background_count = 20
                for i, img in enumerate(image_batch):
                    for j in range(0, background_count):
                        save_name = f"{id_index}_c{camera_id}s1_{i*background_count+j}.jpg"
                        if i == 0 and j == 0:
                            save_path = os.path.join(dataset_generated_dir_path, str(camera_id), "query", save_name)
                        else:
                            save_path = os.path.join(dataset_generated_dir_path, str(camera_id), "bounding_box_test", save_name)
                        # print(f"save to {save_path}")
                        cv2.imwrite(save_path, img)
        print(f"finish generate image test, cost {time.time() - t0}s")
        print(f"image generated finished")


    def get_target_domain_image_path_list(self, domain_name='Market1501'):
        if domain_name == 'Market1501':
            img_paths = glob.glob(os.path.join(MARKET_PATH, '*.jpg'))
            return img_paths
        else:
            raise ValueError(f"Unsupported target domain: {domain_name}")

    def __add_background(self, rendered_img, target_domain_image_paths):
        target_domain_image_path = random.choice(target_domain_image_paths)
        bg_img = cv2.imread(target_domain_image_path)
        synthesis_img = renderer_util.add_image_on_background(bg_img, rendered_img)
        return synthesis_img

    def __create_camera_dir(self, camera_ids, dataset_3d_dir_path):
        for camera_id in camera_ids:
            dir_path = os.path.join(dataset_3d_dir_path, str(camera_id))
            train_path = os.path.join(dir_path, "bounding_box_train")
            gallery_path = os.path.join(dir_path, "bounding_box_test")
            query_path = os.path.join(dir_path, "query")
            for path in [train_path, gallery_path, query_path]:
                if not os.path.exists(path):
                    os.makedirs(path)

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
