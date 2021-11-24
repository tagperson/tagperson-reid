
import argparse
import glob
import os
import sys
import json

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager
from fastreattribute.config import add_reattribute_config

from predictor import FeatureExtractionDemo

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    add_reattribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--option_name",
        default='',
        help='field name of the option'
    )
    parser.add_argument(
        "--option_value_max",
        type=float,
        default=1.0,
        help=''
    )
    parser.add_argument(
        "--option_value_min",
        type=float,
        default=0.0,
        help=''
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def convert_option_output(option_outputs, option_name, option_value_max=1.0, option_value_min=0.0):
    # outputs is `num_option x num_image`
    assert len(option_outputs) > 0, f"option_outputs is empty!"

    num_image = option_outputs[0].shape[0]
    option_results = []

    # for i in range(num_image):
    #     # option 1
    #     camera_azim_norm = option_outputs[0][i]
    #     camera_azim_value = camera_azim_norm * 360
    #     # option 2
    #     camera_elev_norm = option_outputs[1][i]
    #     camera_elev_value = camera_elev_norm * 90
    #     # option 3
    #     camera_distance_norm = option_outputs[2][i]
    #     camera_distance_value = camera_distance_norm * 40
    #     # option 4
    #     img_width_norm = option_outputs[3][i]
    #     img_width_value = img_width_norm * 200
    #     # option 5
    #     img_height_norm = option_outputs[4][i]
    #     img_height_value = img_height_norm * 600
    #     # option 5
    #     cre_x_bias_norm = option_outputs[5][i]
    #     cre_x_bias_value = cre_x_bias_norm
    #     # option 6
    #     cre_y_bias_norm = option_outputs[6][i]
    #     cre_y_bias_value = cre_y_bias_norm
    #     # option 7
    #     cre_z_bias_norm = option_outputs[7][i]
    #     cre_z_bias_value = cre_z_bias_norm

    #     option_results.append({
    #         'camera_azim': round(float(camera_azim_value), 3),
    #         'camera_elev': round(float(camera_elev_value), 3),
    #         'camera_distance': round(float(camera_distance_value), 3),
    #         'img_width': round(float(img_width_value), 3),
    #         'img_height': round(float(img_height_value), 3),
    #         'cre_x_bias': round(float(cre_x_bias_value), 3),
    #         'cre_y_bias': round(float(cre_y_bias_value), 3),
    #         'cre_z_bias': round(float(cre_z_bias_value), 3),
    #     })
    
    for i in range(num_image):
        # option 1
        # light_elev_norm = option_outputs[0][i]
        # light_elev_value = light_elev_norm * 90
        # option_results.append({
        #     'light_elev': round(float(light_elev_value), 3),
        # })
        
        # option 1
        # light_azim_norm = option_outputs[0][i]
        # light_azim_value = light_azim_norm * 360
        # option_results.append({
        #     'light_azim': round(float(light_azim_value), 3),
        # })
        
        # gamma_value = option_outputs[0][i]
        # gamma_value = gamma_value * 3.0
        # option_results.append({
        #     'gamma_value': round(float(gamma_value), 3),
        # })

        # camera_elev_norm = option_outputs[0][i]
        # camera_elev_value = camera_elev_norm * 90
        # option_results.append({
        #     'camera_elev': round(float(camera_elev_value), 3),
        # })

        option_value_normalized = option_outputs[0][i]
        option_value = option_value_normalized * (option_value_max - option_value_min) + option_value_min
        option_results.append({
            option_name: round(float(option_value), 3),
        })



    return option_results


def save_output_options(output_options_list, save_path='tmp/output_options_list.json'):
    with open(save_path, 'w') as f:
        json.dump(output_options_list, f)



if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.option_name == '':
        raise ValueError(f"option_name is empty!")


    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)


    PathManager.mkdirs(args.output)
    if args.input:
        if PathManager.isdir(args.input[0]):
            glob_target_paths = os.path.join(os.path.expanduser(args.input[0]), '**/*.jpg')
            args.input = glob.glob(glob_target_paths, recursive=True)
            assert args.input, "The input path(s) was not found"

        output_options_list = []

        for path in tqdm.tqdm(args.input):
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"img is None for image_path: {path}")
            res = demo.run_on_image(img)
            if not cfg.TEST.ATTRIBUTE.ENABLED:
                # print(f"TEST.ATTRIBUTE.ENABLED is not enabled, output features")
                feat = res
                feat = postprocess(feat)
                np.save(os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy'), feat)
            else:
                # print(f"TEST.ATTRIBUTE.ENABLED is enabled, output options")
                outputs = res
                # the calculate logic depends on `makehuman_attr.py`
                option_outputs = convert_option_output(outputs['option_outputs'], args.option_name, args.option_value_max, args.option_value_min)
                current_option_output = option_outputs[0]
                # add the file name
                current_option_output['file_name'] = os.path.basename(path)
                output_options_list.append(current_option_output)
                # print(current_option_output)
            
        save_output_options(output_options_list)
