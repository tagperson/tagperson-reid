# encoding: utf-8
"""
copy from `demo/demo.py`
"""

import argparse
import glob
import os
import sys

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


def convert_option_output(option_outputs):
    # outputs is `num_option x num_image`
    assert len(option_outputs) > 0, f"option_outputs is empty!"

    num_image = option_outputs[0].shape[0]
    option_results = []
    for i in range(num_image):
        # option 1
        camera_azim_norm = option_outputs[0][i]
        camera_azim_value = camera_azim_norm * 360
        # option 2
        camera_elev_norm = option_outputs[1][i]
        camera_elev_value = camera_elev_norm * 90
        # option 3
        camera_distance_norm = option_outputs[2][i]
        camera_distance_value = camera_distance_norm * 40

        option_results.append({
            'camera_azim': round(float(camera_azim_value), 3),
            'camera_elev': round(float(camera_elev_value), 3),
            'camera_distance': round(float(camera_distance_value), 3),
        })

    return option_results



if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    PathManager.mkdirs(args.output)
    if args.input:
        if PathManager.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"img is None for image_path: {path}")
            res = demo.run_on_image(img)
            if not cfg.TEST.ATTRIBUTE.ENABLED:
                print(f"TEST.ATTRIBUTE.ENABLED is not enabled, output features")
                feat = res
                feat = postprocess(feat)
                np.save(os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy'), feat)
            else:
                print(f"TEST.ATTRIBUTE.ENABLED is enabled, output options")
                outputs = res
                # the calculate logic depends on `makehuman_attr.py`
                option_outputs = convert_option_output(outputs['option_outputs'])
                current_option_output = option_outputs[0]
                print(current_option_output)
