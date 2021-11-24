import os
import json
from tqdm import tqdm


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr_path', default='')
    parser.add_argument('--output_path', default='')
    args = parser.parse_args()
    return args


def abstract_option_info_from_attr_json(attr_dict):
    img_dict = attr_dict['img_dict']
    option_info = {}
    for k, v in tqdm(img_dict.items()):
        option_dict = v['option_dict']
        option_info[k] = option_dict
    return option_info


def save_option_info(option_info, save_path):
    with open(save_path, 'w') as f:
        json.dump(option_info, f)


if __name__ == "__main__":
    args = parse_args()
    attr_path = args.attr_path
    if not os.path.exists(attr_path):
        raise ValueError(f"attr_path is invalid: {attr_path}")
    with open(attr_path) as f:
        attr_dict = json.load(f)

    option_info = abstract_option_info_from_attr_json(attr_dict)
    save_option_info(option_info, args.output_path)



    