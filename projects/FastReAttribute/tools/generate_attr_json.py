import os
import json
import glob
import re

TARGET_ATTRIBUTE_KEYS = [
    'hair',
    'shoes',
    'clothes_up',
    'clothes_down',
]

def format_attrs(attr_dict_origin, pid, attribute_statistics):
    assert pid in attr_dict_origin, f"pid: {pid} not in attr_dict_origin {attr_dict_origin}"
    """
    `attribute_statistics` used to collect all attribute values
    """
    attr_info = attr_dict_origin[pid]

    attrs = []
    for attr_key in TARGET_ATTRIBUTE_KEYS:
        cur_attr_value = attr_info[attr_key]
        attrs.append({
            'name': attr_key,
            'value': cur_attr_value
        })
        
        if attr_key not in attribute_statistics:
            attribute_statistics[attr_key] = {}
        if cur_attr_value not in attribute_statistics[attr_key]:
            attribute_statistics[attr_key][cur_attr_value] = 0
        attribute_statistics[attr_key][cur_attr_value] += 1
        
    return attrs


def generate_attr_json(mhx2_dir, image_dir, option_dir):
    # step 1. collect
    json_paths = glob.glob(os.path.join(mhx2_dir, '*.json'))
    json_paths = list(filter(lambda x: x.find("generate_config_list") == -1, json_paths))
    pattern = re.compile(r'([-\d]+)_attr')
    attr_dict_origin = {}
    for json_path in json_paths:
        pid, = map(int, pattern.search(json_path).groups())
        if pid == -1:
            continue
        with open(json_path) as f:
            attr_info = json.load(f)
            attr_dict_origin[pid] = attr_info
        
    # step 2. format
    img_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    attr_dict_target = {}
    attribute_statistics = {}
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue
        img_name = os.path.basename(img_path)
        option_path = os.path.join(option_dir, img_name.replace(".jpg", ".json"))
        with open(option_path) as f:
            option_dict = json.load(f)
        attr_dict_target[img_name] = {
            'img': img_name,
            'attrs': format_attrs(attr_dict_origin, pid, attribute_statistics),
            'option_dict': option_dict
        }

    attribute_values_dict = {}
    for attr_key in attribute_statistics:
        attribute_values_dict[attr_key] = list(attribute_statistics[attr_key].keys())
    
    global_dict = {
        'attribute_names': TARGET_ATTRIBUTE_KEYS,
        'attribute_values_count': [len(v) for (k, v) in attribute_values_dict.items()],
        'attribute_values_dict': attribute_values_dict,
        'attribute_statistics': attribute_statistics,
        'img_dict': attr_dict_target,
    }
    return global_dict


def save_attr_json(image_dir, attr_json, save_name = 'attr.json'):
    save_path = os.path.join(image_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump(attr_json, f)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mhx2_dir', type=str, default='', help='path to mhx2')
    parser.add_argument('--image_dir', type=str, default='', help='path to rendered images')
    parser.add_argument('--option_dir', type=str, default='', help='path to rendered images relative options')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """
    Given a mhx2 dir and a image dir, generate the attr.json
    logic:
        1. load all json file in mhx2 dir, form the dict1 key by pid
        2. read all image_path in image_dir,
            generate a dict key by image_path, and the config value can be read from the dict1
    """

    args = parse_args()

    attr_json_dict = generate_attr_json(args.mhx2_dir, args.image_dir, args.option_dir)

    save_attr_json(args.image_dir, attr_json_dict)

