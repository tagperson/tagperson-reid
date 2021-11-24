import os
from tqdm import tqdm
import shutil

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='')
    parser.add_argument('--filter_file_path', default='')
    parser.add_argument('--output_folder', default='')
    args = parser.parse_args()
    return args

def split_image(dataset_root, filter_file_path, output_folder):
    include_path = os.path.join(output_folder, os.path.basename(dataset_root) + "_include")
    exclude_path = os.path.join(output_folder, os.path.basename(dataset_root) + "_exclude")
    if not os.path.exists(include_path):
        os.makedirs(include_path)
    if not os.path.exists(exclude_path):
        os.makedirs(exclude_path)

    include_dict = {}
    with open(filter_file_path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            include_dict[line] = True
    
    file_names = os.listdir(dataset_root)
    for file_name in tqdm(file_names):
        file_path = os.path.join(dataset_root, file_name)
        if file_name in include_dict.keys():
            # mv to include path
            dst_path = os.path.join(include_path, file_name)
        else:
            dst_path = os.path.join(exclude_path, file_name)
        
        print(dst_path)
        # shutil.copy(file_path, dst_path)



if __name__ == '__main__':
    args = parse_args()
    
    
    split_image(args.dataset_root, args.filter_file_path, args.output_folder)
    
    