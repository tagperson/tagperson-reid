import numpy as np
import torch
from fastreid.utils.compute_dist import build_dist
from tabulate import tabulate
import os
from tqdm import tqdm
from collections import defaultdict

def display_centers(center_root):
    center_num = 6
    center_feature_dict = {}
    center_feature_list = []
    for i in range(0, center_num):
        center_feature_path = f"{center_root}/{i}.npy"
        center_feature_dict[i] = np.load(center_feature_path)
        center_feature_list.append(torch.Tensor(center_feature_dict[i]))
    
    center_feature_list = torch.stack(center_feature_list, dim=0)
    dist_mat = build_dist(center_feature_list, center_feature_list, 'euclidean')

    print(f"centers distance matrix:")
    print(tabulate(dist_mat))
    print(f"\n")
    return dist_mat, center_feature_list

def display_features(dist_mat, center_feature_list, feature_root_dir, dataset_subpath, maintain_percentage):

    max_dis_between_centers = np.max(dist_mat)
    print(f"max_dis_between_centers={max_dis_between_centers}")
    feature_dir = os.path.join(feature_root_dir, dataset_subpath)
    feature_names = os.listdir(feature_dir)
    feature_list = []
    for feature_name in tqdm(feature_names):
        feature_path = os.path.join(feature_dir, feature_name)
        feature = np.load(feature_path)
        feature_list.append(torch.tensor(feature))
    feature_list = torch.stack(feature_list, dim=0)
    feature_dist_mat = build_dist(feature_list, center_feature_list, 'euclidean')

    percentage_list = defaultdict(int)
    camera_count_dict = defaultdict(int)
    for i, distance_info in tqdm(enumerate(feature_dist_mat)):
        min_dis = np.min(distance_info)
        min_index = np.argmin(distance_info)
        camera_count_dict[min_index] += 1
        for per in range(1, 11):
            per_c = per / 10
            if min_dis < max_dis_between_centers * per_c:
                percentage_list[per] += 1

    # 统计小于相关信息的
    feature_names_maintained = []
    for i, distance_info in tqdm(enumerate(feature_dist_mat)):
        min_dis = np.min(distance_info)
        if min_dis < max_dis_between_centers * maintain_percentage:
            feature_names_maintained.append(feature_names[i].replace(".npy", ""))
    
    save_txt_name = dataset_subpath.replace("/", "") + "_[" + str(maintain_percentage) + "].txt"
    save_path = os.path.join(feature_root_dir, save_txt_name)
    with open(save_path, 'w') as f:
        for feature_name in feature_names_maintained:
            f.write(feature_name + "\n")
    
    print(f"camera distribution: {camera_count_dict}")
    print(f"percentage_list: {percentage_list}")

    print(f"filtered image names has been saved to {save_path}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_subpath', type=str, default='')
    parser.add_argument('--feature_root_dir', type=str, default='tmp/camera_feature')
    parser.add_argument('--maintain_percentage', type=float, default=0.25)
    parser.add_argument('--center_root', type=str, default='tmp/camera_center_feature')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    if args.dataset_subpath == "":
        raise ValueError(f'dataset_subpath not specified')

    print(f"dataset_subpath is {args.dataset_subpath}")

    center_dist_mat, center_feature_list = display_centers(args.center_root)
    display_features(center_dist_mat, center_feature_list, args.feature_root_dir, args.dataset_subpath, args.maintain_percentage)