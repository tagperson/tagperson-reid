# encoding: utf-8
"""
"""

from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image


class ReAttributeDataset(Dataset):
    """
    """

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        
        self.attr_count = len(img_items[0][3])

        pid_set = set()
        cam_set = set()
        attribute_set_array = [set() for i in range(0, self.attr_count)]

        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
            for (attr_idx, attr_value) in enumerate(i[3]):
                attribute_set_array[attr_idx].add(attr_value)


        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.attrs_array = [sorted(list(attr_set)) for attr_set in attribute_set_array]
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            self.attr_dict_array = [dict([p, i] for i, p in enumerate(attrs)) for attrs in self.attrs_array]


    def __len__(self):
        return len(self.img_items)


    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        attr_labels = img_item[3]
        option_labels = img_item[4]
        option_values = img_item[5]

        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
            attr_labels = [self.attr_dict_array[i][attr_label] for (i, attr_label) in enumerate(attr_labels)]
        
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "attr_labels": attr_labels,
            'option_labels': option_labels,
            'option_values': option_values,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
