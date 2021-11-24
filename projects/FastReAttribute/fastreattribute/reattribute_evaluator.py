# encoding: utf-8
"""
"""
import copy
import logging
from collections import OrderedDict

import torch

from fastreid.evaluation.evaluator import DatasetEvaluator
from fastreid.utils import comm

from fastreid.evaluation.clas_evaluator import accuracy

logger = logging.getLogger(__name__)


class ReAttribuetEvaluator(DatasetEvaluator):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pred_logits = []
        self.gt_labels = []
        self.features = []

        self.img_paths = []

        self.option_values_list = []
        self.option_labels_list = []

    def reset(self):
        self.features = []
        self.pred_logits = []
        self.gt_labels = []
        self.img_paths = []

        self.option_values_list = []
        self.option_labels_list = []

    def process(self, inputs, outputs):
        if not self.cfg.TEST.ATTRIBUTE.ENABLED:
            self.gt_labels.extend(inputs["targets"])
            self.pred_logits.extend(outputs.cpu())
        else:
            self.img_paths.extend(inputs["img_paths"])
            assert "attr_labels" in inputs, "Attribute labels annotation are missing in training!"
            attribute_labels_list = inputs["attr_labels"]
            attribute_head_idx = self.cfg.TEST.ATTRIBUTE.ATTRIBUTE_HEAD_INDEX

            gt_label = inputs["attr_labels"][:, attribute_head_idx]
            # print(f"gt_label={gt_label.shape}")
            self.gt_labels.extend(gt_label.cpu())

            if self.cfg.TEST.ATTRIBUTE.EVAL_ONE_ATTRIBUTE:
                one_attribute_output = outputs['attribute_outputs'][attribute_head_idx]
                self.pred_logits.extend(one_attribute_output['pred_class_logits'].cpu())
                self.features.extend(one_attribute_output['features'].cpu())
            else:
                option_outputs = outputs['option_outputs']
                if len(self.option_values_list) == 0:
                    self.option_values_list = [[] for o in option_outputs]
                if len(self.option_labels_list) == 0:
                    self.option_labels_list = [[] for o in option_outputs]

                assert "option_values" in inputs, "Options labels annotation are missing in training!"
                option_labels_list = [inputs["option_values"][:, option_head_id] for option_head_id in range(inputs["option_values"].shape[1])]
                for (idx, _) in enumerate(option_labels_list):
                    self.option_labels_list[idx].extend(option_labels_list[idx].cpu())
                
                for (idx, _) in enumerate(option_outputs):
                    self.option_values_list[idx].extend(option_outputs[idx]['pred_class_logits'].cpu())
                
                    

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            pred_logits = comm.gather(self.pred_logits)
            pred_logits = sum(pred_logits, [])

            gt_labels = comm.gather(self.gt_labels)
            gt_labels = sum(gt_labels, [])

            features = comm.gather(self.features)
            features = sum(features, [])

            # option
            option_values_list = comm.gather(self.option_values_list)
            option_values_list = sum(option_values_list, [])
            
            option_labels_list = comm.gather(self.option_labels_list)
            option_labels_list = sum(option_labels_list, [])

            if not comm.is_main_process():
                return {}
        else:
            pred_logits = self.pred_logits
            gt_labels = self.gt_labels
            features = self.features
            
            option_values_list = self.option_values_list
            option_labels_list = self.option_labels_list

        if self.cfg.TEST.ATTRIBUTE.EVAL_ONE_ATTRIBUTE:
            pred_logits = torch.stack(pred_logits, dim=0)
            print(f"pred_logits={pred_logits}")
            gt_labels = torch.stack(gt_labels, dim=0)
            print(f"gt_labels={gt_labels}")
            features = torch.stack(self.features, dim=0)
            print(f"features={features}")
            print(f"features.shape={features.shape}")

            # temp: save center feature
            # self.save_cls_center_feature(gt_labels, features)
            # self.save_element_feature(self.img_paths, features)

            _, pred = pred_logits.topk(1, 1, True, True)
            # print(f"pred={pred}")
            for i in range(0, 6):
                count = (pred == i).nonzero().size(0)
                print(f"{i}:{count}")
            # print(f"pred.t()={pred.t()}")
            # print(f"pred.shape={pred.shape}")

            acc1, = accuracy(pred_logits, gt_labels, topk=(1,))

            self._results = OrderedDict()
            self._results["Acc@1"] = acc1
            self._results["metric"] = acc1
        else:
            # print(option_values_list)
            # print(option_labels_list)

            self._results = OrderedDict()

            # calculate accuracy
            for (idx, option_values) in enumerate(option_values_list):
                option_labels = option_labels_list[idx]
                total_count = len(option_labels)
                correct_count = 0
                for img_idx, _ in enumerate(option_labels):
                    diff = abs(option_labels[img_idx] - option_values[img_idx])
                    if diff <= 0.05:
                        correct_count += 1               
                self._results[f"Acc@option_{idx}"] = correct_count / (total_count + 1e-6)

        return copy.deepcopy(self._results)

    def save_cls_center_feature(self, gt_labels, features):
        from tqdm import tqdm
        from collections import defaultdict
        import numpy as np
        from fastreid.utils.compute_dist import build_dist
        

        cls_center_feature_collection = defaultdict(list)
        for i, gt_label in tqdm(enumerate(gt_labels.numpy())):
            cls_center_feature_collection[gt_label].append(features[i])
        
        cls_center_feature = {}
        for label, center_feature_list in cls_center_feature_collection.items():
            center_feature_list = torch.stack(center_feature_list)
            cls_center_feature[label] = torch.mean(center_feature_list, dim=0)
            np.save(f"tmp/camera_center_feature/{label}.npy", cls_center_feature[label].numpy())
    
    def save_element_feature(self, img_paths, features):
        from tqdm import tqdm
        import numpy as np
        import os
        for i, img_path in tqdm(enumerate(img_paths)):
            img_name = os.path.basename(img_path)
            np.save(f"tmp/camera_feature/makehuman_0713_3024_7x3_0720_baseline_40_50_shield_bgcoco_camera_group_15/{img_name}.npy", features[i])