# encoding: utf-8
"""
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.losses import *
from .losses.expand_triplet_loss import line_triplet_loss
from .losses.weighted_triplet_loss import weighted_triplet_loss
from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .build_reattribute_head import build_reattribute_heads, build_reattribute_option_heads
from .losses.mse_loss import mse_loss

from fastreid.config import configurable

@META_ARCH_REGISTRY.register()
class ReAttributeBaseline(Baseline):
    def __init__(self, cfg):
        super(ReAttributeBaseline, self).__init__(cfg)
        self.attribute_heads = build_reattribute_heads(cfg)
        self.attribute_heads = nn.ModuleList(self.attribute_heads)

        self.option_heads = build_reattribute_option_heads(cfg)
        self.option_heads = nn.ModuleList(self.option_heads)


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        
        """
        img_paths = batched_inputs['img_paths']
        from conrefa import compose_image_list_into_one
        thumbnail = compose_image_list_into_one.build_img_list_thumbnail(list(img_paths), row=4)
        import cv2
        import uuid
        save_path = f"tmp/debug/{str(uuid.uuid4())}.jpg"
        cv2.imwrite(save_path, thumbnail)
        print(f"save to {save_path}")
        """

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()
            

            # expand training logic
            if self._cfg.DATALOADER.EXPAND_NAIVE_IDENTITY_SAMPLER.CAMERA.ENABLED:
                # TODO: make 32 to mini_batch_size
                mini_batch_size = 32
                main_features = features[0:mini_batch_size,:,:,:]
                camera_part_features = features[mini_batch_size:,:,:,:]
                main_targets = targets[0:mini_batch_size]
                camera_part_targets = targets[mini_batch_size:]
                features = main_features
                targets = main_targets

            option_values = batched_inputs["option_values"]
            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets, option_values)

            # expand training logic
            if self._cfg.DATALOADER.EXPAND_NAIVE_IDENTITY_SAMPLER.CAMERA.ENABLED:
                camera_part_outputs = self.heads(camera_part_features, camera_part_targets)
                loss_cfg = self._cfg.MODEL.LOSSES.EXPAND_CAMERA
                num_instances = self._cfg.DATALOADER.NUM_INSTANCE
                expand_camera_losses_dict = self.expand_camera_losses(camera_part_outputs, camera_part_targets, loss_cfg, num_instances)
                expand_camera_losses_dict_renamed = {}
                for k, v in expand_camera_losses_dict.items():
                    expand_camera_losses_dict_renamed[f"expand_camera_{k}"] = v
                losses.update(expand_camera_losses_dict_renamed)


            if self._cfg.MODEL.LOSSES.ATTRIBUTE.ENABLED:
                assert "attr_labels" in batched_inputs, "Attribute labels annotation are missing in training!"
                attribute_labels_list = batched_inputs["attr_labels"].to(self.device)
                attribute_outputs_list = [attr_head(features, attribute_labels_list[attr_idx]) for (attr_idx, attr_head) in enumerate(self.attribute_heads)]
                attribute_loss_dict = self.attribute_losses(attribute_outputs_list, attribute_labels_list)

                losses.update(attribute_loss_dict)

            if self._cfg.MODEL.LOSSES.OPTION.ENABLED:
                assert "option_values" in batched_inputs, "Option values annotation are missing in training!"
                option_values_list = batched_inputs["option_values"].to(self.device)
                option_outputs_list = [option_head(features, option_values_list[option_idx]) for (option_idx, option_head) in enumerate(self.option_heads)]
                option_loss_dict = self.option_losses(option_outputs_list, option_values_list)
                losses.update(option_loss_dict)

            return losses
        else:
            # TODO:
            if self._cfg.TEST.ANALYSIS.VISUALIZE_ATTENTION.ENABLED:
                from conrefa import visualize_attention
                import os
                if not os.path.exists(self._cfg.TEST.ANALYSIS.VISUALIZE_ATTENTION.SAVE_DIR):
                    os.makedirs(self._cfg.TEST.ANALYSIS.VISUALIZE_ATTENTION.SAVE_DIR)
                visualize_attention.save_attention_visualization(self._cfg, batched_inputs, features)

            if not self._cfg.TEST.ATTRIBUTE.ENABLED:
                outputs = self.heads(features)
            else:
                # attribute_head_idx = self._cfg.TEST.ATTRIBUTE.ATTRIBUTE_HEAD_INDEX
                # outputs = self.attribute_heads[attribute_head_idx](features)

                attribute_outputs = [self.attribute_heads[attribute_head_idx](features) for (attribute_head_idx, head) in enumerate(self.attribute_heads)]
                option_outputs = [self.option_heads[option_head_idx](features) for (option_head_idx, head) in enumerate(self.option_heads)]
                outputs = {
                    'attribute_outputs': attribute_outputs,
                    'option_outputs': option_outputs,
                }
            return outputs

    
    def losses(self, outputs, gt_labels, option_values=None):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # print(f"pred_class_logits: {pred_class_logits}")
        # print(f"pred_class_logits.shape: {pred_class_logits.shape}")
        # print(f"cls_outputs: {cls_outputs}")
        # print(f"cls_outputs.shape: {cls_outputs.shape}")
        # print(f"pred_features: {pred_features}")
        # print(f"pred_features.shape: {pred_features.shape}")
        # print(f"gt_labels: {gt_labels}")
        # print(f"gt_labels.shape: {gt_labels.shape}")

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            if self._cfg.MODEL.LOSSES.TRI.WEIGHTED_V1:
                loss_dict["loss_triplet"] = weighted_triplet_loss(
                    pred_features,
                    gt_labels,
                    option_values,
                    self._cfg.MODEL.LOSSES.TRI.MARGIN,
                    self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
                ) * self._cfg.MODEL.LOSSES.TRI.SCALE
            else:
                loss_dict["loss_triplet"] = triplet_loss(
                    pred_features,
                    gt_labels,
                    self._cfg.MODEL.LOSSES.TRI.MARGIN,
                    self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
                ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        if "Cosface" in loss_names:
            loss_dict["loss_cosface"] = pairwise_cosface(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.COSFACE.MARGIN,
                self._cfg.MODEL.LOSSES.COSFACE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.COSFACE.SCALE

        
        return loss_dict

    def attribute_losses(self, attribute_outputs_list, attribute_labels_list):
        
        assert len(attribute_outputs_list) == len(attribute_labels_list[0])
        loss_dict_list = []
        for attr_idx in range(0, attribute_labels_list.shape[1]):
            loss_dict = self.attribute_loss(attribute_outputs_list[attr_idx], attribute_labels_list[:, attr_idx], self._cfg.MODEL.ATTRIBUTE_HEADS[attr_idx]['LOSSES'])
            loss_dict_list.append(loss_dict)
        
        attr_loss_dict = {}
        for (attr_idx, loss_dict) in enumerate(loss_dict_list):
            for k, v in loss_dict.items():
                attr_loss_dict[f"attr_{attr_idx}_{k}"] = v
        return attr_loss_dict

    def attribute_loss(self, outputs, gt_labels, losses_dict):
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        loss_dict = {}
        loss_names = losses_dict['NAME']

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                losses_dict['CE']['EPSILON'],
                losses_dict['CE']['ALPHA'],
            ) * losses_dict['CE']['SCALE']

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                losses_dict['TRI']['MARGIN'],
                losses_dict['TRI']['NORM_FEAT'],
                losses_dict['TRI']['HARD_MINING'],
            ) * losses_dict['TRI']['SCALE']

        return loss_dict

    def option_losses(self, option_outputs_list, option_values_list):
        assert len(option_outputs_list) == len(option_values_list[0])
        loss_dict_list = []
        for option_idx in range(0, option_values_list.shape[1]):
            loss_dict = self.option_loss(option_outputs_list[option_idx], option_values_list[:, option_idx], self._cfg.MODEL.OPTION_HEADS[option_idx]['LOSSES'])
            loss_dict_list.append(loss_dict)
        
        option_loss_dict = {}
        for (option_idx, loss_dict) in enumerate(loss_dict_list):
            for k, v in loss_dict.items():
                option_loss_dict[f"option_{option_idx}_{k}"] = v
        return option_loss_dict
    
    def option_loss(self, outputs, gt_labels, losses_dict):
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on
        loss_dict = {}
        loss_names = losses_dict['NAME']

        if "MSELoss" in loss_names:
            loss_dict["loss_mse"] = mse_loss(
                cls_outputs.squeeze(),
                gt_labels,
            ) * losses_dict['MSE']['SCALE']

        return loss_dict

    def expand_camera_losses(self, outputs, gt_labels, loss_cfg, num_instances):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']

        # TODO: refactor this codes, which is similar to losses
        loss_dict = {}
        loss_names = loss_cfg.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                loss_cfg.CE.EPSILON,
                loss_cfg.CE.ALPHA,
            ) * loss_cfg.CE.SCALE

        if not loss_cfg.TRI.USE_LINE:
            if "TripletLoss" in loss_names:
                loss_dict["loss_triplet"] = triplet_loss(
                    pred_features,
                    gt_labels,
                    loss_cfg.TRI.MARGIN,
                    loss_cfg.TRI.NORM_FEAT,
                    loss_cfg.TRI.HARD_MINING,
                ) * loss_cfg.TRI.SCALE
        else:
            if "TripletLoss" in loss_names:
                loss_dict["loss_triplet"] = 0
                mini_batch_size = 32    #TODO: modify this
                p = int(mini_batch_size / num_instances)
                for i in range(p):
                    cur_pred_features = pred_features[i*num_instances:(i+1)*num_instances]
                    cur_gt_labels = gt_labels[i*num_instances:(i+1)*num_instances]
                    cur_tri_loss = line_triplet_loss(
                        cur_pred_features,
                        cur_gt_labels,
                        loss_cfg.TRI.MARGIN,
                        loss_cfg.TRI.NORM_FEAT,
                        loss_cfg.TRI.HARD_MINING,
                    ) * loss_cfg.TRI.SCALE
                    loss_dict["loss_triplet"] += cur_tri_loss

        return loss_dict