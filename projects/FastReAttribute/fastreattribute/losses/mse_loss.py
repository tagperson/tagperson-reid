# encoding: utf-8
"""
anonymous
anonymous
"""

import torch
import torch.nn.functional as F


def ratio2weight(targets, ratio):
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    weights[targets > 1] = 0.0
    return weights


def mse_loss(pred_parameter_values, gt_values, sample_weight=None):
    # loss = F.binary_cross_entropy_with_logits(pred_class_logits, gt_classes, reduction='none')

    # print(f"pred_parameter_values: {pred_parameter_values}")
    # print(f"pred_parameter_values.shape: {pred_parameter_values.shape}")
    # print(f"gt_values: {gt_values}")
    # print(f"gt_values.shape: {gt_values.shape}")
    # exit()
    loss = F.mse_loss(pred_parameter_values, gt_values)

    # if sample_weight is not None:
    #     targets_mask = torch.where(gt_classes.detach() > 0.5,
    #                                torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"))  # dtype float32
    #     weight = ratio2weight(targets_mask, sample_weight)
    #     loss = loss * weight

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt
    return loss
