# encoding: utf-8
"""
anonymous
anonymous
"""

import torch
import torch.nn.functional as F

from fastreid.modeling.losses.utils import euclidean_dist, cosine_dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, relative_p_inds = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, relative_n_inds = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an, relative_p_inds, relative_n_inds


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def weighted_triplet_loss(embedding, targets, option_values, margin, norm_feat, hard_mining):
    """
    
    distance is weighted by factor
    
    """


    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an, relative_p_inds, relative_n_inds = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    # print(relative_p_inds)
    # print(option_values)
    # print(is_pos)
    camera_azim_matrix = option_values[:, 0].view(1, N)[0]
    camera_elev_matrix = option_values[:, 1].view(1, N)[0]
    # print(camera_azim_matrix)
    camera_azim_weight = torch.zeros(N, N)
    camera_elev_weight = torch.zeros(N, N)
    # print(camera_azim_weight.shape)
    for i in range(N):
        for j in range(N):
            camera_azim_weight[i][j] = abs(abs(camera_azim_matrix[i] - camera_azim_matrix[j]) - 0.5)
            camera_elev_weight[i][j] = abs(camera_elev_matrix[i] - camera_elev_matrix[j])
    dist_ap_weight = torch.ones(N)
    dist_an_weight = torch.ones(N)
    for i in range(N):
        dist_ap_weight[i] += camera_azim_weight[i][relative_p_inds[i]]
        dist_ap_weight[i] += camera_elev_weight[i][relative_p_inds[i]]
        dist_an_weight[i] -= camera_azim_weight[i][relative_n_inds[i]]
        dist_an_weight[i] -= camera_elev_weight[i][relative_n_inds[i]]
    
    # print(dist_ap_weight)
    # print(dist_an.shape)
    dist_an_weight = dist_an_weight.to(dist_an.device)
    dist_ap_weight = dist_ap_weight.to(dist_an.device)
    # print(dist_an)
    # exit()
    # print(dist_an_weight)
    # print(dist_an * dist_an_weight)

    y = dist_an.new().resize_as_(dist_an).fill_(1)
    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss
