# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank.py

import warnings
from collections import defaultdict
from fastreid.utils import analyze
import os
import cv2
import logging
logger = logging.getLogger(__name__)
import numpy as np
import json
from tqdm import tqdm

try:
    from .rank_cylib.rank_cy import evaluate_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10

    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
                format(num_g)
        )

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, q_paths=None, g_paths=None, save_dir='', cfg=None):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)


    # print(f"matches: {matches}")
    # print(f"indices: {indices}")

    # print(f"q_pid 0: {q_pids[0]}")
    # print(f"q_pid -> top1 : {g_pids[indices[0][0]]}")

    # incorrect_count = 0
    # for i in range(0, len(matches)):
    #     if matches[i][0] != 1:
    #         # print(f"q_pids {i}: {q_pids[i]} -> top1 {g_pids[indices[i][0]]} ")
    #         incorrect_count += 1
    # print(f"incorrect_count= {incorrect_count}")

    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    top1_error_count = 0
    top1_score_list = []
    
    retrieval_info_dict = {}

    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        

        # print(f"matches.shape={matches.shape}, matches={matches}")
        # print(f"indices.shape={indices.shape}, indices={indices}")
        # print(f"keep.shape={keep.shape}, keep={keep}")
        # print(f"distmat.shape={distmat.shape}, distmat={distmat}")
        # print(f"distmat[q_idx].shape={distmat[q_idx].shape}, distmat[q_idx]={distmat[q_idx]}")

        # print(f"distmat[q_idx][indices[q_idx]].shape={distmat[q_idx][indices[q_idx]].shape}, distmat[q_idx][indices[q_idx]]={distmat[q_idx][indices[q_idx]]}")
        # print(f"distmat[q_idx][indices[q_idx]][keep].shape={distmat[q_idx][indices[q_idx]][keep].shape}, distmat[q_idx][indices[q_idx]][keep]={distmat[q_idx][indices[q_idx]][keep]}")
        # `distmat[q_idx][indices[q_idx]][keep]` is the distance for q_idx and sorted_gallery
        # it's shape is 1 x `numof(valid_gallery)`, value is distance from small to large

        # exit()
        # query_path = q_paths[q_idx]
        # print(f"query_path={query_path}")
        # cur_ranked_gallery_paths = g_paths[indices[q_idx]][keep]
        # print(f"cur_ranked_gallery_paths.shape={cur_ranked_gallery_paths.shape}, cur_ranked_gallery_paths[0]={cur_ranked_gallery_paths[0]}")
        if cfg.TEST.ANALYSIS.RANK_LIST.SAVE_RETRIEVAL_INFO:
            retrieval_info_dict[os.path.basename(q_paths[q_idx]).replace(".jpg", "")] = [os.path.basename(p).replace(".jpg", "") for p in g_paths[indices[q_idx]][keep][0:100]]
       
        

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        # print(f"raw_cmc.shape={raw_cmc.shape}, raw_cmc={raw_cmc}")
        # raw_cmc.shape is 1 x `numof(valid_gallery)`, value is 0 or 1 which means whether matched

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)

        top1_score_list.append((distmat[q_idx][indices[q_idx]][keep][0], np.min(pos_idx) == 0))

        
        # if np.min(pos_idx) != 0:
        if not cfg.TEST.ANALYSIS.RANK_LIST.SKIP_SAVE_FILE:
            # write retrieval dict
            top_k = 10
            if q_paths is not None:
                visualized_img = analyze.visualize_one_query(
                    q_paths[q_idx], 
                    g_paths[indices[q_idx]][keep][:top_k], 
                    # distmat[i][indices[i]][valid][:top_k],
                    distmat[q_idx][indices[q_idx]][keep][:top_k],
                    # gallery_paths_npy[indices[i]][valid][y_true][:top_k],
                    g_paths[indices[q_idx]][keep][pos_idx][:top_k], 
                    # distmat[i][indices[i]][valid][y_true][:top_k],
                    distmat[q_idx][indices[q_idx]][keep][pos_idx][:top_k],
                    raw_cmc[:top_k],
                    )
            top1_error_count += 1
            # print(f"top1 error!")
            correct_int = int(np.min(pos_idx) == 0)
            
            query_name = os.path.basename(q_paths[q_idx])
            logger.info(f"[Top1 Error] #q: {query_name}, #got: {[os.path.basename(f) for f in list(g_paths[indices[q_idx]][keep][:top_k])]}")
            top1_score_str = str(round(distmat[q_idx][indices[q_idx]][keep][0], 2))
            cv2.imwrite(f"{save_dir}/{query_name}__{correct_int}__{top1_score_str}.png", visualized_img)
    
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if cfg.TEST.ANALYSIS.RANK_LIST.SAVE_RETRIEVAL_INFO:
        retrieval_info_file_path = os.path.join(save_dir, 'retrieval_info.json')
        with open(retrieval_info_file_path, 'w') as f:
            json.dump(retrieval_info_dict, f)

    # print(f"top1_error_count={top1_error_count}")
    # np.save('top1_score.npy', top1_score_list)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


def evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03, q_paths=None, g_paths=None, save_dir='', cfg=None):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, q_paths, g_paths, save_dir, cfg)


def evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
        use_metric_cuhk03=False,
        use_cython=True,
        q_paths=None,
        g_paths=None,
        save_dir='',
        cfg=None,
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    use_cython = q_paths is None
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
    else:
        return evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03, q_paths, g_paths, save_dir, cfg)
