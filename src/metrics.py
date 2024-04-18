# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_squared_error, roc_auc_score)

# import random
from logger import logger


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def recall_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: recall score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    n = 0
    for idx in argsort:
        if idx in ground_truth:
            n += 1
    return n / len(ground_truth)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_score (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_accuracy_metric(labels, preds, metrics):
    """Calculate metrics.

    Available options are: `auc`, `rmse`, `logloss`, `acc` (accurary), `f1`, `mean_mrr`,
    `ndcg` (format like: ndcg@2;4;6;8), `hit` (format like: hit@2;4;6;8), `group_auc`.

    Args:
        labels (array-like): Labels.
        preds (array-like): Predictions.
        metrics (list): List of metric names.

    Return:
        dict: Metrics.

    Examples:
        >>> cal_accuracy_metric(labels, preds, ["ndcg@2;4;6", "group_auc"])
        {'ndcg@2': 0.4026, 'ndcg@4': 0.4953, 'ndcg@6': 0.5346, 'group_auc': 0.8096}

    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric.startswith('recall'):
            recall_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                recall_list = [int(token) for token in ks[1].split(';')]
            for k in recall_list:
                recall_temp = np.mean([
                    recall_score(each_labels, each_preds, k)
                    for each_labels, each_preds in zip(labels, preds)
                ])
                res['recall@{0}'.format(k)] = round(recall_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("Metric {0} not defined".format(metric))
    return res


def calc_single_diversity_metric(candidates, preds, ks, dataset, uid):
    order = np.argsort(preds)[::-1]
    res = {}
    for _k in ks:
        k = min(_k, len(order))
        
        category_dict = {}
        for item in candidates[order[:k]]:
            for cat in dataset.item_categories[item]:
                category_dict[cat] = category_dict.get(cat, 0) + 1
        
        n_categories = sum(category_dict.values())
        si = 1
        for v in category_dict.values():
            si -= (v / n_categories) ** 2

        res[f'si@{_k}'] = si

        category_similarities = []
        for i in range(k):
            for j in range(i + 1, k):
                cat_i = dataset.item_categories[candidates[order[i]]]
                cat_j = dataset.item_categories[candidates[order[j]]]
                category_similarities.append(len(cat_i.intersection(cat_j)) / len(cat_i.union(cat_j)))

        res[f'cat@{_k}'] = sum(category_similarities) / len(category_similarities)

        
    return res


def calc_diversity_metric(diversity_metrics):
    res = {}
    for metrics in diversity_metrics:
        for metric in metrics:
            res[metric] = res.get(metric, 0) + metrics[metric]
    for metric in res:
        res[metric] = round(res[metric] / len(diversity_metrics), 4)
    return res
