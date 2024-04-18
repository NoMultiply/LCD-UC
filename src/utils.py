import random

import numpy as np
import torch
from tqdm import tqdm
from logger import logger
from metrics import cal_accuracy_metric, calc_single_diversity_metric, calc_diversity_metric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def difference(ta, tb):
    vs, cs = torch.cat([ta, tb]).unique(return_counts=True)
    return vs[cs == 1]


def intersection(ta, tb):
    vs, cs = torch.cat([ta, tb]).unique(return_counts=True)
    return vs[cs > 1]

def valid_model(model, data, dataset, args, metrics, ks=None, diversity=False,):
    if ks is None:
        ks = [20]
    model.eval()
    all_labels = []
    all_preds = []
    all_items = torch.LongTensor(list(range(dataset.n_items))).to(args.device)

    diversity_metrics = []
    with torch.no_grad():
        user_embeddings, item_embeddings = model.get_embeddings(
            torch.LongTensor(list(range(dataset.n_users))).to(args.device),
            torch.LongTensor(list(range(dataset.n_items))).to(args.device)
        )
        for uid, (pos_iids, _) in tqdm(data.items(), bar_format='{l_bar}{r_bar}', desc='[Valid Model]'):
            if uid in dataset.train_data:
                candidates_tensor = difference(
                    difference(all_items, dataset.train_data_tensor[uid][0]),
                    dataset.train_data_tensor[uid][1])
            else:
                candidates_tensor = all_items

            if intersection(pos_iids, candidates_tensor).size(0) == 0:
                continue

            labels = torch.isin(candidates_tensor,
                                pos_iids).long().detach().cpu().numpy()
            candidates = candidates_tensor.detach().cpu().numpy()
            
            preds = model.get_score_for_valid(
                user_embeddings, item_embeddings,
                torch.LongTensor(
                    [uid] * candidates_tensor.size(0)).to(args.device),
                candidates_tensor
            ).detach().cpu().numpy()

            if diversity:
                diversity_metrics.append(calc_single_diversity_metric(
                    candidates, preds, ks, dataset, uid))

            all_labels.append(labels)
            all_preds.append(preds)
    accuracy_metrics = cal_accuracy_metric(all_labels, all_preds, metrics)
    model.train()
    if not diversity:
        return accuracy_metrics
    
    diversity_metrics = calc_diversity_metric(diversity_metrics)
    accuracy_metrics.update(diversity_metrics)
    return accuracy_metrics


def print_results(res):
    logger.print('[Testing]')
    k = 20
    logger.print(f'NDCG@{k}:', res[f'ndcg@{k}'])
    logger.print(f'Recall@{k}:', res[f'recall@{k}'])
    logger.print(f'ILCS@{k}:', res[f'cat@{k}'])
    logger.print(f'ICSI@{k}:', res[f'si@{k}'])
