import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

from logger import logger, DATA_ROOT


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class Dataset:
    def __init__(self, dataset, device=None):
        self.dataset = dataset

        self.user_map = {}
        self.user_features = []
        with open(os.path.join(DATA_ROOT, dataset, 'users.txt')) as fin:
            for line in tqdm(fin, desc=f'[{dataset}][users.txt]'):
                uid, features = line.strip().split('\t')
                self.user_map[uid] = len(self.user_map)
                self.user_features.append(list(map(float, features.split(','))))
        assert len(self.user_map) == len(self.user_features)
        self.n_users = len(self.user_map)
        self.n_user_features = len(self.user_features[0])
        self.user_features = np.array(self.user_features)

        self.item_map = {}
        self.item_features = []
        self.item_categories = []
        self.categories = set()
        self.cid_dict = {}
        self.item2cid = []
        with open(os.path.join(DATA_ROOT, dataset, 'items.txt')) as fin:
            for line in tqdm(fin, desc=f'[{dataset}][items.txt]'):
                iid, features, categories = line.strip().split('\t')
                self.item_map[iid] = len(self.item_map)
                self.item_features.append(list(map(float, features.split(','))))
                categories = set(categories.split('|'))
                self.item_categories.append(categories)
                self.categories.update(categories)
                categories_key = '|'.join(sorted(categories))
                if categories_key not in self.cid_dict:
                    self.cid_dict[categories_key] = len(self.cid_dict)
                self.item2cid.append(self.cid_dict[categories_key])
        assert len(self.item_map) == len(self.item_features)
        self.n_items = len(self.item_map)
        self.n_category = len(self.categories)
        self.mc_cid_dict = {c: i for i, c in enumerate(self.categories)}
        self.mc_item_catetories = []
        for categories in self.item_categories:
            label = [0] * self.n_category
            for c in categories:
                label[self.mc_cid_dict[c]] = 1
            self.mc_item_catetories.append(label)
        self.mc_item_catetories = torch.FloatTensor(self.mc_item_catetories).to(device)
        self.n_item_features = len(self.item_features[0])
        self.item_features = np.array(self.item_features)

        self.train_data, self.valid_data, self.test_data = {}, {}, {}
        self.user_categories = [set()] * self.n_users
        for split in ('train', 'valid', 'test'):
            data = {}
            n_skip = 0
            with open(os.path.join(DATA_ROOT, dataset, f'{split}.txt')) as fin:
                for line in tqdm(fin, desc=f'[{dataset}][{split}.txt]'):
                    uid, pos_iids, neg_iids = line.strip('\n').split('\t')
                    if pos_iids == '' or neg_iids == '' or uid not in self.user_map:
                        n_skip += 1
                        continue
                    data[self.user_map[uid]] = [
                        [self.item_map[iid] for iid in pos_iids.split(',')],
                        [self.item_map[iid] for iid in neg_iids.split(',')]
                    ]
                    for iid in data[self.user_map[uid]][0]:
                        self.user_categories[self.user_map[uid]].update(self.item_categories[iid])
            if n_skip > 0:
                logger.print(
                    f'[WARNING][{split}] Skip {n_skip} users due to lack of positive or negative interactions or no features.')
            setattr(self, split + '_data', data)

        for split in ('valid', 'test'):
            data = getattr(self, split + '_data')
            for k in data:
                data[k] = [torch.LongTensor(data[k][0]).to(device), torch.LongTensor(data[k][1]).to(device)]

        self.train_data_tensor = {}
        for k in self.train_data:
            self.train_data_tensor[k] = [torch.LongTensor(self.train_data[k][0]).to(device),
                                            torch.LongTensor(self.train_data[k][1]).to(device)]

    def __repr__(self):
        string = ''
        string += '-------------------------------------\n'
        string += f'[{self.dataset}]\n'
        string += '\n'

        string += f'# of users: {self.n_users}\n'
        string += f'# of items: {self.n_items}\n'
        string += f'# of nodes: {self.n_users} + {self.n_items} = {self.n_users + self.n_items}\n'
        string += '\n'

        string += f'# of user features: {self.n_user_features}\n'
        string += f'# of item features: {self.n_item_features}\n'
        string += '\n'

        n_train_pos = sum(len(x[0]) for x in self.train_data.values())
        n_train_neg = sum(len(x[1]) for x in self.train_data.values())
        string += f'# of training interactions: {n_train_pos} + {n_train_neg} = {n_train_pos + n_train_neg}\n'

        n_valid_pos = sum(len(x[0]) for x in self.valid_data.values())
        n_valid_neg = sum(len(x[1]) for x in self.valid_data.values())
        string += f'# of valid interactions: {n_valid_pos} + {n_valid_neg} = {n_valid_pos + n_valid_neg}\n'

        n_test_pos = sum(len(x[0]) for x in self.test_data.values())
        n_test_neg = sum(len(x[1]) for x in self.test_data.values())
        string += f'# of testing interactions: {n_test_pos} + {n_test_neg} = {n_test_pos + n_test_neg}\n'

        n_pos = n_train_pos + n_valid_pos + n_test_pos
        n_neg = n_train_neg + n_valid_neg + n_test_neg
        string += f'# of interactions: {n_pos} + {n_neg} = {n_pos + n_neg}\n'

        string += '-------------------------------------'
        return string
