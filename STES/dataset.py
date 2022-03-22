# !/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
import random
import torch
import torch.utils.data as Data
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np

warnings.filterwarnings('ignore')


class SequenceContextDataset:
    def __init__(self, sequence_info, param, category2id):
        """
        :param sequence_info: [user_id, check-in, check-in, ...] check-in:venue_id#category@time_stamp
        :param param:
        """
        self.param = param
        self.sequence_info = sequence_info
        self.category2id = category2id

    def get_center_and_contexts(self):
        all_centers, all_contexts, category_train_pairs = [], [], []
        window_size = self.param.window_size
        temp = 0
        for sequence in tqdm(self.sequence_info,
                             desc='get center and context, window size=' + str(self.param.window_size)):

            user_id = sequence[0]
            sequence = sequence[1:]
            temp += len(sequence)
            for i, cur_pos in enumerate(sequence):

                center = cur_pos.split('#')[1].split('@')[0]
                if center not in self.category2id.keys():
                    self.category2id[center] = len(self.category2id.keys())

                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(sequence) - 1)):
                    if j == i:  # remove myself
                        continue
                    context_pos = sequence[j]
                    context = context_pos.split('#')[1].split('@')[0]

                    all_centers.append(self.category2id[center])
                    all_contexts.append(self.category2id[context])
                    category_train_pairs.append((self.category2id[center], self.category2id[context]))

        print('avg sequence length:' + str(temp / len(self.sequence_info)))

        return category_train_pairs, all_centers, all_contexts

    def get_PMI_matrix(self):
        category_train_pairs, all_centers, all_contexts = self.get_center_and_contexts()
        pair_counter = Counter(category_train_pairs)
        center_counter = Counter(all_centers)
        context_counter = Counter(all_contexts)

        num_category = len(self.category2id.keys())
        D_number = len(category_train_pairs)

        PMI_matrix = np.full((num_category, num_category), 0, dtype=float)
        for i in range(num_category):
            for j in range(num_category):
                PMI_matrix[i][j] = self.get_PPMI(i, j, center_counter, context_counter, pair_counter, D_number)

        return PMI_matrix

    def get_PPMI(self, w, c, w_counter, c_counter, w_c_counter, D_number):
        if (w, c) not in w_c_counter.keys():
            return 0
        PMI = np.log2((w_c_counter[(w, c)] * D_number) / (w_counter[w] * c_counter[c]))
        return max(PMI, 0)
