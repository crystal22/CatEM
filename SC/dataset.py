# !/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
import random
import torch
import torch.utils.data as Data
import pandas as pd
from tqdm import tqdm
from collections import Counter


warnings.filterwarnings('ignore')


class SpatialContextDataset:
    def __init__(self, poi_info, context_poi_path):
        self.category_set = set()
        self.poi_info = poi_info
        self.context_poi_path = context_poi_path
        self.get_categories()

    def get_categories(self):
        for poi in self.poi_info.keys():
            _, _, category = self.poi_info[poi]
            self.category_set.add(category)

    def get_centers_and_contexts(self):
        with open(self.context_poi_path, 'r', encoding='UTF-8') as contexts_f:
            contexts_lines = contexts_f.readlines()
        context_pois = {}
        for line in contexts_lines:
            items = line.strip().split(',')
            context_pois[items[0]] = items[1:]

        print('number of categories:' + str(len(self.category_set)))
        category_index = range(len(self.category_set))
        category2id = dict(zip(self.category_set, category_index))

        categories = []
        for poi in self.poi_info.keys():
            categories.append(category2id[self.poi_info[poi][2]])
        counter = Counter(categories)

        centers, contexts = [], []
        for center_poi in tqdm(self.poi_info.keys(), desc='getting centers and contexts'):

            centers.append(category2id[self.poi_info[center_poi][2]])

            temp = []
            for context_poi in context_pois[center_poi]:
                context_category = self.poi_info[context_poi][2]
                temp.append(category2id[context_category])
            contexts.append(temp)

        return centers, contexts, counter

    # 负采样,选取噪声词(所有的）
    def get_negatives(self, all_contexts, sampling_weights, K):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in tqdm(all_contexts, desc='get negative samples'):
            negatives = []
            while len(negatives) < len(contexts) * K:
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                    # 为了高效计算 ，可以将k设得稍大一点
                    # random.choices为加权采样函数
                    i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                # 噪声词不能是背景词
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return self.centers[index], self.contexts[index], self.negatives[index]

    def __len__(self):
        return len(self.centers)
