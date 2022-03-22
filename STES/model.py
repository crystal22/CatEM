# !/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
import sys
import torch
from torch import nn
import numpy as np


warnings.filterwarnings('ignore')
random_seed = 1388
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


class STESTrainer:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train()

    def iteration(self, net, M, steps, loss, optimizer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("train on", device)
        net = net.to(device)
        ex = 10000
        for i in range(steps):
            W = net[0].weight
            C = net[1].weight
            pred = W.mm(C.T)
            target = torch.tensor(M, dtype=pred.dtype).to(device)
            l = loss(pred.view(target.shape), target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % 10 == 0:
                print('step %d, loss %.4f' % (i, l))
            if ex - l < 1e-6:
                break
            ex = l

    def train(self):

        PMI_matrix = torch.tensor(self.dataset.get_PMI_matrix()).to(self.device)
        num_category = len(self.dataset.category2id.keys())

        for embed_size in range(self.params.init_embed_size, self.params.end_embed_size, 10):
            print('embed_size:' + str(embed_size))
            net = nn.Sequential(
                nn.Embedding(num_embeddings=num_category, embedding_dim=embed_size),
                nn.Embedding(num_embeddings=num_category, embedding_dim=embed_size)
            )
            net[0].weight.data = torch.from_numpy(
                np.random.normal(0, 1, size=(num_category, embed_size)) * 0.01)
            net[1].weight.data = torch.from_numpy(
                np.random.normal(0, 1, size=(num_category, embed_size)) * 0.01)

            optimizer = torch.optim.Adam(net.parameters(), lr=self.params.lr)
            loss = nn.MSELoss()

            self.iteration(net.to(self.device), PMI_matrix, self.params.epoch, loss, optimizer)
            self.save(net, embed_size)
            # torch.save(net, 'vectors/place2vec#5@' + str(embed_size) + '$5' + '.pth')

    def save(self, net, embed_size):
        vector_matrix_u = net[0].cpu().weight.data
        vector_matrix_v = net[1].cpu().weight.data

        center_category_embedding = np.array(vector_matrix_u)
        context_category_embedding = np.array(vector_matrix_v)
        # f = open(self.params.embedding_output_path + '/stes' + '#' + str(self.params.window_size) + '@' + str(embed_size) +
        #          '_' + self.params.city + '.txt', 'w', encoding='utf-8')
        f = open(
            self.params.embedding_output_path + '/stes' + '@' + str(embed_size) +
            '_' + self.params.city + '.txt', 'w', encoding='utf-8')
        for i, category in enumerate(self.dataset.category2id.keys()):
            f.write(category + ',')
            f.write(','.join([str(_) for _ in center_category_embedding[i]]) + '\n')

        f.close()

        # f = open(self.params.embedding_output_path + '/context' + '#' + str(self.params.window_size) + '@' + str(embed_size) +
        #          '_' + self.params.city + '.txt', 'w', encoding='utf-8')
        # for i, category in enumerate(self.dataset.category2id.keys()):
        #     f.write(category + ',')
        #     f.write(','.join([str(_) for _ in context_category_embedding[i]]) + '\n')
        # f.close()

