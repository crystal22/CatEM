# !/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
import sys
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np

from dataset import MyDataset

warnings.filterwarnings('ignore')


class SCTrainer:
    def __init__(self, knn_dataset, params):
        self.knn_dataset = knn_dataset
        self.params = params
        self.train()

    def skip_gram(self, center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        # 小批量乘法，对两个小批量中的矩阵一一做乘法
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    # 选取Batch
    def batchify(self, data):
        """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
        list中的每个元素都是Dataset类调⽤用__getitem__得到的结果
        """
        max_len = max(len(c) + len(n) for _, c, n in data)  # 获取语境词与噪声词拼接后的最长长度
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]  # 填充
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]  # 用于区分原有的词和填充词
            labels += [[1] * len(context) + [0] * (max_len - len(context))]  # 用于区分语境词和噪声词
        return (torch.tensor(centers).view(-1, 1),
                torch.tensor(contexts_negatives),
                torch.tensor(masks),
                torch.tensor(labels))

    def iteration(self, net, num_epochs, data_iter, loss, optimizer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("train on", device)
        net = net.to(device)
        for epoch in range(num_epochs):
            l_sum, n = 0.0, 0
            for batch in data_iter:
                center, context_negative, mask, label = [d.to(device) for d in batch]

                pred = self.skip_gram(center, context_negative, net[0], net[1])
                # 使⽤用掩码变量mask来避免填充项对损失函数计算的影响
                # 一个batch的平均loss
                l = (loss(pred.view(label.shape), label, mask) * mask.shape[1] / mask.float().sum(dim=1)).mean()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1
            print('epoch %d, loss %.4f' % (epoch + 1, l_sum / n))

    def train(self):
        all_centers, all_contexts, counter = self.knn_dataset.get_centers_and_contexts()
        all_negatives = self.knn_dataset.get_negatives(all_contexts, counter, self.params.negative_sample_num)
        num_category = len(self.knn_dataset.category_set)
        for embed_size in range(self.params.init_embed_size, self.params.end_embed_size, 10):
            print('embed_size:' + str(embed_size))
            net = nn.Sequential(
                nn.Embedding(num_embeddings=num_category, embedding_dim=embed_size),
                nn.Embedding(num_embeddings=num_category, embedding_dim=embed_size)
            )

            # 高斯分布乘以0.01初始化向量
            net[0].weight.data = torch.from_numpy(
                np.random.normal(0, 1, size=(num_category, embed_size)) * 0.01)
            net[1].weight.data = torch.from_numpy(
                np.random.normal(0, 1, size=(num_category, embed_size)) * 0.01)

            num_workers = 0 if sys.platform.startswith('win32') else 4
            optimizer = torch.optim.Adam(net.parameters(), lr=self.params.lr)
            loss = SigmoidBinaryCrossEntropyLoss()

            dataset = MyDataset(all_centers, all_contexts, all_negatives)
            data_iter = Data.DataLoader(dataset, self.params.batch_size, shuffle=True, collate_fn=self.batchify,
                                        num_workers=num_workers)

            self.iteration(net, self.params.epoch, data_iter, loss, optimizer)
            self.save(net, embed_size)
            # torch.save(net, 'vectors/place2vec#5@' + str(embed_size) + '$5' + '.pth')

    def save(self, net, embed_size):
        vector_matrix_u = net[0].cpu().weight.data
        vector_matrix_v = net[1].cpu().weight.data

        center_category_embedding = np.array(vector_matrix_u)
        context_category_embedding = np.array(vector_matrix_v)
        f = open(self.params.embedding_output_path + '/sc' + '@' + str(embed_size) +
                 '_' + self.params.city + '.txt', 'w', encoding='utf-8')
        for i, category in enumerate(self.knn_dataset.category_set):
            f.write(category + ',')
            f.write(','.join([str(_) for _ in center_category_embedding[i]]) + '\n')

        f.close()
        # f = open(self.params.embedding_output_path + '/context' + '#' + str(self.params.k) + '@' + str(embed_size) +
        #          '_' + self.params.city + '.txt', 'w', encoding='utf-8')
        # for i, category in enumerate(self.knn_dataset.category_set):
        #     f.write(category + ',')
        #     f.write(','.join([str(_) for _ in context_category_embedding[i]]) + '\n')
        # f.close()


# 二元交叉熵损失函数
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):  # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)
