import numpy as np
import torch
import torch.nn as nn
import math
import os
from loss import MeanSquareWithAdaptiveConstraintAndSpatialEnhanced

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def updata_W(net, k):
    U = net[0].weight.cpu().detach().numpy()
    W = np.zeros((U.shape[0], U.shape[0]))

    for i in range(U.shape[0]):
        temp_u = U[i].reshape(-1, U[i].shape[0])
        repeat_temp_u = np.repeat(temp_u, U.shape[0], axis=0)
        dis = np.sqrt(np.sum(np.square(repeat_temp_u - U), axis=1))
        sorted_dis = np.sort(dis, axis=0)
        mean_nearest_value = (1 + np.sum(sorted_dis[-(k + 1):-1])) / k
        mean_nearest_value = np.repeat(mean_nearest_value, U.shape[0])
        W[i] = np.clip(mean_nearest_value - dis, a_min=0, a_max=1)

    W = (W + np.transpose(W)) / 2
    return W


class Trainer:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.categories = list(self.dataset.category2id.keys())
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train()

    def iteration(self, net, M, k, W2, lambda1, lambda2, loss_fn, optimizer):
        lambda1, lambda2 = torch.tensor(lambda1).to(self.device), torch.tensor(lambda2).to(self.device)

        M, W2, net = torch.tensor(M, dtype=torch.float64).to(self.device), torch.tensor(W2).to(self.device), net.to(
            self.device)
        for i in range(self.config['epoch']):
            W1 = updata_W(net, k)
            W1 = torch.tensor(W1, dtype=torch.float64).to(self.device)
            net = net.to(self.device)
            epoch_loss = loss_fn(net, M, W1, W2, lambda1, lambda2).to(self.device)
            optimizer.zero_grad()
            epoch_loss.backward(retain_graph=True)
            optimizer.step()
            print('epoch%d, loss=%.4f' % (i + 1, epoch_loss))

    def train(self):
        similarity_matrix = self.dataset.get_similarity_matrix()
        PMI_matrix = self.dataset.get_PMI_matrix()
        num_category = similarity_matrix.shape[0]

        loss_fn = MeanSquareWithAdaptiveConstraintAndSpatialEnhanced()

        for embed_size in range(self.config['init_embed_size'], self.config['end_embed_size'] + 1, 20):
            for lambda1 in range(self.config['init_lambda1'], self.config['end_lambda1'] + 1, 1):
                lambda1 = math.pow(10, lambda1)
                for lambda2 in range(self.config['init_lambda2'], self.config['end_lambda2'] + 1, 1):
                    lambda2 = math.pow(10, lambda2)
                    for nk in range(self.config['init_nk'], self.config['end_nk'], 1):
                        reset_random_seed(1)
                        net = nn.Sequential(
                            nn.Embedding(num_embeddings=num_category, embedding_dim=embed_size),
                            nn.Embedding(num_embeddings=num_category, embedding_dim=embed_size)
                        )
                        net[0].weight.data = torch.from_numpy(
                            np.random.normal(0, 1, size=(num_category, embed_size)) * 0.01)
                        net[1].weight.data = torch.from_numpy(
                            np.random.normal(0, 1, size=(num_category, embed_size)) * 0.01)

                        optimizer = torch.optim.Adam(net.parameters(), lr=self.config['lr'],
                                                     weight_decay=self.config['wd'])

                        print('training: lambda1=' + str(format(lambda1, '.4f')) + ', lambda2=' + str(
                            format(lambda2, '.4f')) + ', vector_size=' + str(
                            embed_size) + ', window_size=' + str(self.config['window_size']) + ', k=' + str(nk))

                        self.iteration(net, PMI_matrix, nk, similarity_matrix,
                                       lambda1, lambda2, loss_fn, optimizer)

                        self.save(net, embed_size, self.config['window_size'],
                                  lambda1, lambda2, nk)

    def save(self, net, embed_size, window_size, lambda1, lambda2, nk):
        vector_matrix_u = net[0].cpu().weight.data

        center_category_embedding = np.array(vector_matrix_u)
        f = open(
            self.config['embedding_output_path'] + '/CatEM_' + '#' + str(nk) + '@' + str(embed_size)
            + '#' + str(format(lambda1, '.4f')) + '#' + str(format(lambda2, '.4f')) + '#' + str(window_size)
            + '_' + self.config['city'] + '.txt', 'w', encoding='utf-8')
        for i, category in enumerate(self.categories):
            f.write(category + ',')
            f.write(','.join([str(_) for _ in center_category_embedding[i]]) + '\n')

        f.close()


def reset_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
