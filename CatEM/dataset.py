from tqdm import tqdm
import collections
import numpy as np
from scipy.spatial.distance import pdist, euclidean


class SequenceContextDataset:
    def __init__(self, config, sequence_info, category2id, rk_path):
        self.config = config
        self.sequence_info = sequence_info
        self.category2id = category2id
        self.rk_path = rk_path

    def get_PMI_matrix(self):
        all_centers, all_contexts, category_train_pairs = [], [], []
        for sequence in tqdm(self.sequence_info, desc='get center and context, window size='
                                                      + str(self.config['window_size'])):
            checkins = sequence[1:]
            for i, cur_pos in enumerate(checkins):

                center_category_info, center_time_stamp = cur_pos.split('@')[0], int(cur_pos.split('@')[1])
                center = center_category_info.split('#')[1]

                for j in range(max(i - self.config['window_size'], 0),
                               min(i + self.config['window_size'], len(checkins) - 1)):
                    if j == i:  # remove myself
                        continue
                    context_pos = checkins[j]
                    context_category_info, context_time_stamp = context_pos.split('@')[0], int(
                        context_pos.split('@')[1])

                    context = context_category_info.split('#')[1]
                    all_centers.append(self.category2id[center])
                    all_contexts.append(self.category2id[context])
                    category_train_pairs.append((self.category2id[center], self.category2id[context]))

        pair_counter = collections.Counter(category_train_pairs)
        center_counter = collections.Counter(all_centers)
        context_counter = collections.Counter(all_contexts)
        D_number = len(category_train_pairs)

        # 计算PMI矩阵
        num_category = len(self.category2id.keys())
        PMI_matrix = np.zeros((num_category, num_category))
        for i in range(num_category):
            for j in range(num_category):
                PMI_matrix[i][j] = get_PPMI(i, j, center_counter, context_counter, pair_counter, D_number)

        return PMI_matrix

    def get_similarity_matrix(self):
        category2rk = self.get_rk_vectors()
        categories = list(category2rk.keys())
        distance_matrix = np.zeros((len(categories), len(categories)))
        for i, c_i in enumerate(categories):
            for j, c_j in enumerate(categories):
                x = category2rk[c_i]
                y = category2rk[c_j]

                distance_matrix[self.category2id[c_i]][self.category2id[c_j]] = pdist(np.vstack([x, y]))

        distance_matrix = distance_matrix / distance_matrix.max()
        distance_matrix = maximum_process(distance_matrix, self.config['mad'])
        similarity_matrix = 1 - distance_matrix

        return similarity_matrix

    def get_rk_vectors(self):
        f = open(self.rk_path, 'r', encoding='utf-8')
        content = f.readline()
        category2rk = dict()

        while content != '':
            items = content.strip().split(',')
            observedKs = [float(i) for i in items[1:]]
            for i in observedKs:
                if i != 0.0:
                    min_value = i
                    break
            # min_value = observedKs[1] if observedKs[0] == 0.0 else observedKs[0]
            category2rk[items[0]] = [float(i) / min_value for i in items[1:]]
            content = f.readline()
        f.close()
        return category2rk


def get_PPMI(w, c, w_counter, c_counter, w_c_counter, D_number):
    if (w, c) not in w_c_counter.keys():
        return 0
    PMI = np.log2((w_c_counter[(w, c)] * D_number) / (w_counter[w] * c_counter[c]))
    return max(PMI, 0)


def maximum_process(matrix, n):
    median = np.median(matrix)
    abs_matrix = np.abs(matrix - median)
    new_median = np.median(abs_matrix)
    max_range = median + n * new_median
    return np.clip(matrix, 0, max_range)
