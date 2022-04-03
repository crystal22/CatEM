import math
import random
import time
from collections import Counter

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


def get_cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    sim = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
    return sim


def get_global_sim_pair(city):
    path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + city.upper() + \
           'Data\\RK_result_200Scale_category_ExpectedK_ObservedK.csv'

    f = open(path, 'r', encoding='utf-8')
    content = f.readline()
    category2rk = dict()

    while content != '':
        items = content.strip().split(',')
        observedKs = [float(i) for i in items[1:]]
        for i in observedKs:
            if i != 0.0:
                min_value = i
                break
        category2rk[items[0]] = [float(i) / min_value for i in items[1:]]
        # category2rk[items[0]] = [float(i) for i in items[1:]]

        content = f.readline()
    f.close()
    categories = category2rk.keys()
    similarity_pair = dict()

    n = 101  # nyc 100 tky 50
    for c_i in categories:
        temp_c, min_value = '', math.inf
        for c_j in categories:
            if c_i == c_j:
                continue
            x = category2rk[c_i][:n]
            y = category2rk[c_j][:n]
            temp = cosine(x, y)
            #  cityblock
            # temp = pdist(np.vstack([x, y]))
            if temp < min_value:
                min_value = temp
                temp_c = c_j
        similarity_pair[c_i] = temp_c

    category2id = dict(zip(sorted(categories), range(len(categories))))
    return similarity_pair, category2id


def get_local_sim_pair(city, category2id):
    with open('../../distance/' + city + '/category_mean_distance_matrix.txt', 'r',
              encoding='utf-8') as f:
        distance_list = f.readlines()
    categories = list(category2id.keys())
    similarity_pair = dict()

    for i in range(len(distance_list)):
        min_dis, temp_c = math.inf, ''
        distances = distance_list[i].split(',')[:-1]
        for j in range(len(distances)):
            if i == j: continue
            if float(distances[j]) < min_dis:
                min_dis = float(distances[j])
                temp_c = categories[j]
        similarity_pair[categories[i]] = temp_c

    count_category = list(similarity_pair.values())
    counter = Counter(count_category)
    # print(counter)
    return similarity_pair


def evaluation(city, models):

    similarity_pair, category2id = get_global_sim_pair(city)

    for i in range(len(models)):
        category_embedding_path = 'E:\\PycharmProjects\\Embedding Space\\CatEM\\embeddings1\\' + models[i] + '@100_' + city + '.txt'
        print(models[i])
        time.sleep(0.1)
        category_embedding = {}
        with open(category_embedding_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split(',')

                category = temp[0]
                embedding = [float(temp[i]) for i in range(1, len(temp))]
                category_embedding[category] = embedding

        categories = similarity_pair.keys()

        MRR, hit5, hit10 = 0, 0, 0
        for turn in tqdm(range(0, 10)):
            # random.seed(turn)
            for c_i in categories:
                sim_pair = similarity_pair[c_i]
                temp_rank = dict()
                pos_neg_categories = random.choices(list(categories), k=50)
                pos_neg_categories.append(sim_pair)
                for c_j in pos_neg_categories:
                    if c_i == c_j: continue
                    try:
                        v1, v2 = category_embedding[c_i], category_embedding[c_j]
                    except KeyError:
                        continue
                    sim = get_cosine_similarity(v1, v2)
                    temp_rank[c_j] = sim

                temp_rank = [(temp_rank[k], k) for k in temp_rank.keys()]
                temp_rank.sort(reverse=True)

                for rank, (_, c) in enumerate(temp_rank):
                    if c == sim_pair:
                        MRR += (1 / (rank + 1))
                        break
                # for _, c in temp_rank[:10]:
                #     if c == sim_pair:
                #         hit10 += 1
                #         break
                for _, c in temp_rank[:5]:
                    if c == sim_pair:
                        hit5 += 1
                        break

        print('MRR:' + '%.4f' % (MRR / (len(categories) * 10))
              + ', Hit@5:' + '%.4f' % (hit5 / (len(categories) * 10))
              + ', Hit@10:' + '%.4f' % (hit10 / (len(categories) * 10)))
        time.sleep(0.1)


def obtain_group_list(city):
    f = open('../task2/group_info_' + city + '.txt', 'r', encoding='utf-8')
    content = f.readline()
    items = content.strip().split(',')
    num_clustered, num_middle, num_even = int(items[0]), int(items[1]), int(items[2])
    category2group = dict()
    categories = []
    group2list = {0: [], 1: [], 2: []}
    idx = 1
    content = f.readline()
    while content != '':
        item = content.strip()
        if idx <= num_clustered:
            category2group[item] = 0
            group2list[0].append(item)
        elif idx <= num_clustered + num_middle:
            category2group[item] = 1
            group2list[1].append(item)
        else:
            category2group[item] = 2
            group2list[2].append(item)
        categories.append(item)
        idx += 1
        content = f.readline()
    f.close()
    return group2list


def evaluation1(city, models):
    group2list = obtain_group_list(city)
    similarity_pair, category2id = get_global_sim_pair(city)
    # similarity_pair = get_local_sim_pair(city, category2id)
    print(len(group2list[0]), len(group2list[1]), len(group2list[2]))

    for i in range(len(models)):
        category_embedding_path = '../embeddings/' + models[i] + '@100_' + city + '.txt'
        print(models[i])
        time.sleep(0.1)
        category_embedding = {}
        with open(category_embedding_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split(',')[:-1]

                category = temp[0]
                embedding = [float(temp[i]) for i in range(1, len(temp))]
                category_embedding[category] = embedding

        categories = similarity_pair.keys()
        MRR_all, hit5_all = 0, 0
        res = []
        for group_id in group2list.keys():
            MRR, hit5, hit10 = 0, 0, 0
            for turn in range(0, 10):
                for c_i in group2list[group_id]:
                    sim_pair = similarity_pair[c_i]
                    temp_rank = dict()
                    pos_neg_categories = random.choices(list(categories), k=50)
                    pos_neg_categories.append(sim_pair)
                    for c_j in pos_neg_categories:
                        if c_i == c_j: continue
                        try:
                            v1, v2 = category_embedding[c_i], category_embedding[c_j]
                        except KeyError:
                            continue
                        sim = get_cosine_similarity(v1, v2)
                        temp_rank[c_j] = sim

                    temp_rank = [(temp_rank[k], k) for k in temp_rank.keys()]
                    temp_rank.sort(reverse=True)

                    for rank, (_, c) in enumerate(temp_rank):
                        if c == sim_pair:
                            MRR += (1 / (rank + 1))
                            break
                    for _, c in temp_rank[:10]:
                        if c == sim_pair:
                            hit10 += 1
                            break
                    for _, c in temp_rank[:5]:
                        if c == sim_pair:
                            hit5 += 1
                            break

            # print('MRR:' + '%.4f' % (MRR / (len(group2list[group_id]) * 10))
            #       + ', Hit@5:' + '%.4f' % (hit5 / (len(group2list[group_id]) * 10))
            #       + ', Hit@10:' + '%.4f' % (hit10 / (len(group2list[group_id]) * 10)))
            MRR_group = MRR / (len(group2list[group_id]) * 10)
            hit5_group = hit5 / (len(group2list[group_id]) * 10)
            MRR_all += MRR_group
            hit5_all += hit5_group
            res.append(format(MRR_group, '.3f'))
            res.append(format(hit5_group, '.3f'))
            time.sleep(0.1)
        res.append(format(MRR_all / 3, '.3f'))
        res.append(format(hit5_all / 3, '.3f'))
        print(' & '.join(list(map(str, res))))


# models = ['sc',  'habit2vec', 'stes', 'lstm2', 'poi2vec', 'geosan']
models = ['pte_global']
# different dataset need modify k !!!
# The results will vary slightly from run to run.
city = 'nyc'  # nyc l
# evaluation1(city, models)
evaluation(city, models)
