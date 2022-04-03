import numpy as np

category_path = 'E:\\JupyterNoteBookSpace\\Hierarchical Tree\\data\\category.csv'
np.seterr(invalid='ignore')


def get_category_to_type():
    category_to_type = {}
    # Arts & Entertainment/College & University/Food/
    # Nightlife Spot/Outdoors & Recreation/Professional & Other Places/
    # Residence/Shop & Service/Travel & Transport
    category_to_id = {'4d4b7104d754a06370d81259': 0, '4d4b7105d754a06372d81259': 1, '4d4b7105d754a06374d81259': 2,
                      '4d4b7105d754a06376d81259': 3, '4d4b7105d754a06377d81259': 4, '4d4b7105d754a06375d81259': 5,
                      '4e67e38e036454776db1fb3a': 6, '4d4b7105d754a06378d81259': 7, '4d4b7105d754a06379d81259': 8,
                      '4d4b7105d754a06373d81259': 9}

    with open(category_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in lines:

        temp = line.split(',')
        category_name, category_id, category_type = temp[1], temp[0], temp[2]
        # print(category_name)
        if '$' not in category_type:
            category_to_type[category_name] = category_to_id[category_id]
        else:
            temp = category_type.split('$')
            category_to_type[category_name] = category_to_id[temp[len(temp) - 2]]
    return category_to_type


def semantic_overlap(category_embedding_path, category_to_type, k):
    """
    对于每一个venue type，寻找在向量空间中最相似的N个venue type，判断是否属于同一个top-layer venue type
    :return:
    """
    category_embedding = {}
    with open(category_embedding_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        temp = line.split(',')

        category = temp[0]
        embedding = [float(temp[i]) for i in range(1, len(temp))]
        category_embedding[category] = embedding

    mean_match_rate, num_category = 0, len(category_embedding)
    for current_category in category_embedding.keys():
        rank_list = {}
        current_embedding = category_embedding.get(current_category)
        for candidate_category in category_embedding.keys():
            if current_category == candidate_category: continue
            candidate_embedding = category_embedding.get(candidate_category)
            cos_sim = get_cosine_similarity(current_embedding, candidate_embedding)
            rank_list.update({candidate_category: cos_sim})
        rank_list = sorted(rank_list.items(), key=lambda x: x[1], reverse=False)
        category_rank = [t[0] for t in rank_list][-k:]
        match_count = 0
        for candidate_category in category_rank:
            if category_to_type.get(current_category) == category_to_type.get(candidate_category):
                match_count += 1

        match_rate = match_count / k
        mean_match_rate += match_rate
    mean_match_rate = mean_match_rate / num_category

    return mean_match_rate


def get_cosine_similarity(vector1, vector2):
    v1, v2 = np.array(vector1), np.array(vector2)
    sim = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
    return sim


def main():
    category_to_type = get_category_to_type()
    k = 1
    city = 'tky'
    models = ['sc', 'habit2vec', 'stes', 'lstm2', 'poi2vec', 'geosan', 'CatEM']
    # models = ['pte_global']
    # models = ['poi2vec', 'geosan','pte_l', 'pte_g']

    # cate_groups = obtain_group_list(city)
    # print(len(cate_groups[0]), len(cate_groups[1]), len(cate_groups[2]))
    # for i in range(len(models)):
    #     cate_embed_path = '../embeddings1/' + models[i] + '@100_' + city + '.txt'
    #     print(cate_embed_path)
    #     res, res_sum = [], 0
    #     for group_id in cate_groups.keys():
    #         mrr = semantic_overlap_group(cate_embed_path, category_to_type, k, cate_groups[group_id])
    #         res_sum += mrr
    #         res.append(str(format(mrr, '.3f')))
    #     res.append(str(format(res_sum / 3, '.3f')))
    #     print(' & '.join(res))

    # for i in range(len(models)):
    #     cate_embed_path = '../embeddings1/' + models[i] + '@100_' + city + '.txt'
    #     print(cate_embed_path)
    #     mrrs = []
    #     for k in [1, 5, 10]:
    #         mrr = semantic_overlap(cate_embed_path, category_to_type, k)
    #         mrrs.append(str(format(mrr, '.3f')))
    #     print(' & '.join(mrrs))

    for k in [1, 5, 10]:
        print('k=' + str(k))
        for model in models:
            mrrs = []
            for city in ['nyc', 'tky']:
                cate_embed_path = '../embeddings1/' + model + '@100_' + city + '.txt'
                # print(cate_embed_path)
                mrr = semantic_overlap(cate_embed_path, category_to_type, k)
                mrrs.append(str(format(mrr, '.3f')))
            print(', '.join(mrrs))

    # for model in models:
    #     print(model)
    #     for city in ['nyc', 'tky']:
    #         print(city)
    #         mrrs = []
    #         for embed_size in range(20, 201, 20):
    #             cate_embed_path = '../../output/' + model + '@' + str(embed_size) + '_' + city + '.txt'
    #
    #             mrr = semantic_overlap(cate_embed_path, category_to_type, k)
    #             mrrs.append(str(format(mrr, '.3f')))
    #         print(', '.join(mrrs))

    # for model in models:
    #     print(model)
    #     for city in ['nyc', 'tky']:
    #         print(city)
    #         k = 2 if city == 'nyc' else 5
    #         mrrs = []
    #         for lambda1 in range(-4, 4, 1):
    #             cate_embed_path = '../../output/' + model + '#5@100#' + \
    #                               str(format(pow(10, lambda1), '.4f')) + '#0.0100#' + str(k) + '_' + city + '.txt'
    #             print(cate_embed_path)
    #             mrr = semantic_overlap(cate_embed_path, category_to_type, k)
    #             mrrs.append(str(format(mrr, '.3f')))
    #         print(', '.join(mrrs))


if __name__ == '__main__':
    main()
