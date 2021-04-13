from gensim.models import word2vec, KeyedVectors
from tqdm import tqdm

CITY = 'MY'
TSMC_CATEGORIES_USED_DIR = 'E:\\Documents\\DataSet\\TIST2015\\' + CITY + '\\categories_used.txt'
TSMC_CHECKIN_TRAJECTORY_DIR = 'E:\\Documents\\DataSet\\TIST2015\\' + CITY + '\\Checkins_Trajectory_FilterByUser10-' + CITY + '.csv'
TSMC_VENUE_INFO_DIR = 'E:\\Documents\\DataSet\\TIST2015\\' + CITY + '\\PoisFilterBy10-' + CITY + '.csv'
TSMC_CATEGORIES_USED_DIR = 'E:\\Documents\\DataSet\\TIST2015\\' + CITY + '\\categories_used.txt'

from utils.embedding_utils import get_category_used

sentences = word2vec.Text8Corpus('./category_sequence_for_gensim_my.txt')


def baseline1():
    negative_list = [1, 5]

    for vector_size in tqdm(range(10, 151, 10)):
        for negative in negative_list:
            model = word2vec.Word2Vec(sentences=sentences, window=5,
                                      size=vector_size,
                                      sg=1,
                                      negative=negative,
                                      hs=0,
                                      min_count=1)
            word_vectors = model.wv
            # embedding_path = 'embedding5@' + str(negative) + '#' + str(vector_size) + '.wordvectors'
            # word_vectors.save(embedding_path)

            # Store just the words + their trained embeddings
            # wv = KeyedVectors.load(embedding_path, mmap='r')

            with open('./' + CITY + '/vectors/embedding5@' + str(negative) + '#' + str(vector_size) + '.txt', 'w') as f:
                categories, _ = get_category_used(TSMC_CATEGORIES_USED_DIR)
                for category in categories:
                    vector = word_vectors[category.replace(' ', '_')]
                    vector = [str(v) for v in vector]
                    vector = ','.join(vector)
                    f.write(category + ',')
                    f.write(vector)
                    f.write('\n')


def baseline2():
    vector_size_list = [10, 50, 100]

    for vector_size in tqdm(vector_size_list):
        for negative in range(1, 11):
            model = word2vec.Word2Vec(sentences=sentences, window=5,
                                      size=vector_size,
                                      sg=1,
                                      negative=negative,
                                      hs=0,
                                      min_count=1)
            word_vectors = model.wv
            # embedding_path = 'embedding5@' + str(negative) + '#' + str(vector_size) + '.wordvectors'
            # word_vectors.save(embedding_path)

            # Store just the words + their trained embeddings
            # wv = KeyedVectors.load(embedding_path, mmap='r')

            with open('./' + CITY + '/vectors/embedding5@' + str(negative) + '#' + str(vector_size) + '.txt', 'w') as f:
                categories, _ = get_category_used(TSMC_CATEGORIES_USED_DIR)
                for category in categories:
                    vector = word_vectors[category.replace(' ', '_')]
                    vector = [str(v) for v in vector]
                    vector = ','.join(vector)
                    f.write(category + ',')
                    f.write(vector)
                    f.write('\n')

def get_category_sequence_for_gensim():
    """
    生成category_sequence_for_gensim_my.txt文件 生成之后将空格替换为_后将,替换为空格，适配gensim输入要求。
    :return:
    """
    with open(TSMC_CHECKIN_TRAJECTORY_DIR, 'r', encoding='utf-8') as f:
        transition_sequences = f.readlines()
    with open('./category_sequence_for_gensim_my.txt', 'w') as f:
        for sequence in tqdm(transition_sequences, desc='get center and context'):
            transitions = sequence.split(',')[1:]
            category_list = []
            for i, cur_pos in enumerate(transitions):
                category_info, _ = cur_pos.split('@')[0], int(cur_pos.split('@')[1])
                category = category_info.split('#')[1]
                category_list.append(category)

            category_sequence = ','.join(category_list)
            f.write(category_sequence)
            f.write('\n')


def write_category_to_file():
    """
    生成category_used.txt文件
    :return:
    """
    csv_data = pd.read_csv(TSMC_VENUE_INFO_DIR)
    category_list = csv_data.loc[:]['category'].tolist()
    category_set = set(category_list)
    with open(TSMC_CATEGORIES_USED_DIR, 'w', encoding='utf-8') as f:
        for category in category_set:
            f.write(category)
            f.write('\n')


def main():
    baseline2()


if __name__ == '__main__':
    main()
