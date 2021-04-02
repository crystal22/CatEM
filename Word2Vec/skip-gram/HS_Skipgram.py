from tqdm import tqdm
from collections import Counter
import math
import numpy as np
from sklearn.utils import shuffle

min_count=1
embed_size = 50
embed_epoch = 12
DECAY=0.5
window_size = 5
indi_context = False
init_lr = 0.025

input_file_name='E:\jupyer notebook\cbow\cate_sequences.txt'
output_file_name1="E:\jupyer notebook\cbow\SG_u_embedding.txt"
output_file_name2="E:\jupyer notebook\cbow\SG_embedding.txt"

class W2VData:
    def __init__(self, sentences, indi_context):
        self.indi_context = indi_context
        self.word_freq = get_token_freq(sentences)  # 统计词频

class HSData(W2VData):
    """
    Data supporter for Hierarchical Softmax.
    """
    def __init__(self, sentences, indi_context):  ##数据预处理
        super().__init__(sentences, indi_context)   ##统计词频
        self.sentences = sentences
        self.huffman_tree = HuffmanTree(self.word_freq)   ##构建哈夫曼树

    def get_path_pairs(self, window_size):       ##根据target构造[contest、pos节点、neg节点]序列
        path_pairs = []
        for sentence in self.sentences:
            for i in range(window_size, len(sentence) -  window_size  + 1):
                target = sentence[i]
                for j in range(i-window_size,i+window_size):
                    if j ==i:
                        continue
                    context=sentence[j]
                    pos_path = self.huffman_tree.id2pos[context]
                    neg_path = self.huffman_tree.id2neg[context]
                    path_pairs.append([target, pos_path, neg_path])
        return path_pairs


class HuffmanNode:
    """
    A node in the Huffman tree.
    """
    def __init__(self, id, frequency):
        """
        :param id: index of word (leaf nodes) or inner nodes.
        :param frequency: frequency of word.
        """
        self.id = id
        self.frequency = frequency
        self.left = None
        self.right = None
        self.father = None
        self.huffman_code = []
        self.path = []  # (path from root node to leaf node)

    def __str__(self):
        return 'HuffmanNode#{},freq{}'.format(self.id, self.frequency)


class HuffmanTree:
    """
    Huffman Tree class used for Hierarchical Softmax calculation.
    """
    def __init__(self, freq_array):
        """
        :param freq_array: numpy array containing all words' frequencies, format {id: frequency}.
        """
        self.num_words = freq_array.shape[0]
        self.id2code = {}       ##节点id2编码（0 or 1）
        self.id2path = {}
        self.id2pos = {}
        self.id2neg = {}
        self.root = None  # Root node of this tree.
        self.num_inner_nodes = 0  # 记录内部节点数量
        unmerged_node_list = [HuffmanNode(id, frequency) for id, frequency in freq_array]   ##生成叶子节点
        self.tree = {node.id: node for node in unmerged_node_list}
        self.id_offset = max(self.tree.keys())  # tree开始的id.
        self._offset = self.id_offset
        self._build_tree(unmerged_node_list)  ##建树
        self._get_path()     ##得到所有从根节点出发到叶子节点的路径以及编码
        self._get_all_pos_neg()   ##id2pos and id2neg 注意这里的id是减去了叶子节点数之后的id（内部节点的id）

    def _merge_node(self, node1: HuffmanNode, node2: HuffmanNode):
        """
        Merge two nodes into one, adding their frequencies.
        """
        sum_freq = node1.frequency + node2.frequency
        self._offset += 1
        mid_node_id = self._offset
        father_node = HuffmanNode(mid_node_id, sum_freq)
        if node1.frequency >= node2.frequency:
            father_node.left, father_node.right = node1, node2
        else:
            father_node.left, father_node.right = node2, node1
        self.tree[mid_node_id] = father_node
        self.num_inner_nodes += 1
        return father_node

    def _build_tree(self, node_list):
        """
        :param node_list:
        """
        while len(node_list) > 1:
            i1, i2 = 0, 1
            if node_list[i2].frequency < node_list[i1].frequency:
                i1, i2 = i2, i1
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        i1, i2 = i2, i1
            father_node = self._merge_node(node_list[i1], node_list[i2])
            assert not i1 == i2
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            else:
                node_list.pop(i1)
                node_list.pop(i2)
            node_list.insert(0, father_node)
        self.root = node_list[0]

    def _get_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.huffman_code
                path = node.path
                node.left.huffman_code = code + [1]
                node.right.huffman_code = code + [0]
                node.left.path = path + [node.id]
                node.right.path = path + [node.id]
                stack.append(node.right)
                node = node.left
            id = node.id
            code = node.huffman_code
            path = node.path
            self.tree[id].huffman_code, self.tree[id].path = code, path
            self.id2code[id], self.id2path[id] = code, path

    def _get_all_pos_neg(self):
        for id in self.id2code.keys():
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.tree[id].huffman_code):
                if code == 1:
                    pos_id.append(self.tree[id].path[i] - self.id_offset)  # This will make the generated inner node IDs starting from 1.
                else:
                    neg_id.append(self.tree[id].path[i] - self.id_offset)
            self.id2pos[id] = pos_id
            self.id2neg[id] = neg_id

def get_token_freq(sentences):
    """
    :param sentences:
    :return: the sorted frequency of words.
    """
    freq = Counter()
    for sentence in sentences:
        freq.update(sentence)
    freq = np.array(sorted(freq.items()))
    return freq
def train_skip(train_set,num_epoch, init_lr, num_vocab, embed_dimension):
    """
    :param train_set: 存储的数据,存储形式：每行为一个target的[context,pos,neg]
    :param num_epoch:迭代次数
    :param init_lr:学习率
    :param num_vocab:单词数
    :param embed_dimension:维度
    :return: u_embedding,即单词向量的array形式
    """
    u_embedding = np.random.rand(num_vocab,embed_dimension)   #向量初始化  叶子节点的向量
    w_embedding = np.random.rand(num_vocab,embed_dimension)   #初始化   内部节点的向量
    #u_embedding=np.random.normal(0, 0.01, (num_vocab, embed_dimension))
    #w_embedding = np.random.normal(0, 0.01, (num_vocab, embed_dimension))
    lr = init_lr
    trained = 0
    for epoch in range(num_epoch):
        train_set = shuffle(train_set)  ##乱序
        loss_log = []
        for pair in tqdm(train_set):  ##一行数据，即处理一个target的[context,pos节点,neg节点]
            loss=0
            target, pos_pairs, neg_pairs =pair
            e = np.zeros(embed_size)
            for j in pos_pairs:
                z = np.dot(u_embedding[target],w_embedding[j])     #将整合后的neul也就是xω和target到root的路径节点相乘
                q=sigmoid(z) #使用sigmoid激活，会得到第target个中间节点选择1还是0
                g=lr*(-q) #alpha学习率，根据已知的djω编码{0,1}，p是sigmoid(z)
                e += g*w_embedding[j] #e记录的是传递给xω的误差
                w_embedding[j]+=g*u_embedding[target]  ##内部节点更新
                loss+=math.log2(1-q)  ##计算loss
            for j in neg_pairs:
                z = np.dot(u_embedding[target],w_embedding[j])    #将整合后的neul也就是xω和target到root的路径节点相乘
                q=sigmoid(z) #使用sigmoid激活，会得到第target个中间节点选择1还是0
                g=lr*(1-q) #alpha学习率、根据已知的djω编码{0,1}，p是sigmoid(z)
                e += g * w_embedding[j]
                w_embedding[j]+=g*u_embedding[target]  ##内部节点更新
                loss += math.log2(q)   ##计算loss
            u_embedding[target]+=e   ##叶子节点，也就是这个target的所有context进行更新
            loss_log.append(-loss)
            trained+=1
        print('Epoch %d avg loss: %.5f' % (epoch, np.mean(loss_log)))    ##输出loss值
        #lr = init_lr -(init_lr-0.00001)* ( trained_batches / embed_epoch*batch_count)
        lr = init_lr*DECAY**epoch     ##可变学习率变化方法
        if lr<=0.000025:
            lr=0.000025
    return np.array(u_embedding),np.array(w_embedding)   ##返回叶子节点的向量


def sigmoid(z):  #sigmoid激活函数
    return 1 / (1 + math.exp(-z))

def HS_skipgram(input_file_name,output_file_name1,output_file_name2,embed_size,window_size,embed_epoch, init_lr,decay,indi_context):
    f = open(input_file_name, "r")  # 设置文件对象
    all_paper = f.read().replace("'", "").replace('"', '')
    sentences = all_paper.split("\n")  # 将txt文件的所有内容读入到数组中
    f.close()
    print('Window size {}, initial lr={}'.format(window_size,init_lr))
    word2id = {}  ##单词用id记录
    for sen in sentences:
        words = sen.split(" ")  ##我这里存储形式是用空格隔开不同的词，可根据输入文件实际情况进行修改。
        for word in words:
            if word not in word2id:
                word2id[word] = len(word2id)
    id2word = {word2id[key]: key for key in word2id}
    id_sentences = []
    for sen in sentences:
        words = sen.split(" ")
        id_sentences.append([word2id[word] for word in words])  # 从单词序列转为id序列
    embed_dataset = HSData(sentences=id_sentences, indi_context=indi_context)  ##数据预处理
    train_set = embed_dataset.get_path_pairs(window_size)  ##获取数据集
    embed_u,embed_w = train_skip(train_set, embed_epoch, init_lr, len(word2id), embed_size)  ##模型训练
    file_output = open(output_file_name1, 'w')  # 打开要保存的文件路径
    for id, word in id2word.items():
        e = embed_u[id]
        e = ' '.join(map(lambda x: str(x), e))  # 把word名放在向量最前面
        file_output.write('%s %s\n' % (word, e))
    file_output.close()
    file_output = open(output_file_name2, 'w')  # 打开要保存的文件路径
    for id, word in id2word.items():
        e = embed_w[id]
        e = ' '.join(map(lambda x: str(x), e))  # 把word名放在向量最前面
        file_output.write('%s %s\n' % (word, e))
    file_output.close()

if __name__ == '__main__':
    u_embed_mat,w_embed_mat = HS_skipgram(input_file_name, output_file_name1,output_file_name2, embed_size, window_size, embed_epoch, init_lr, DECAY,indi_context)