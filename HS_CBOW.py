from tqdm import tqdm
from collections import Counter
import math
from itertools import zip_longest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils import shuffle
from utils import next_batch

min_count=1
embed_size = 50
embed_batch_size = 64
embed_epoch = 12
embed_name = 'p2v'
DECAY=0.5
window_size = 4
indi_context = False
init_lr = 0.025
embed_adam = False  # Set True to use Adam optimizer, or else SGD.
device = 'cpu'  # 'cuda:0'
dataset_name = 'mynyc'
input_file_name='C:/Users/dell/Desktop/nyc_sequence/cate_sequences.txt'
output_file_name="E:\jupyer notebook\cbow\category_embedding.txt"

class W2VData:
    def __init__(self, sentences, indi_context):
        self.indi_context = indi_context
        self.word_freq = gen_token_freq(sentences)  # 统计词频

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
            for i in range(0, len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                pos_path = self.huffman_tree.id2pos[target]
                neg_path = self.huffman_tree.id2neg[target]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    path_pairs += [[[c], pos_path, neg_path] for c in context]
                else:
                    path_pairs.append([context, pos_path, neg_path])
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
        #print(self.id_offset)
        # Because the ID of leaf nodes will not be needed during calculation,
        # you can minus this value to all inner nodes' IDs to save some space in output embeddings.

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


class HS(nn.Module):
    """
    A Hierarchical Softmax variation of Word2Vec. Can only operate in CBOW mode.
    """
    def __init__(self, num_vocab, embed_dimension):
        super().__init__()
        self.num_vocab = num_vocab
        self.embed_dimension = embed_dimension

        # Input embedding.
        self.u_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)
        # Output embedding. Here is actually the embedding of inner nodes.
        self.w_embeddings = nn.Embedding(num_vocab, embed_dimension, padding_idx=0, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.w_embeddings.weight.data.uniform_(-0, 0)
        #self.w_embeddings.weight.data = torch.Tensor(np.random.normal(0, 0.01, (num_vocab, embed_dimension)))

    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        @param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        @param pos_w: positive output tokens, shape (batch_size, num_pos)
        @param neg_w: negative output tokens, shape (batch_size, num_neg)
        @param sum: whether to sum up all scores.
        """
        pos_u_embed = self.u_embeddings(pos_u)  # (batch_size, window_size * 2, embed_size)
        pos_u_embed = pos_u_embed.sum(1, keepdim=True)  # (batch_size, 1, embed_size)

        pos_w_mask = torch.where(pos_w == 0, torch.ones_like(pos_w), torch.zeros_like(pos_w)).bool()  # (batch_size, num_pos)
        pos_w_embed = self.w_embeddings(pos_w)  # (batch_size, num_pos, embed_size)
        score = torch.mul(pos_u_embed, pos_w_embed).sum(dim=-1)  # (batch_size, num_pos)
        score = F.logsigmoid(-1 * score)  # (batch_size, num_pos)
        score = score.masked_fill(pos_w_mask, torch.tensor(0.0).to(pos_u.device))

        neg_w_mask = torch.where(neg_w == 0, torch.ones_like(neg_w), torch.zeros_like(neg_w)).bool()
        neg_w_embed = self.w_embeddings(neg_w)
        neg_score = torch.mul(pos_u_embed, neg_w_embed).sum(dim=-1)  # (batch_size, num_neg)
        neg_score = F.logsigmoid(neg_score)
        neg_score = neg_score.masked_fill(neg_w_mask, torch.tensor(0.0).to(pos_u.device))
        if kwargs.get('sum', True):
            return -1 * (torch.sum(score) + torch.sum(neg_score))
        else:
            print("error")
            return score, neg_score

def gen_token_freq(sentences):
    freq = Counter()
    for sentence in sentences:
        freq.update(sentence)
    freq = np.array(sorted(freq.items()))
    return freq
def train_cbow(dataset,num_epoch, init_lr, num_vocab, embed_dimension):
    u_embedding = np.random.rand(num_vocab, embed_dimension)  #向量初始化  叶子节点的向量
    #u_embedding=np.random.normal(0, 0.01, (num_vocab, embed_dimension))
    w_embedding = np.random.normal(0, 0.01, (num_vocab, embed_dimension))  #初始化   内部节点的向量
    lr = init_lr
    train_set = dataset.get_path_pairs(window_size)   ##从dataset中提取数据
    train_set = shuffle(train_set)  ##乱序
    context, pos_pairs, neg_pairs = zip(*train_set)
    pair_count = num_epoch * len(train_set)
    trained = 0
    for epoch in range(num_epoch):
        loss_log = []
        for pair in tqdm(range(len(context))):   ##一行数据，即处理一个target的[context,pos节点,neg节点]
            loss=0
            neul = np.mean(u_embedding[context[pair]], axis=0)
            neu1e = np.zeros(embed_size)  # 这个是预测的target的embedding，也就是cbow中的ωt
            for j in pos_pairs[pair]:
                z = np.dot(neul,w_embedding[j])     #将整合后的neul也就是xω和target到root的路径节点相乘
                p=sigmoid(z) #使用sigmoid激活，会得到第target个中间节点选择1还是0
                g=lr*(-p) #alpha学习率，根据已知的djω编码{0,1}，p是sigmoid(z)
                neu1e += g*w_embedding[j] #neule记录的是传递给xω的误差，但是cbow模型会将此误差全部传递给每一个context中的word embedding也就是u中context_word的词向量
                w_embedding[j]+=g*neul  ##内部节点更新
                loss+=math.log2(1-p)  ##计算loss
            for j in neg_pairs[pair]:
                z = np.dot(neul,w_embedding[j])     #将整合后的neul也就是xω和target到root的路径节点相乘
                p=sigmoid(z) #使用sigmoid激活，会得到第target个中间节点选择1还是0
                g=lr*(1-p) #alpha学习率、根据已知的djω编码{0,1}，p是sigmoid(z)
                neu1e += g * w_embedding[j]
                w_embedding[j]+=g*neul  ##内部节点更新
                loss += math.log2(p)   ##计算loss
            for con in context[pair]:
                u_embedding[con]+=neu1e   ##叶子节点，也就是这个target的所有context进行更新
            loss_log.append(-loss)
            trained+=1
        print('Epoch %d avg loss: %.5f' % (epoch, np.mean(loss_log)))    ##输出loss值
        #lr = init_lr -(init_lr-0.00001)* ( trained_batches / embed_epoch*batch_count)
        lr = init_lr*DECAY**epoch     ##可变学习率变化方法
        if lr<=0.000025:
            lr=0.000025
    return np.array(u_embedding)   ##返回叶子节点的向量


def sigmoid(z):  #sigmoid激活函数
    return 1 / (1 + math.exp(-z))

def train_p2v(my_model, dataset, window_size, batch_size, num_epoch, init_lr, optim_class, device):
    my_model = my_model.to(device)
    optimizer = optim_class(my_model.parameters(), lr=init_lr)
    train_set = dataset.get_path_pairs(window_size)
    trained_batches = 0
    batch_count = math.ceil(num_epoch * len(train_set) / batch_size)
    for epoch in range(num_epoch):
        loss_log = []
        for pair_batch in tqdm(next_batch(shuffle(train_set), batch_size)):
            context, pos_pairs, neg_pairs = zip(*pair_batch)
            context = torch.tensor(context).long().to(device)
            pos_pairs, neg_pairs = (torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(device).transpose(0, 1)
                                    for item in (pos_pairs, neg_pairs))  # (batch_size, longest)
            optimizer.zero_grad()
            loss = my_model(context, pos_pairs, neg_pairs)
            loss.backward()
            optimizer.step()
            trained_batches += 1
            loss_log.append(loss.detach().cpu().numpy().tolist())

        if isinstance(optimizer, torch.optim.SGD):
            #lr = init_lr -(init_lr-0.00001)* ( trained_batches / embed_epoch*batch_count)
            lr = init_lr*DECAY**epoch
            if lr<=0.000025:
                lr=0.000025
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('Epoch %d avg loss: %.5f' % (epoch, np.mean(loss_log)))
    return my_model.u_embeddings.weight.detach().cpu().numpy()


if __name__ == '__main__':
    f = open(input_file_name, "r")  # 设置文件对象
    all_paper=f.read().replace("'","").replace('"','')
    sentences = all_paper.split("\n")  # 将txt文件的所有内容读入到数组中
    f.close()
    print('Window size {}, {}initial lr={}, use {} optimizer'.format(window_size,'individual context, ' if indi_context else '',init_lr, 'Adam' if embed_adam else 'SGD'))
    if embed_adam:
        optim_class = torch.optim.SparseAdam
    else:
        optim_class = torch.optim.SGD
    word2id={}   ##单词用id记录
    for sen in sentences:
        words=sen.split(" ")        ##我这里存储形式是用空格隔开不同的词，可根据输入文件实际情况进行修改。
        for word in words:
            if word not in word2id:
                word2id[word]=len(word2id)
    id2word={word2id[key]:key for key in word2id}
    id_sentences=[]
    for sen in sentences:
        words=sen.split(" ")
        id_sentences.append([word2id[word] for word in words])    #从单词序列转为id序列
    embed_dataset = HSData(sentences=id_sentences,indi_context=indi_context)   ##数据预处理

    #embed_model = HS(num_vocab=len(word2id), embed_dimension=embed_size)     ##这两句是自动梯度下降的方法
    #embed_mat = train_p2v(embed_model, embed_dataset, window_size=window_size, batch_size=embed_batch_size,num_epoch=embed_epoch, init_lr=init_lr, optim_class=optim_class, device=device)
    embed_mat=train_cbow(embed_dataset,embed_epoch, init_lr, len(word2id), embed_size)   ##模型训练
    file_output = open(output_file_name, 'w')   #打开要保存的文件路径
    for id, word in id2word.items():
        e = embed_mat[id]
        e = ' '.join(map(lambda x: str(x), e))      #把word名放在向量最前面
        file_output.write('%s %s\n' % (word, e))
    file_output.close()