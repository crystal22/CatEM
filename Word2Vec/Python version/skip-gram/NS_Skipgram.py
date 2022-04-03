import numpy as np
import random
from tqdm import tqdm
from math import log2, exp

EXP_TABLE_SIZE = 1000
MAX_EXP = 6
expTable = np.zeros(EXP_TABLE_SIZE)


def getTrainList(inputPath):
    """
    从文件中获取训练数据
    :param inputPath: 训练数据文件路径 文件中存储格式为 target,context1#context2#context3#...#contextn
    :return:
    """
    trainList = []
    with open(inputPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            trainList.append(line)
    return trainList


def initialize(categorySet, vecSize):
    """
    使用高斯分布乘以0.01 初始化target,context向量
    :param categorySet: 位置类别的集合
    :param vecSize: 向量的长度
    """
    target2Vec, context2Vec = {}, {}
    for category in categorySet:
        vec = np.random.normal(0, 1, vecSize) * 0.01
        target2Vec[category] = vec
        vec = np.random.normal(0, 1, vecSize) * 0.01
        context2Vec[category] = vec
    return target2Vec, context2Vec


def trainModel(trainList, targetVectors, contextVectors, targetPath, contextPath, targetFrequencyList):
    """
    模型训练，参照《word2vec中的数学原理》
    :param trainList: 训练数据的列表集合 每一条数据格式为 'target,context1#context2#context3#...#contextn'
    :param targetVectors: target向量字典
    :param contextVectors: context向量字典
    :param targetPath: 保存target向量字典路径
    :param contextPath: 保存context向量字典路径
    :param targetFrequencyList: 用负采样的target频率列表
    """
    itrNum = 0
    negativeNum = 5
    lr = 0.025
    epoch = 20
    while 1:
        print('The ' + str(itrNum + 1) + 'th iteration starts.')
        random.shuffle(trainList)
        print('Training data finishes shuffling.')
        loss = []

        for each in tqdm(trainList, desc='epoch:' + str(itrNum + 1)):
            epochLoss = 0.0
            each = each.strip().split(',')
            target = each[0]
            targetVector = targetVectors[target]
            contextCategory = each[1].split('#')
            negativeCategory = getNegativeCategory(target, negativeNum, targetFrequencyList)
            for context in contextCategory:
                e = np.zeros(len(targetVector), dtype=float)
                contextVector = contextVectors[context]
                vecMul = np.dot(contextVector, targetVector)
                q = getSigmoid(vecMul)
                g = lr * (1 - q)
                e += g * targetVector
                targetVectors[target] += g * contextVector
                epochLoss += log2(q)

                for negative in negativeCategory:
                    negativeVector = contextVectors[negative]
                    vecMul = np.dot(contextVector, negativeVector)
                    q = getSigmoid(vecMul)
                    g = lr * (-q)
                    e += g * negativeVector
                    targetVectors[negative] += g * contextVector
                    epochLoss += log2(1 - q)
                contextVectors[context] += e
            loss.append(-epochLoss)
        loss = np.array(loss)
        print('loss=', loss.mean())
        itrNum += 1
        if itrNum >= epoch:
            break
        if itrNum % 2 == 0:
            lr /= 2
        if itrNum < 0.0000025:
            itrNum = 0.0000025

    saveModel(targetVectors, targetPath)
    saveModel(contextVectors, contextPath)


def saveModel(categoryVector, path):
    """
    保存vector向量
    """
    with open(path, 'a', encoding='utf-8') as f:
        for t in categoryVector.keys():
            vector = categoryVector[t].tolist()
            vector = ''.join(map(lambda x: ',' + str(x), vector))
            f.write(str(t) + vector + '\n')


def getNegativeCategory(target, negativeNum, targetFrequencyList):
    """
    生成负采样
    :param target:
    :param negativeNum: 负采样的个数
    :param targetFrequencyList: target频率列表 target出现频率高，被选为负样本的概率就越高
    """
    negativeCategoryList = []
    sampleCount, count = 0, 0
    while 1:
        count += 1
        index = min(len(targetFrequencyList) - 1, np.random.random() * len(targetFrequencyList))
        index = int(index)
        sampleTargetCategory = targetFrequencyList[index]
        if sampleTargetCategory != target:
            negativeCategoryList.append(sampleTargetCategory)
            sampleCount += 1
        if sampleCount == negativeNum or count > 50:
            break

    return negativeCategoryList


def computeTargetFrequencyList(trainList):
    """
    统计target出现次数，生成频率列表，用于带权重的负采样。
    """
    categorySet = set()
    targetFrequencyList = []
    candidateTargetCountMap = dict()
    for each in trainList:
        each = each.strip().split(',')
        target = each[0]
        if target in candidateTargetCountMap:
            candidateTargetCountMap[target] += 1
        else:
            candidateTargetCountMap[target] = 1
        categorySet.add(target)

    minCount = 10000
    for tar in candidateTargetCountMap:
        if candidateTargetCountMap[tar] < minCount:
            minCount = candidateTargetCountMap[tar]

    for tar in candidateTargetCountMap:
        count = candidateTargetCountMap[tar]
        newCount = int((count / (minCount + 0.0)) ** 0.75)
        for i in range(newCount):
            targetFrequencyList.append(tar)

    return categorySet, targetFrequencyList


def createExpTable():
    for i in range(EXP_TABLE_SIZE):
        expTable[i] = exp(((i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP))
        expTable[i] = expTable[i] / (expTable[i] + 1)


def getSigmoid(z):
    if z >= 8:
        return 0.9999
    if z <= -8:
        return 0.0001
    sigmoidZ = 1 / (1 + exp(-z))
    if sigmoidZ <= 0.0001:
        sigmoidZ = 0.0001
    if sigmoidZ >= 0.9999:
        sigmoidZ = 0.9999
    return sigmoidZ


def main():
    inputPath = 'E:\\PycharmProjects\\TSMC\\output\\NYC\\center_context_with_sequence.txt'
    vecSize = 100
    contextPath = 'E:\\PycharmProjects\\TSMC\\output\\NYC\\context_vector' + str(vecSize) + '.txt'
    targetPath = 'E:\\PycharmProjects\\TSMC\\output\\NYC\\target_vector' + str(vecSize) + '.txt'
    createExpTable()
    trainList = getTrainList(inputPath)
    categorySet, targetFrequencyList = computeTargetFrequencyList(trainList)
    targetVector, contextVector = initialize(categorySet, vecSize)
    trainModel(trainList, targetVector, contextVector, targetPath, contextPath, targetFrequencyList)


if __name__ == '__main__':
    main()
