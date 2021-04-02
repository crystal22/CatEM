import numpy as np
import random
import time
from tqdm import tqdm
from math import log2

negativeNum = 5
learnRate = 0.025
epoch = 20


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def saveModel(VecMap, path):
    for t in VecMap:
        vector = VecMap[t].tolist()
        vector = ''.join(map(lambda x: ',' + str(x), vector))
        with open(path, 'a', encoding='utf-8') as f:
            f.write(str(t) + vector + '\n')


class CBOW:
    def __init__(self, inputPath, contextPath, targetPath, dim):
        self.epoch = epoch
        self.negativeNum = negativeNum
        self.vecSize = dim
        self.learnRate = learnRate
        self.contextPath = contextPath
        self.targetPath = targetPath
        self.inputPath = inputPath
        self.categorySet = set()
        self.trainList = list()
        self.contextVectors = dict()
        self.targetVectors = dict()
        self.targetFrequencyList = list()
        self.getTrainList()
        self.computeTargetFrequencyList()

    def getTrainList(self):
        """
        在文件中获取trainList，文件中的格式是target,context1#context2#context3
        """
        with open(self.inputPath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                self.trainList.append(line)

    def initialize(self):
        """
        使用高斯分布乘以0.01 初始化target,context向量
        """
        for category in self.categorySet:
            vec = np.random.normal(0, 1, self.vecSize) * 0.01
            self.targetVectors[category] = vec
            vec = np.random.normal(0, 1, self.vecSize) * 0.01
            self.contextVectors[category] = vec

    def trainModel(self):
        itrNum = 0
        while 1:
            print('The ' + str(itrNum + 1) + 'th iteration starts.')
            random.shuffle(self.trainList)  # 对trainList中所有序列打乱顺序
            print('Training data finishes shuffling.')
            time1 = time.time()
            loss = []
            for each in tqdm(self.trainList, desc='epoch:' + str(itrNum + 1)):
                epochLoss = 0
                each = each.strip().split(',')
                target = each[0]
                targetVector = self.targetVectors[target]
                contextCategory = each[1].split('#')
                contextVector = np.zeros(self.vecSize, dtype=float)
                negativeCategory = self.getNegativeCategory(target)

                for context in contextCategory:  # 求出上下文向量之和Xw
                    contextVector += self.contextVectors[context]
                e = np.zeros(self.vecSize, dtype=float)

                vecMul = np.dot(targetVector, contextVector)
                q = sigmoid(vecMul)
                g = self.learnRate * (1 - q)
                e += g * targetVector
                targetVector += g * contextVector
                self.targetVectors[target]=targetVector
                epochLoss += log2(q)  # 记录loss

                for negative in negativeCategory:
                    negativeVector = self.targetVectors[negative]
                    vecMul = np.dot(negativeVector, contextVector)

                    q = sigmoid(vecMul)
                    g = self.learnRate * (-q)
                    e += g * negativeVector
                    negativeVector += g * contextVector
                    self.targetVectors[negative]= negativeVector
                    epochLoss += np.log(1 - q)

                for context in contextCategory:
                    contextVector = self.contextVectors[context]
                    contextVector += e
                    self.contextVectors[context]=contextVector

                loss.append(epochLoss)

            loss = np.array(loss)
            print('loss=', loss.mean())
            itrNum += 1
            if itrNum >= self.epoch:
                break
            time2 = time.time()
            runtime = time2 - time1
            print(runtime, 'seconds')
            if itrNum % 2 == 0:
                self.learnRate /= 2
            if itrNum < 0.0000025:
                itrNum = 0.0000025

    def getNegativeCategory(self, target):
        negativeCategoryList = list()
        sampleCount = 0
        count = 0
        while 1:
            count += 1
            index = min(len(self.targetFrequencyList) - 1, np.random.random() * len(self.targetFrequencyList))
            index = int(index)
            sampleTargetCategory = self.targetFrequencyList[index]
            if sampleTargetCategory != target:  # 在target以外取样
                negativeCategoryList.append(sampleTargetCategory)
                sampleCount += 1
            if sampleCount == self.negativeNum or count > 50:
                break
        return negativeCategoryList

    def computeTargetFrequencyList(self):
        candidateTargetCountMap = dict()
        for each in self.trainList:
            each = each.strip().split(',')
            target = each[0]
            if target in candidateTargetCountMap:
                candidateTargetCountMap[target] += 1
            else:
                candidateTargetCountMap[target] = 1
            self.categorySet.add(target)

        minCount = 10000
        for tar in candidateTargetCountMap:
            if candidateTargetCountMap[tar] < minCount:
                minCount = candidateTargetCountMap[tar]  # 求出最小的次单词出现次数

        for tar in candidateTargetCountMap:
            count = candidateTargetCountMap[tar]
            newCount = int((count / (minCount + 0.0)) ** 0.75)  # 按照频率的0.75次方构造负采样列表
            for i in range(newCount):
                self.targetFrequencyList.append(tar)  # 得到按词频进行分布的负采样列表

    def process(self):
        self.initialize()
        self.trainModel()
        saveModel(self.contextVectors, self.contextPath)
        saveModel(self.targetVectors, self.targetPath)


if __name__ == '__main__':
    dim = 50
    cb = CBOW(inputPath='', contextPath='' + str(dim) + '.txt',
              targetPath='' + str(dim) + '.txt', dim=dim)
    cb.process()
