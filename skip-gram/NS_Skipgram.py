import numpy as np
import random
import time
from tqdm import tqdm
from math import e

neg_num=5 #负采样数量
lr=0.025  #学习率
epoch=20  #迭代次数

class Skipgram:
    def __init__(self,input_path,context_emb_path,target_emb_path,dim):
        self.epoch=epoch
        self.neg_num = neg_num
        self.vecSize = dim
        self.lr=lr
        self.context_emb_path=context_emb_path
        self.target_emb_path=target_emb_path
        self.input_path=input_path
        self.target_set = set()
        self.contextSet = set()
        self.trainList=list()
        self.context2Vec=dict()
        self.target2Vec=dict()
        self.targetFrequencyList=list()
        self.get_trainList()
        self.computeTargetFrequencyList()

    def get_trainList(self): #读文件中的check_insequence，初始化训练序列
        with open(self.input_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                self.trainList.append(line)

    # 对向量进行初始化，采用高斯分布
    def initialize(self):
        for target in self.target_set:
            vec=np.random.normal(0,1,self.vecSize)*0.01
            self.target2Vec[target] = vec
        for context in self.contextSet:
            vec=np.random.normal(0,1,self.vecSize)*0.01
            self.context2Vec[context] = vec


    def trainModel(self):
        itrNum=0 # 统计迭代次数
        while 1 :
            print('The ' + str(itrNum + 1) + 'th iteration starts.')
            random.shuffle(self.trainList)  #对trainlist中的数据进行shuffle
            print('Training data finishes shuffling.')
            time1=time.time()
            loss=[]
            # 对trainlist进行遍历
            for each in tqdm(self.trainList, desc='epoch:' + str(itrNum + 1)):  # 对trainlist进行遍历
                l=0.0 #用来记录本次循环中的loss
                each = each.strip().split(',')  # 将target和context分开
                target = each[0]  # 取target
                postiveTargetVec = self.target2Vec[target]  # 取target对应的向量
                contextTemp = each[1].split('#')  # 将context存在contextTemp列表中
                # 遍历contexTemp
                for context in contextTemp:
                    contextvec = self.context2Vec[context] # V(w)
                    e=np.zeros(len(contextvec),dtype=float) # 初始化e

                    # 处理positive target,以下过程参照《中文详解》，与里面的过程是一致的
                    mul_vec=np.multiply(postiveTargetVec,contextvec)
                    VwThetau=np.sum(mul_vec)
                    q=self.sigmoid(VwThetau)
                    g=self.lr*(1-q)
                    e+=g*postiveTargetVec #更新e
                    postiveTargetVec+=g*contextvec #更新target
                    l+=np.log(q)

                    # 处理negative target:
                    negativeTargetTemp=self.getNegativeCategory(target) #获取负采样
                    for k in range(len(negativeTargetTemp)):   # 遍历负采样
                        negativeTarget=negativeTargetTemp[k]
                        negativeTargetVec=self.target2Vec[negativeTarget]
                        mul_vec1=np.multiply(negativeTargetVec,contextvec)  # negativeTargetVec和contextvec对应位置相乘以求点积
                        VwThetau1=np.sum(mul_vec1)  #得到点积
                        q1=self.sigmoid(VwThetau1)
                        g1=self.lr*(0-q1)
                        e+=g1*negativeTargetVec
                        negativeTargetVec+=g1*contextvec  #更新
                        l += np.log(1-q1)  #记录损失
                    contextvec+=e  #更新contex向量
                    loss.append(l)
            loss=np.array(loss)
            print('loss=',np.mean(loss))  # 输出loss的均值
            itrNum+=1
            if itrNum>=float(self.epoch) :
                break
            time2=time.time()
            runtime=time2-time1
            print(runtime,'seconds')  #输出运行时间，单位:秒

            if itrNum % 2 == 0:   #学习率每迭代两次减半
                self.lr/=2
            if itrNum < 0.0000025: #学习率下降到0.0000025以下，便不再减少
                itrNum = 0.0000025

    #保存模型
    def saveModel(self,VecMap,path):
        for t in VecMap:
            vec=VecMap[t]
            veclist=vec.tolist()
            vec=''.join(map(lambda x:','+str(x),veclist))
            with open(path,'a',encoding='utf-8') as f:
                f.write(str(t)+vec+'\n')

    # 获取负采样
    def getNegativeCategory(self,target):
        negativeCategoryList=list() #对target 取的负采样
        sampleCount = 0 #当前的采样的个数
        count = 0
        while 1:
            count+=1
            index=min(len(self.targetFrequencyList)-1,  np.random.random()*len(self.targetFrequencyList))
            index=int(index)
            sampleTargetCategory=self.targetFrequencyList[index]
            if sampleTargetCategory!=target:
                negativeCategoryList.append(sampleTargetCategory)
                sampleCount+=1
            if sampleCount==self.neg_num or count>50: #负采样采够了就停止
                break
        return negativeCategoryList  #返回负采样，为一个列表

    #根据词频得到负采样的序列
    def computeTargetFrequencyList(self):
        candidateTargetCountMap=dict()
        for each in self.trainList:
            each = each.strip().split(',')
            target = each[0]
            if target in candidateTargetCountMap:
                candidateTargetCountMap[target] += 1
            else:
                candidateTargetCountMap[target] = 1
            self.target_set.add(target)
            firstContext=each[1]
            temp=firstContext.split('#')
            for t in temp:
                self.contextSet.add(t)
        min=10000
        for tar in candidateTargetCountMap:
            if candidateTargetCountMap[tar]<min:
                min=candidateTargetCountMap[tar]

        for tar in candidateTargetCountMap:
            count=candidateTargetCountMap[tar]
            newcount=int((count/(min+0.0))**0.75)
            for i in range(newcount):
                self.targetFrequencyList.append(tar)


    def sigmoid(self,x):
            return 1. / (1. + np.exp(-x))


    def Process(self):
        self.initialize()
        self.trainModel()
        self.saveModel(self.context2Vec, self.context_emb_path)
        self.saveModel(self.target2Vec, self.target_emb_path)

if __name__ == '__main__':
    dim=50
    cb=Skipgram(input_path='data/target_context_win5.txt',context_emb_path='result_neg5/context_vec'+str(dim)+'.txt',target_emb_path='result_neg5/target_vec'+str(dim)+'.txt',dim=dim)
    cb.Process()





