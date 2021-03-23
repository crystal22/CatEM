import numpy as np
import random
import time
from tqdm import tqdm
from math import e

negativeNum=5
learnRate=0.025
epoch=20


class Skipgram:
    def __init__(self,inputPath,contextPath,targetPath,dim):
        self.epoch=epoch
        self.negativeNum = negativeNum
        self.vecSize = dim
        self.learnRate=learnRate
        self.contextPath=contextPath
        self.targetPath=targetPath
        self.inputPath=inputPath
        self.targetSet = set()
        self.contextSet = set()
        self.trainList=list()
        self.contextVecMap=dict()
        self.targetVecMap=dict()
        self.targetFrequencyList=list()
        self.get_trainList()
        self.computeTargetFrequencyList()
    def get_trainList(self):
        with open(self.inputPath,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                self.trainList.append(line)

    def initialize(self):
        for target in self.targetSet:
            vec=np.random.normal(0,1,self.vecSize)*0.01
            self.targetVecMap[target] = vec
        for context in self.contextSet:
            vec=np.random.normal(0,1,self.vecSize)*0.01
            self.contextVecMap[context] = vec


    def trainModel(self):
        itrNum=0
        while 1 :
            print('The ' + str(itrNum + 1) + 'th iteration starts.')
            random.shuffle(self.trainList)
            print('Training data finishes shuffling.')
            time1=time.time()
            loss=[]
            for each in tqdm(self.trainList,desc='epoch:'+str(itrNum+1)):
                each=each.strip().split(',')
                target=each[0]
                postiveTargetVec = self.targetVecMap[target]
                contextTemp = each[1].split('#')
                for context in contextTemp:
                    contextvec = self.contextVecMap[context] # V(w)
                    e=np.zeros(len(contextvec),dtype=float)
                    # positive target
                    mul_vec=np.multiply(postiveTargetVec,contextvec)
                    VwThetau=np.sum(mul_vec)
                    q=self.sigmoid(VwThetau)
                    g=self.learnRate*(1-q)
                    e+=g*postiveTargetVec
                    postiveTargetVec+=g*contextvec
                    loss.append(np.log(q))
                    #negative target:
                    negativeTargetTemp=self.getNegativeCategory(target)
                    for k in range(len(negativeTargetTemp)):
                        negativeTarget=negativeTargetTemp[k]
                        negativeTargetVec=self.targetVecMap[negativeTarget]
                        mul_vec1=np.multiply(negativeTargetVec,contextvec)
                        VwThetau1=np.sum(mul_vec1)
                        q1=self.sigmoid(VwThetau1)
                        g1=self.learnRate*(0-q1)
                        e+=g1*negativeTargetVec
                        negativeTargetVec+=g1*contextvec
                        loss.append(np.log(1-q1))
                    contextvec+=e
            loss=np.array(loss)
            print('loss=',np.mean(loss))
            itrNum+=1
            if itrNum>=float(self.epoch) :
                break
            time2=time.time()
            runtime=time2-time1
            print(runtime,'seconds')

            if itrNum % 2 == 0:
                self.learnRate/=2
            if itrNum < 0.0000025:
                itrNum = 0.0000025

    def saveModel(self,VecMap,path):
        for t in VecMap:
            vec=VecMap[t]
            veclist=vec.tolist()
            vec=''.join(map(lambda x:','+str(x),veclist))
            with open(path,'a',encoding='utf-8') as f:
                f.write(str(t)+vec+'\n')
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
            if sampleCount==self.negativeNum or count>50:
                break
        return negativeCategoryList




    def computeTargetFrequencyList(self):
        candidateTargetCountMap=dict()
        for each in self.trainList:
            each = each.strip().split(',')
            target = each[0]
            if target in candidateTargetCountMap:
                candidateTargetCountMap[target] += 1
            else:
                candidateTargetCountMap[target] = 1
            self.targetSet.add(target)
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
        self.saveModel(self.contextVecMap, self.contextPath)
        self.saveModel(self.targetVecMap, self.targetPath)

if __name__ == '__main__':
    # for i in range(10,101,10):
    #     cb=Skipgram(inputPath='data/target_context_win5.txt',contextPath='result_neg5/context_vec'+str(i)+'.txt',targetPath='result_neg5/target_vec'+str(i)+'.txt',dim=i)
    #     cb.Process()
    dim=50
    cb=Skipgram(inputPath='data/target_context_win5.txt',contextPath='result_neg5/context_vec'+str(dim)+'.txt',targetPath='result_neg5/target_vec'+str(dim)+'.txt',dim=dim)
    cb.Process()





