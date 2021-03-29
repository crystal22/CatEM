import numpy as np
import random
import time
from tqdm import tqdm

negativeNum=5
learnRate=0.025
epoch=20

class CBOW:
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

    def get_trainList(self): #在文件中获取trainList，文件中的格式是target,contex1#contex2#contex3
        with open(self.inputPath,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                self.trainList.append(line)

    #初始化向量
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
            random.shuffle(self.trainList) #对trainList中所有序列打乱顺序
            print('Training data finishes shuffling.')
            time1=time.time()
            loss=[]
            for each in tqdm(self.trainList,desc='epoch:'+str(itrNum+1)):
                l=0
                each=each.strip().split(',')
                target=each[0]  #取target
                contextTemp = each[1].split('#') #得到contxt列表
                Xw = np.zeros(self.vecSize, dtype=float)
                for context in contextTemp: # 求出上下文向量之和Xw
                    Xw += self.contextVecMap[context]
                e=np.zeros(self.vecSize,dtype=float) #初始化e为一个0向量

                # positive target，以下完全参照《中文详解》
                postiveTargetVec = self.targetVecMap[target]
                mul_vec = np.multiply(postiveTargetVec, Xw)
                XwThetau = np.sum(mul_vec)
                q = self.sigmoid(XwThetau)
                g = self.learnRate * (1 - q) #正采样的标签l为1
                e += g * postiveTargetVec #e=e+g*θu
                postiveTargetVec += g * Xw  #θu=θu+g*Wx
                l += np.log(q) #记录loss

                # negative target:
                negativeTargetTemp = self.getNegativeCategory(target)
                for k in range(len(negativeTargetTemp)):
                    negativeTarget = negativeTargetTemp[k]
                    negativeTargetVec = self.targetVecMap[negativeTarget]
                    mul_vec1 = np.multiply(negativeTargetVec, Xw)
                    XwThetau1 = np.sum(mul_vec1)
                    q1 = self.sigmoid(XwThetau1)
                    g1 = self.learnRate * (0 - q1)  #负采样的标签l为0
                    e += g1 * negativeTargetVec #e=e+g*θu
                    negativeTargetVec += g1 * Xw #θu=θu+g*Wx
                    l += np.log(1 - q1) #记录loss

                for context in contextTemp: #遍历context，更新contex向量
                    contextvec=self.contextVecMap[context]
                    contextvec+=e

                loss.append(l)

            loss=np.array(loss)  #转化为array以便用mean求均值
            print('loss=',np.mean(loss)) #将loss求均值输出
            itrNum+=1
            if itrNum>=float(self.epoch) :
                break #迭代停止
            time2=time.time()
            runtime=time2-time1
            print(runtime,'seconds') #输出运行时间

            if itrNum % 2 == 0: #每隔两次，学习率减半
                self.learnRate/=2
            if itrNum < 0.0000025: #当学习率小于0.0000025时，便不再减小
                itrNum = 0.0000025

    #保存模型
    def saveModel(self,VecMap,path):
        for t in VecMap:
            vec=VecMap[t]
            veclist=vec.tolist() #将词向量转为列表
            vec=''.join(map(lambda x:','+str(x),veclist))
            with open(path,'a',encoding='utf-8') as f:
                f.write(str(t)+vec+'\n') #这里用的保存格式是：word+词向量的格式。用逗号隔开

    #获取负采样的列表
    def getNegativeCategory(self,target):
        negativeCategoryList=list() #对target 取的负采样
        sampleCount = 0 #当前的采样的个数
        count = 0
        while 1:
            count+=1
            index=min(len(self.targetFrequencyList)-1,  np.random.random()*len(self.targetFrequencyList))
            index=int(index)
            sampleTargetCategory=self.targetFrequencyList[index]
            if sampleTargetCategory!=target: #在target以外取样
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
            if target in candidateTargetCountMap: #统计词频
                candidateTargetCountMap[target] += 1
            else:
                candidateTargetCountMap[target] = 1
            self.targetSet.add(target) #顺便构造targetSet
            firstContext=each[1]
            temp=firstContext.split('#')
            for t in temp:
                self.contextSet.add(t) #构造contextSet
        min=10000
        for tar in candidateTargetCountMap:
            if candidateTargetCountMap[tar]<min:
                min=candidateTargetCountMap[tar] #求出最小的次单词出现次数


        for tar in candidateTargetCountMap:
            count=candidateTargetCountMap[tar]
            newcount=int((count/(min+0.0))**0.75) #按照频率的0.75次方构造负采样列表
            for i in range(newcount):
                self.targetFrequencyList.append(tar)  #得到按词频进行分布的负采样列表

    def sigmoid(self,x):
            return 1. / (1. + np.exp(-x))

    def Process(self): #将整个过程放在Process里，直接调用它
        self.initialize()
        self.trainModel()
        self.saveModel(self.contextVecMap, self.contextPath)
        self.saveModel(self.targetVecMap, self.targetPath)

if __name__ == '__main__':

    dim=50
    cb = CBOW(inputPath='data/target_context_win5.txt', contextPath='result/context_vec' + str(dim) + '.txt',targetPath='result/target_vec' + str(dim) + '.txt', dim=dim)
    cb.Process()





