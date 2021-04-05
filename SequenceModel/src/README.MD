SequenceModel
evalution文件夹中evalution.java主要是对于学习到category进行评测，代码中有几部分需要进行对应：
1.categoryPath 对应文件上一级data文件夹中的category.csv
2.embeddingPath 为训练category向量的位置
3.categoryGroundtruthPath1 对应文件上一级data文件夹中的各个评测方法的ground truth csv文件。
具体细节请看代码。

HierarchicalModel文件夹中Skipgram.java为基于《Word2vec中的数学原理》 负采样skip-gram模型进行实现。
调用模型主要有以下参数：
trainList：格式为[[target,context1#context2,...contextn],...,[]]
negativeNum：负采样个数
vecSize：训练向量大小
contextPath/targetPath:向量存储路径
learnRate:学习率

HierarchicalModel文件夹中SequentialModel.java为从sequence文件中获取训练数据trainList,调用skip-gram模型训练,代码中有几部分需要进行对应：
trainPath:原始数据文件位置
contextPath/targetPath:向量存储路径