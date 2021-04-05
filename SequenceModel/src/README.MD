**evalution文件夹中evalution.java**主要是对于学习到category向量进行评测，代码中有几部分需要进行对应：

- categoryPath 对应文件上一级data文件夹中的category.csv
- embeddingPath 为训练category向量的位置
- categoryGroundtruthPath1 对应文件上一级data文件夹中的各个评测方法的ground truth csv文件。
  具体MMR计算细节请看代码。

**HierarchicalModel文件夹中Skipgram.java**为基于**《Word2vec中的数学原理》** 负采样skip-gram模型进行实现。
调用skip-gram模型方法主要有以下参数：

- **trainList**：格式为[[target,context1#context2,...contextn],...,[]]
- negativeNum：负采样个数
- vecSize：训练向量大小
- contextPath/targetPath：向量存储路径
- learnRate：学习率

**HierarchicalModel文件夹中SequentialModel.java**为从sequence文件中获取训练数据**trainList**,调用skip-gram模型训练,代码中有几部分需要进行对应：

- trainPath：原始数据文件位置
- contextPath/targetPath：向量存储路径

