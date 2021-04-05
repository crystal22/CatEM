### SequenceModel

1. data文件夹中**category.csv文件**描述的是**FourSquare数据集上**category之间层次关系，根据文件中的'parent'字段的信息可以构建一颗category层次树。
2. 其余的csv文件是根据**ITDL论文**中提到的六种评测方法生成的ground truth，每一个csv文件对应一种基于层次树的评测方法，文件中的每行代表为第一列的category根据此评测方法与其他category计算得到的值，格式为**category_id@value**。
3. **Hierarchical Tree**文件夹中pairwise_similarity.ipynb包含基于**FourSquare数据集上**生成ground truth和基于**Yelp数据集上**生成ground truth。

**代码或者数据有疑问请及时反馈。**

