## note

## Data Augmentation for Graph Neural Networks

### Abstract

数据增强被广泛的应用

但是图很少：因为

- 图复杂
- 非欧式结构，可以很少的操作



神经边预测器

- 编码亲类结构，促进类内边
- 削弱类间边

GAug 







### Introduction

通过plausible variations of existing data without additional gt labels

最明显的方法是**adding和removing 节点和边**

对于节点分类任务：

- 节点

  - 增加节点在**标记**、**输入新节点的特征**以及**连接性**方面具有挑战
  - 减少节点只会减少数据的可用性

- 边

  的增减是最佳的数据增强策略

  哪条边呢？



**三种现有的方法**

- DropEdge

  在训练之前随机丢弃一些边

  不能享受增加边的好处

- ADAEDGE

  迭代的增加/丢弃边，当预测具有相同/不同的labels with high confidence

  很大程度依赖于数据的规模，容易导致错误传播

- BGCN

  迭代地训练一个具有GCN预测的选择性混合成员随机块模型，以生成多个去噪图，

  也会错误传播



**present work**

edge manipulation



通过：

- 移除noisy edges
- 添加missing edge

可以提升GNN的性能以及类内和类间的edges



主要的贡献：

- 提出了GAug，图数据增强

  - GAug-M

    用了一个边预测模块

  - GAug-O

    学习生成一个可能出现的边



### Other Related Work

一些相关的工作

- Graph Neural Networks
- Data Augmentation
- 

