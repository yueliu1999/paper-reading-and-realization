## Note



### Abstract

在半监督表示学习、图结构信息经常使用

**graph Laplacian regularization**：图拉普拉斯正则化

给模型$f(X)$使用



GNN模型会直接将图结构A encoding到模型中

$f(A,X)$



认为对于GNN来说，graph Laplacian regularization没啥用



提出了一种graph Laplacian regularization的变种

**Propagation-regularization(P-reg)**

- 不仅仅infuses extra information into GNNs

  给GNNs提供新的信息

- 还具有和infinite-depth GCN拥有同样的能力





### Introduction

**节点分类：**

基于假设：

邻居节点可能共享相同的标签



使用了Laplacian regularization

$f(X) : R^{N\times F}->R^{N\times C}$



所以有了GNN和GCN。。

许多GNN都直接将图结构信息编码进模型

$f(A,X): (R^{N\times N},R^{N \times F})->R^{N \times C}$

A是邻接矩阵



**QUESTION：**

**是不是可以把GNN和传统的graph regularization结合在一起？**





答案是可以的：

GNN已经可以获取到Laplacian regularization可以提供的信息了

所以提出了一种新的graph regularization：

**Propagation-regularization**

是一个拉普拉斯正则化的变种



- 可以和infinite-depth GCN一样的能力

  - 每个节点可以获取到所有节点的信息

  - 但是比它更加**灵活**并且可以避免**over-smoothing**的问题
  - 计算量少





### Propagation Regularization

2-layer GCN model

$f_1(A,X) = \tilde A(\sigma(\tilde AXW_0))W_1$

其中

- $W_0\in R^{F \times H}$和$W_1 \in R^{H \times C}$是线性变化矩阵

- $H$是隐藏层的大小
- $\tilde A = D^{-1}A$是normalized 邻接矩阵，D是度矩阵



$P_{ij} = \frac{exp(Z_{ij})}{\sum^C_{k=1}exp(Z_{ik})}$是输出的softmax，$P$是预测所有节点的类后验概率



另外

propagating the output Z of $f_1$则是$Z' = \tilde AZ \in R^{N \times C}$

softmax为$Q_{ij} = \frac{exp(Z'_{ij})}{\sum_{k=1}^Cexp(Z_{ik}')}$

则**propagation-regularization(P-reg)**为

$L_{P-reg} = \frac{1}{N} \phi(Z, \tilde AZ)$

其中$\tilde AZ$是further propagated output of $f_1$

$\phi$是用于衡量$Z$和$\tilde AZ$之间的不同

- Mean Square Error

  $\frac{1}{2}\sum_{i=1}^N||(\tilde AZ)_i^T - (Z)_i^T||$

- Cross Entropy

  $\sum_{i=1}^N\sum_{j=1}^CP_{ij}log(Q_{ij})$

- KL Divergence

  $\sum_{i=1}^N\sum_{j=1}^CP_{ij}log(\frac{P_{ij}}{Q_{ij}})$



composition loss

$L = L_{cls} + \mu L_{P-reg} = -\frac{1}{M}\sum\sum Y_{ij} log(P_{ij}) + \mu\frac{1}{N}\phi(Z,\tilde AZ)$





### P-reg的理解

通过

- Laplacian Regularization
- Infinite-Dapth GCN





#### Squared-Error P-Reg 和Squared Laplacian Regularization是一致的

考虑使用均方差作为$\phi$

$L_{P-SE} = \frac{1}{N}\phi_{SE}(Z,\tilde AZ) = \frac{1}{2N}\sum_{i=1}^N||(\tilde AZ)_i^T-(Z)_i^T||^2_2$



图拉普拉斯$<Z,\Delta Z>$

其中$\Delta = D-A$是拉普拉斯矩阵，$<>$是内积



一种正则方程为

$<Z,RZ>:=<Z,r(\tilde \Delta)Z>$



Normalized Laplacian matrix

$\tilde \Delta = D^{-1}\Delta = D^{-1}(D-A) = I-D^{-1}A$



- 定理3.1

  均方差 P-reg 等价于正则

  R = $\tilde \Delta^T \tilde \Delta$

  等价于传统的正则，可以享受好属性







#### Minimizing P-Reg和Infinite-Dapth GCN一致



#### 首先分析无限深度GCN的行为



- 引理3.1

  将无穷GCN应用到图G(A,Z)中，

  则所有的Z都相等

  

  也就是：$\tilde Z = \tilde A^{∞}Z$

  则$\tilde Z_1 = ...=\tilde Z_N$

  $\tilde z_i$是$\tilde Z$的第i行

  引理3.1显示，无限GCN可以使**每个节点**捕获并表示**全图的信息**



- 引理3.2

  最小化squared-error P-reg

  也就是

  $\tilde Z = arg min_Z||\tilde AZ-Z||^2$





从3.1和3.2得知，无穷的图卷积和最小化P-reg是一样的

如果迭代的最小化下式，

$||\tilde AZ-Z||^2_F = ||(D^{-1}A-I)Z||^2_F = \sum_{i=1}^N||(\frac{1}{d_i}\sum_{j\in N(v_i)}z_j)-z_i||^2_2$



- 定理3.2

  当忽略所以的线性mappings，也是说

  $W_i = I$

  最小化P-reg（均方差、KL散度、交叉熵）

  是和用一个graph convolution infinitely 是一样的，也就是说

  $\tilde Z = \hat A^∞Z$，$\tilde Z = argmin_Z\phi(Z,\hat AZ)$











### 为什么P-Reg可以提升现有的GNNs

- 图正则化的好处
- 从深度GCN获得的好处





### 图正则化

#### P-reg可以提供额外的信息给GNN

节点越多，则知识越多？（可以添加节点？）

图正则化可以带来更多的信息



#### 图拉普拉斯正则化的局限性

不能提升



#### P-Reg如何解决图拉普拉斯正则化的局限性

- 提供了新的**监督信息**for GNNs

  拉普拉斯正则化

  $L_{lap} = \sum_{(i,j)\in \epsilon}||Z_i^T-Z_j^T||^2_2$

  是edge-centric

  

  $L_{P-reg} = \phi(Z, \hat AZ) = \sum_{i=1}^N\phi(Z_i^T,(\hat AZ)_i^T)$

  是node-centric

  使用了聚合的预测的邻居作为每个节点的目标

- 实验验证

  加入了mask







### 深度GCN

#### 更深的GCN提供更远节点的信息

对于K层的GCN，图G中任意节点X的影响分布$I_x$对于任意的节点x是等价于

k-step随机游走的分布，从x开始



#### Deep GCNs的局限性

**更多的层数不一定是更好的的**

层数更多，则节点越容易相似



会产生著名的over-smooth问题



计算上的问题：

一般来说对于所有的深度神经网络，梯度消失或者爆炸

- 虽然可以用残差进行连接，训练会变得慢和困难
- 变得更深，计算量大



#### P-Reg如何解决Deep GCNs的局限性

1. 可以有效地平衡**信息捕获**以及**over-smoothing**

   $\mu$很重要

   - 当$\mu=0$的时候GNN是vanilla model without P-reg
   - 当$\mu \to +∞$的时候，变成了一个depth-infinite GCN

   $\mu$是连续的，而非离散的

2. 实验证明：

   































