## Over-smooth and over-fitting

图神经网络（GCN）中的过度平滑（over-smooth）问题：

随机网络层数的增加和迭代次数的增加，每个节点的隐藏表征会趋向于**收敛到同一个值**



同一个连通分量的节点的表征会趋向于收敛到同一个值



over-smooth的现象就是经过多次卷积之后，同一个连通分量内所有节点的特征都趋于一致了



如何解决：

首先看看节点是不是本身就非常一致，如果非常一致就不需要解决该问题，但是如果节点信息非常丰富，则需要解决该问题





## 如何解决over-smooth问题

图卷积会使同一个联通分量内的节点的表征都收敛到同一个值

1. 针对图卷积，在当前任务上，是否能够使用RNN+RandomWalk
2. 对图进行cut的预处理，图的联通分量越多，over-smooth越不明显





## DIRECT MULTI-HOP ATTENTION BASED GRAPH NEURAL NETWORKS

GNNs用self-attention机制已取得较好的结果。但目前的注意力机制只是考虑相连的节点，却不能利用图结构上下文信息的**多跳邻居（multi-hop neighbors）**



提出了Direct multi-hop attention based GNN

在注意力机制中加入多跳信息，从邻居节点扩展到非邻居节点，增加每层网络的感受野



reference:

[over-smooth and muliti-hops](https://blog.csdn.net/qq_38556984/article/details/110387724)













