## Note

## Structural Deep Clustering Network



### Abstract

powerful representation

- GCN
- a dual self-supervised mechanism



关键词：

- 深度聚类
- 图卷积神经网络
- 自监督学习






### Introduction

idea：学习一个powerful representation



为了能够捕捉到结构信息，首先构建了一个KNN图

可以将GCN的表示和auto encoder的表示组合在一起



contribution

- SDCN，Structural Deep Clustering Network

  将GCN和auto encoder组合在了一起

  通过一个新的操作以及双子监督模块

- GCN模块二点over-smoothing问题会被解决

- sota



### proposed model

首先在数据上构建了KNN图，然后将数据和KNN图分别输入到auto encoder和GCN中，合成auto encoder和结构表示。用了一个双监督机制



#### KNN Graph

找到top-K相似的邻居，并为之建立边来连接其节点。有许多方法来计算相似度矩阵

1. Heat Kernel

   i和j可以被计算为

   $S_{ij} = e^{-\frac{||x_i-x_j||^2}{t}}$

   其中t是time parameter

   

2. Dot-product

   相似度

   $S_{ij} = x_j^Tx_i$

计算完相似度后选择top-k相似的点，去重建无向的k最近图

可以从非图数据中获取邻接矩阵



#### DNN模块

许多不同的学习表示的非监督的方法

- denoising autoencoder
- convolutional auto encoder
- LSTM encoder-decoder
- adversarial auto encoder



第l层的encoder part

$H^l = \phi(W^l_eH^{l-1}+b_e^l)$

输出重建

$\hat X = H^L$

MSE loss



### GCN模块

auto encoder是从**数据本身**学习有效的表示

忽视了样本之间的关系

使用GCN模块可以



卷积操作

$Z^l = \phi(\tilde D^{-\frac{1}{2}}\tilde A \tilde D^{-\frac{1}{2}}Z^{l-1}W^{l-1})$

其中$\tilde A = A+I$

D是度矩阵



$\tilde Z^{l-1} = (1-\epsilon)Z^{l-1}+\epsilon H^{l-1}$

结合GCN和auto encoder模块



### Dual self-supervised Module











